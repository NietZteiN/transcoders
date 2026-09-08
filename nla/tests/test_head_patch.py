"""CPU tests for nla/src/head_patch.py on a Gemma-shaped toy LM — no checkpoint, no GPU.

The toy mirrors the two facts the hooks rely on: decoder blocks live at
`model.language_model.layers` (the Gemma-3 path), and the attention module reshapes heads into a
contiguous `[B, T, H*D]` tensor immediately before a positional call to `o_proj`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import head_patch as hp  # noqa: E402
from head_patch import (AttentionKnockout, Component, ComponentPatcher, ShapeMismatch,  # noqa: E402
                        build_layer_mask, components, draw_layer_matched, draw_uniform,
                        half_of, rank_on_other_half)

SEED = 20260724
D_MODEL, H, D, N_LAYERS = 24, 4, 6, 3


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(D_MODEL, H * D)
        self.k = torch.nn.Linear(D_MODEL, H * D)
        self.v = torch.nn.Linear(D_MODEL, H * D)
        self.o_proj = torch.nn.Linear(H * D, D_MODEL)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        B, T, _ = hidden_states.shape
        q = self.q(hidden_states).view(B, T, H, D).transpose(1, 2)
        k = self.k(hidden_states).view(B, T, H, D).transpose(1, 2)
        v = self.v(hidden_states).view(B, T, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                             is_causal=attention_mask is None)
        out = out.transpose(1, 2).reshape(B, T, H * D).contiguous()      # as modeling_gemma3:380
        return self.o_proj(out), None


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D_MODEL, 2 * D_MODEL), torch.nn.GELU(),
                                       torch.nn.Linear(2 * D_MODEL, D_MODEL))

    def forward(self, hidden_states, attention_mask=None):
        a, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        h = hidden_states + a
        return h + self.mlp(h)


class _ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([_Layer() for _ in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer(x)
        return x


ALL_LAYERS = tuple(range(N_LAYERS))


@pytest.fixture()
def toy():
    torch.manual_seed(SEED)
    lm = _ToyLM().eval()
    x = torch.randn(1, 7, D_MODEL)
    return lm, x


def _patcher(lm, layers=ALL_LAYERS):
    return ComponentPatcher(lm, layers=layers, n_heads=H, head_dim=D)


def _record(patcher, lm, x):
    cache = patcher.record()
    with torch.no_grad():
        y = lm(x)
    patcher.stop_recording()
    return cache, y


# 1 ─ patching a run with its own cache is bit-identical (the SELF identity)
def test_self_patch_is_identity(toy):
    lm, x = toy
    with _patcher(lm) as p:
        cache, y = _record(p, lm, x)
        assert cache.T == 7 and set(cache.heads) == set(ALL_LAYERS) and set(cache.mlps) == set(ALL_LAYERS)
        p.set_patches({c: cache.get(c) for c in components(ALL_LAYERS, H)})
        with torch.no_grad():
            y2 = lm(x)
        p.assert_written()
        assert p.n_components_written == N_LAYERS * (H + 1)
        assert torch.equal(y, y2)


# 2 ─ a head patch touches only its own columns of the o_proj input
def test_head_patch_touches_only_its_columns(toy):
    lm, x = toy
    seen = {}
    blk = lm.model.language_model.layers[1]
    with _patcher(lm) as p:
        cache, _ = _record(p, lm, x)
        # a second pre-hook, registered AFTER the patcher's, sees the patched input
        hdl = blk.self_attn.o_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].clone()))
        with torch.no_grad():
            lm(x)
        base = seen["x"]
        src = torch.full((7, D), 3.0)
        p.set_patches({Component(1, 2): src})
        with torch.no_grad():
            lm(x)
        p.assert_written()
        got = seen["x"]
        hdl.remove()
    assert torch.equal(got[0, :, 2 * D:3 * D], src)
    keep = torch.ones(H * D, dtype=torch.bool); keep[2 * D:3 * D] = False
    assert torch.equal(got[0][:, keep], base[0][:, keep])
    assert torch.equal(base[0].view(7, H, D), cache.heads[1])        # record == module input


# 3 ─ an MLP patch replaces the module output exactly
def test_mlp_patch_replaces_output(toy):
    lm, x = toy
    seen = {}
    blk = lm.model.language_model.layers[0]
    with _patcher(lm) as p:
        src = torch.randn(7, D_MODEL)
        p.set_patches({Component(0, hp.MLP): src})
        hdl = blk.mlp.register_forward_hook(lambda m, a, o: seen.__setitem__("o", o.clone()))
        with torch.no_grad():
            lm(x)
        hdl.remove()
        p.assert_written()
    assert torch.equal(seen["o"][0], src)


# 4 ─ the ALL identity: patch every component from S into U, and U reproduces S wherever the
#     inputs agree (the write differs at one position only, like the L32 span write)
def test_all_from_S_reproduces_S_off_the_written_position(toy):
    lm, xu = toy
    xs = xu.clone(); xs[0, 2] += 5.0
    with _patcher(lm) as p:
        cS, yS = _record(p, lm, xs)
        p.set_patches({c: cS.get(c) for c in components(ALL_LAYERS, H)})
        with torch.no_grad():
            yU_all = lm(xu)
        p.assert_written()
    keep = torch.ones(7, dtype=torch.bool); keep[2] = False
    assert torch.equal(yU_all[0][keep], yS[0][keep])
    assert not torch.equal(yU_all[0][2], yS[0][2])        # only the written position differs


# 5 ─ a cache from another length raises instead of misaligning
def test_length_mismatch_raises(toy):
    lm, x = toy
    with _patcher(lm) as p:
        cache, _ = _record(p, lm, x)
        p.set_patches({Component(0, 1): cache.get(Component(0, 1))})
        with pytest.raises(ShapeMismatch):
            with torch.no_grad():
                lm(torch.randn(1, 5, D_MODEL))
        p.set_patches(None)
        cache2 = p.record()
        with torch.no_grad():
            lm(x)
        with pytest.raises(ShapeMismatch):                 # same open cache, different T
            with torch.no_grad():
                lm(torch.randn(1, 6, D_MODEL))
        p.stop_recording()
    assert cache2.T == 7


# 6 ─ installed hooks with no patches are an exact no-op; recording and patching exclude each other
def test_no_patch_is_noop_and_modes_exclusive(toy):
    lm, x = toy
    with torch.no_grad():
        y0 = lm(x)
    with _patcher(lm) as p:
        p.set_patches(None)
        with torch.no_grad():
            y1 = lm(x)
        assert torch.equal(y0, y1) and p.n_components_written == 0
        p.record()
        with pytest.raises(RuntimeError):
            p.set_patches({Component(0, 0): torch.zeros(7, D)})
        p.stop_recording()
        p.set_patches({Component(0, 0): torch.zeros(7, D)})
        with pytest.raises(RuntimeError):
            p.record()
        with pytest.raises(KeyError):
            p.set_patches({Component(N_LAYERS + 3, 0): torch.zeros(7, D)})
    with torch.no_grad():
        y2 = lm(x)
    assert torch.equal(y0, y2)                             # hooks removed on close


# 7 ─ the mask reproduces transformers' documented sliding-window example
def test_build_layer_mask_matches_hf_docstring():
    got = build_layer_mask(5, sliding=True, window=3)[0, 0].int().tolist()
    assert got == [[1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0],
                   [1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1]]
    g = build_layer_mask(5, sliding=False)[0, 0]
    assert torch.equal(g, torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    assert sorted(L for L in hp.GLOBAL_LAYERS if L > hp.WRITE_LAYER) == [35, 41, 47]


# 8 ─ the knockout zeroes exactly the knocked keys for exactly one head
def test_knockout_zeroes_only_target_head_and_keys(toy):
    lm, x = toy
    attn = lm.model.language_model.layers[1].self_attn
    T, keys, head = 7, [2, 3], 1
    with torch.no_grad():
        q = attn.q(x).view(1, T, H, D).transpose(1, 2)
        k = attn.k(x).view(1, T, H, D).transpose(1, 2)
        v = attn.v(x).view(1, T, H, D).transpose(1, 2)
    ko = AttentionKnockout(lm, layers=(1,), n_heads=H, window=3, global_fn=lambda L: False)
    ko.set(1, head, keys)
    m = ko.mask_for(1, T)
    assert m.shape == (1, H, T, T) and m.dtype == torch.bool
    other = [h for h in range(H) if h != head]
    assert torch.equal(m[0, other], build_layer_mask(T, True, 3).expand(1, H - 1, T, T)[0])
    assert not m[0, head, :, keys].any()
    assert m[0, head, 4, 4] and m[0, head, 5, 4]           # target head still sees other keys
    with torch.no_grad():
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        scores = (q @ k.transpose(-1, -2)) / D ** 0.5
        scores = scores.masked_fill(~m, float("-inf"))
        w = scores.softmax(-1)
        ref = w @ v
    assert torch.allclose(out, ref, atol=1e-5)
    assert float(w[0, head, :, keys].abs().max()) == 0.0
    assert float(w[0, other][:, :, keys].abs().max()) > 0.0
    # the hook path. With a window wider than T the sliding mask IS the causal mask, so the
    # mask-only reference must equal the mask-free run up to kernel choice (none on CPU).
    ko.window = T + 1
    ko.clear()
    with torch.no_grad():
        y_free = lm(x)
        assert ko.n_masks_applied == 0
        ko.set(1, None, None, materialize_only=True)
        y_ref = lm(x)
        assert ko.n_masks_applied == 1
        ko.set(1, head, keys)
        y_ko = lm(x)
    ko.close()
    assert torch.allclose(y_free, y_ref, atol=1e-5)
    assert not torch.allclose(y_ref, y_ko, atol=1e-5)
    with pytest.raises(ValueError):
        ko.set(1, None, keys)


# 9 ─ the component grid
def test_components_grid_and_names():
    cs = components(hp.SWEEP_LAYERS)
    assert len(cs) == 255 and len(set(cs)) == 255
    assert [c.name for c in cs[:2]] == ["L33H0", "L33H1"] and cs[16].name == "L33M"
    for c in cs:
        assert Component.from_name(c.name) == c
    assert Component(35, hp.MLP).kind == "mlp" and Component(35, 0).kind == "head"


# 10 ─ split-half selection and the nulls
def test_split_half_ranking_and_nulls():
    names = [c.name for c in components((33, 34), H)]
    ids = [f"s{i}" for i in range(12)]
    assert {half_of(s) for s in ids} == {0, 1}
    rng = np.random.default_rng(SEED)
    per_item = {s: {n: float(rng.normal()) for n in names} for s in ids}
    for s in ids:
        per_item[s]["L34H1"] = 10.0 if half_of(s) == 1 else -10.0
    assert rank_on_other_half(per_item, eval_half=0, names=names)[0] == "L34H1"
    assert rank_on_other_half(per_item, eval_half=1, names=names)[-1] == "L34H1"
    pool = components(hp.SWEEP_LAYERS)
    top = [Component(47, 3), Component(47, hp.MLP), Component(35, 0), Component(41, 7)]
    a = draw_layer_matched(top, pool, np.random.default_rng(1))
    b = draw_layer_matched(top, pool, np.random.default_rng(1))
    assert a == b and sorted(c.layer for c in a) == sorted(c.layer for c in top)
    assert len(set(a)) == len(a)
    u = draw_uniform(pool, 32, np.random.default_rng(2))
    assert len(set(u)) == 32 and all(c in pool for c in u)
