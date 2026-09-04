"""Unit tests for the activation steering write hook. CPU-only, no GPU required.

Uses Qwen2.5-0.5B-Instruct — the same Qwen2 decoder architecture as the 7B subject model, so
the hook code path under test is identical; only the width and depth differ. That keeps these
runnable while the GPUs are busy.

The tests check the properties that matter rather than proxies. "Was the hook called" proves
nothing; what has to hold is that the edit lands on the intended token, leaves its neighbours
alone, survives the prefill/decode cache boundary, and is exactly inert at alpha=0. Two of this
project's earlier harness bugs each produced a confidently wrong conclusion because the test
measured a proxy, so these assert placement and value directly.

Run:
  /data/jvl210002/conda_envs/nla-mi/bin/python -m pytest nla/tests/test_steer.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from steer import ActivationSteerer, PositionReplacer, SteerSpec, generate_steered  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 8               # mid-stack of the 0.5B (24 layers); stands in for L20 of the 7B
SEED = 20260724
PROMPT = "Write one sentence about sorting algorithms."


@pytest.fixture(scope="module")
def lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  tokenize=True, add_generation_prompt=True,
                                  return_dict=False)
    return model, tok, torch.tensor([ids])


@pytest.fixture
def steerer(lm):
    model, _, _ = lm
    st = ActivationSteerer(model, LAYER)
    yield st
    st.close()


def _delta(model, scale: float = 1.0) -> torch.Tensor:
    d_model = model.config.hidden_size
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(d_model, generator=g) * scale


# ── T1: the identity test ────────────────────────────────────────────────────
def test_alpha_zero_is_byte_identical(lm, steerer):
    """alpha=0 must reproduce the unsteered generation exactly.

    This is the test that proves the hook is inert when it should be. If it fails, every
    steering number downstream is confounded by the mere presence of the hook.
    """
    model, tok, ids = lm
    base, _ = generate_steered(model, tok, ids, steerer, None, max_new_tokens=24)
    zero, diag = generate_steered(
        model, tok, ids, steerer,
        SteerSpec(delta=_delta(model), alpha=0.0, positions="last_prompt"),
        max_new_tokens=24)
    assert zero == base, f"alpha=0 changed the output:\n base={base!r}\n zero={zero!r}"
    assert diag["positions_written"] == 0


def test_no_spec_is_byte_identical_to_no_hook(lm):
    """Installing the hook at all must not perturb generation."""
    model, tok, ids = lm
    with torch.no_grad():
        raw = model.generate(ids, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    raw_text = tok.decode(raw[0][ids.shape[1]:], skip_special_tokens=True)
    st = ActivationSteerer(model, LAYER)
    try:
        hooked, _ = generate_steered(model, tok, ids, st, None, max_new_tokens=24)
    finally:
        st.close()
    assert hooked == raw_text


# ── T2: the edit lands where it is aimed, with the right value ───────────────
def test_edit_lands_at_the_target_position_only(lm, steerer):
    """Verify placement and value directly: read layer output with and without the spec."""
    model, _, ids = lm
    captured: list[torch.Tensor] = []
    layers = steerer._layers()
    h = layers[LAYER].register_forward_hook(
        lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
    try:
        steerer.set_spec(None)
        with torch.no_grad():
            model(ids)
        clean = captured[-1]

        target = int(ids.shape[1]) - 3
        delta = _delta(model)
        steerer.set_spec(SteerSpec(delta=delta, alpha=0.5, positions=[target]),
                         prompt_len=int(ids.shape[1]))
        with torch.no_grad():
            model(ids)
        steered = captured[-1]
    finally:
        h.remove()

    d_hat = delta / delta.norm()
    expected = clean[0, target] + 0.5 * clean[0, target].norm() * d_hat
    assert torch.allclose(steered[0, target], expected, atol=1e-4), "value at target is wrong"

    # every other position untouched — this is the "leaves p-1 alone" property
    mask = torch.ones(clean.shape[1], dtype=torch.bool)
    mask[target] = False
    assert torch.allclose(steered[0, mask], clean[0, mask], atol=1e-6), \
        "positions other than the target were modified"


def test_alpha_scales_the_edit_linearly(lm, steerer):
    model, _, ids = lm
    captured: list[torch.Tensor] = []
    layers = steerer._layers()
    h = layers[LAYER].register_forward_hook(
        lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
    try:
        target = int(ids.shape[1]) - 1
        delta = _delta(model)
        outs = {}
        for a in (0.0, 0.5, 1.0):
            steerer.set_spec(SteerSpec(delta=delta, alpha=a, positions=[target]),
                             prompt_len=int(ids.shape[1]))
            with torch.no_grad():
                model(ids)
            outs[a] = captured[-1][0, target].clone()
    finally:
        h.remove()
    step_half = outs[0.5] - outs[0.0]
    step_full = outs[1.0] - outs[0.0]
    assert torch.allclose(step_full, 2 * step_half, atol=1e-4)


# ── T3: the KV-cache boundary — the bug-prone part ───────────────────────────
def test_cursor_spans_prefill_and_decode(lm, steerer):
    """A reply-side position is only reachable if the cursor survives into decode steps."""
    model, tok, ids = lm
    prompt_len = int(ids.shape[1])
    reply_target = prompt_len + 3
    _, diag = generate_steered(
        model, tok, ids, steerer,
        SteerSpec(delta=_delta(model), alpha=1.0, positions=[reply_target]),
        max_new_tokens=12)
    assert diag["positions_written"] == 1, (
        "a decode-step position was never written — the prefill/decode cursor is wrong")
    assert diag["hook_calls"] > 1, "expected one prefill call plus per-token decode calls"


def test_position_beyond_generation_is_never_written(lm, steerer):
    model, tok, ids = lm
    far = int(ids.shape[1]) + 10_000
    _, diag = generate_steered(
        model, tok, ids, steerer,
        SteerSpec(delta=_delta(model), alpha=1.0, positions=[far]), max_new_tokens=8)
    assert diag["positions_written"] == 0


def test_reset_rewinds_the_cursor(lm, steerer):
    """Without a reset the second generation would target absolute positions that never recur."""
    model, tok, ids = lm
    spec = SteerSpec(delta=_delta(model), alpha=1.0, positions="last_prompt")
    _, d1 = generate_steered(model, tok, ids, steerer, spec, max_new_tokens=8)
    _, d2 = generate_steered(model, tok, ids, steerer, spec, max_new_tokens=8)
    assert d1["positions_written"] == d2["positions_written"] == 1


# ── T4: behavioural sanity and shape contracts ───────────────────────────────
def test_large_alpha_changes_the_output(lm, steerer):
    """A big enough edit must move the text — otherwise the hook is writing into the void."""
    model, tok, ids = lm
    base, _ = generate_steered(model, tok, ids, steerer, None, max_new_tokens=20)
    steered, diag = generate_steered(
        model, tok, ids, steerer,
        SteerSpec(delta=_delta(model), alpha=8.0, positions="all"),
        max_new_tokens=20)
    assert diag["positions_written"] > 0
    assert steered != base, "a large edit at every position left the output unchanged"


def test_output_tuple_shape_is_preserved(lm, steerer):
    """Rebuilding the block's return value must not drop the cache/attention payload."""
    model, _, ids = lm
    seen: list[type] = []
    layers = steerer._layers()
    h = layers[LAYER].register_forward_hook(lambda m, i, o: seen.append(type(o)))
    try:
        steerer.set_spec(SteerSpec(delta=_delta(model), alpha=1.0, positions=[2]),
                         prompt_len=int(ids.shape[1]))
        with torch.no_grad():
            model(ids, use_cache=True)
    finally:
        h.remove()
    assert seen, "downstream hook never fired"
    assert all(t is seen[0] for t in seen)


def test_resolve_position_specs():
    d = torch.ones(4)
    assert SteerSpec(d, 1.0, "last_prompt").resolve(10) == {9}
    assert SteerSpec(d, 1.0, "all_reply").resolve(10, 13) == {10, 11, 12}
    assert SteerSpec(d, 1.0, "all").resolve(10, 12) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    assert SteerSpec(d, 1.0, [3, 5]).resolve(10) == {3, 5}
    with pytest.raises(ValueError):
        SteerSpec(d, 1.0, "all_reply").resolve(10)


def test_direction_normalized_in_fp32_not_activation_dtype():
    """A bf16 norm over 3584 dims loses precision; the applied coefficient must match alpha.

    Checked by comparing the realized edit magnitude against alpha*||h|| — in bf16 the
    normalization error shows up directly as a wrong step size.
    """
    import torch as t
    from steer import ActivationSteerer, SteerSpec

    class Tiny(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = t.nn.Module()
            self.model.layers = t.nn.ModuleList([t.nn.Identity()])

    m = Tiny()
    st = ActivationSteerer(m, 0)
    try:
        g = t.Generator().manual_seed(0)
        d = t.randn(3584, generator=g)
        st.set_spec(SteerSpec(delta=d, alpha=1.0, positions=[0]), prompt_len=1)
        h = t.randn(1, 1, 3584, generator=g).to(t.bfloat16)
        out = st._hook(None, None, h)
        step = (out - h).float().norm().item()
        want = h[0, 0].float().norm().item()
        assert abs(step - want) / want < 0.02, f"edit magnitude {step:.3f} vs alpha*||h||={want:.3f}"
    finally:
        st.close()


# ── PositionReplacer ────────────────────────────────────────────────────────────────────────
# Deliberately fixture-free: a toy module rather than the 0.5B, so these run even when the test
# LM is not in the offline cache (it currently is not on juno). What is under test is position
# arithmetic and the written value, neither of which needs a real transformer.

class _Identity(torch.nn.Module):
    def forward(self, x):                      # noqa: D102
        return x


class _ToyLM(torch.nn.Module):
    """Minimal `model.layers` shape so PositionReplacer._layers() resolves."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Identity()])

    def forward(self, x):                      # noqa: D102
        return self.model.layers[0](x)


@pytest.fixture()
def toy():
    torch.manual_seed(SEED)
    return _ToyLM(), torch.randn(1, 6, 8)


def test_replacer_empty_targets_is_exact_noop(toy):
    """The identity test: no targets must leave the tensor byte-identical, not merely close."""
    m, h = toy
    r = PositionReplacer(m, 0)
    r.set_targets(None)
    assert torch.equal(m(h.clone()), h)
    r.close()


def test_replacer_writes_norm_matched_value_per_position(toy):
    """Each position gets ITS OWN direction at ITS OWN norm — the property SteerSpec cannot express.

    Asserts the value, not that a hook fired: the AR's output norm is meaningless (trained on
    MSE = 2(1-cos)), so writing it unnormalised would be a silent magnitude error.
    """
    m, h = toy
    v2, v4 = torch.randn(8), torch.randn(8)
    r = PositionReplacer(m, 0)
    r.set_targets({2: v2, 4: v4})
    out = m(h.clone())

    for p, v in ((2, v2), (4, v4)):
        assert torch.allclose(out[0, p], h[0, p].norm() * v / v.norm(), atol=1e-6)
    assert torch.allclose(out[0, [2, 4]].norm(dim=-1), h[0, [2, 4]].norm(dim=-1), atol=1e-5)
    # The two positions must not collapse to one shared direction.
    u2, u4 = out[0, 2] / out[0, 2].norm(), out[0, 4] / out[0, 4].norm()
    assert not torch.allclose(u2, u4, atol=1e-3)
    assert torch.equal(out[0, [0, 1, 3, 5]], h[0, [0, 1, 3, 5]])
    assert r.n_positions_written == 2
    r.close()


def test_replacer_tracks_positions_across_the_decode_boundary(toy):
    """Position 7 exists only after prefill(6) + one decode step — the cache-boundary bug class."""
    m, h = toy
    r = PositionReplacer(m, 0)
    r.set_targets({7: torch.randn(8)})
    m(h.clone())                                   # prefill, positions 0..5 — nothing written
    assert r.n_positions_written == 0
    m(torch.randn(1, 1, 8))                        # position 6
    assert r.n_positions_written == 0
    m(torch.randn(1, 1, 8))                        # position 7 — the target
    assert r.n_positions_written == 1
    r.close()


def test_no_grad_decorator_still_belongs_to_generate_steered():
    """Regression: PositionReplacer was first inserted BETWEEN @torch.no_grad() and the function
    it decorates, silently transferring the decorator to the class and leaving generate_steered
    running with gradients enabled — in code paths banked experiments call. A FutureWarning was
    the only symptom.
    """
    import inspect
    import steer as _steer
    src = inspect.getsource(_steer)
    i, j = src.index("class PositionReplacer"), src.index("def generate_steered")
    assert "@torch.no_grad()" in src[j - 40:j], "generate_steered lost its no_grad decorator"
    assert "@torch.no_grad()" not in src[i - 40:i], "decorator re-attached to the class"
