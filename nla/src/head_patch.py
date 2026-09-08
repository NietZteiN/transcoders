"""Per-head / per-MLP activation patching and attention-read knockout for H-W31.

Pre-registered in `log/nla-harness/2026-09-07_head-mediation-prereg.md`. Three interventions on
the layers DOWNSTREAM of the NLA write site (Gemma-3-12B: write at layer 32, sweep 33..47):

  * **head patch** — overwrite head h's slice of the tensor entering `layers[L].self_attn.o_proj`
    with the same slice recorded from another run of the same sequence;
  * **MLP patch** — overwrite the output of `layers[L].mlp` likewise;
  * **read knockout** — leave the run intact but forbid head h at layer L from attending to a
    chosen set of key positions (the identifier spans the write landed on).

WHY THESE SITES (verified against transformers 5.12.1, `models/gemma3/modeling_gemma3.py`).
The attention output is `attn_output.reshape(B, T, -1).contiguous()` immediately before
`self.o_proj(attn_output)` (lines 380-381), so the tensor a pre-hook on `o_proj` receives is
`[B, T, n_heads*head_dim]`, fresh, contiguous, head h at columns `h*head_dim:(h+1)*head_dim`.
That is the ONLY well-defined per-head site: `post_attention_layernorm` is applied to the SUM over
heads, so per-head contributions to the residual stream are not separable after it. For the MLP,
`layers[L].mlp` output goes through `post_feedforward_layernorm` (per-position RMSNorm) and then
into the residual; replacing the pre-norm tensor at position t reproduces exactly the post-norm
contribution the source run had at t, so hooking the module boundary loses nothing.

NO POSITION CURSOR, ON PURPOSE. `steer.PositionReplacer` tracks absolute positions across prefill
and decode because it serves `generate()`. Every forward in H-W31 is one teacher-forced pass over
the full sequence, so a cache is valid only for the exact T it was recorded at; the hooks RAISE on
a length mismatch instead of tracking a cursor that could misalign silently. `ComponentPatcher`
therefore does not support `generate()`, and says so.

ATTENTION MASKS UNDER SDPA (the trap that ruled out reusing obtune's `attention_knockout`).
For a single unpadded sequence, `Gemma3Model` hands global layers `attention_mask=None` (the
`is_causal` fast path) and sliding layers `None` when T < 1024 or a **bool** `[1,1,T,T]` mask when
T >= 1024 (`masking_utils.sdpa_mask`: True = may attend). Adding a float bias to that — obtune's
approach, written for an additive float mask — would promote it to float with allowed=1.0 /
disallowed=0.0 and destroy the causal structure. So the knockout hook ignores the incoming mask
and builds its own bool `[1, n_heads, T, T]` mask from the documented rules (`kv <= q`, and for
sliding layers `kv > q - window`), then clears the knocked columns for one query head. With a mask
present, HF expands K/V to all query heads (`repeat_kv`), so head h of the mask is query head h.
A materialised mask moves that one layer off the flash kernel; callers must therefore score
against a **mask-only reference** (`materialize_only=True`) rather than against the mask-free run.

Model-free and CPU-importable: `nla/tests/test_head_patch.py` exercises every hook on a toy LM.
"""
from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

import torch
from torch import nn

from steer import resolve_layers

# Gemma-3-12B-it geometry (config.json of google/gemma-3-12b-it; head_dim is the class default).
N_HEADS = 16
HEAD_DIM = 256
D_MODEL = 3840
N_LAYERS = 48
WRITE_LAYER = 32
SWEEP_LAYERS = tuple(range(WRITE_LAYER + 1, N_LAYERS))     # 33..47
SMOKE_LAYERS = (45, 46, 47)
SLIDING_WINDOW = 1024
MLP = -1                                                    # Component.head value for the MLP


def is_global_layer(layer: int, every: int = 6) -> bool:
    """`layer_types[i] == "full_attention"` iff `(i + 1) % 6 == 0` (configuration_gemma3.py)."""
    return (layer + 1) % every == 0


GLOBAL_LAYERS = frozenset(L for L in range(N_LAYERS) if is_global_layer(L))   # 5, 11, ..., 47


def decoder_layers(model: Any) -> Any:
    """Same probe list as extract.py / steer.py, so read, write and patch land on one block list."""
    return resolve_layers(model)


@dataclass(frozen=True, order=True)
class Component:
    """One patchable unit: attention head (layer, 0..n_heads-1) or the MLP (layer, MLP=-1)."""
    layer: int
    head: int

    @property
    def kind(self) -> str:
        return "mlp" if self.head == MLP else "head"

    @property
    def name(self) -> str:
        return f"L{self.layer}M" if self.head == MLP else f"L{self.layer}H{self.head}"

    @classmethod
    def from_name(cls, s: str) -> "Component":
        assert s.startswith("L"), s
        if s.endswith("M"):
            return cls(int(s[1:-1]), MLP)
        L, h = s[1:].split("H")
        return cls(int(L), int(h))


def components(layers: Sequence[int], n_heads: int = N_HEADS) -> list[Component]:
    """`n_heads` heads then the MLP, per layer, in layer order: 255 for the 33..47 sweep."""
    out: list[Component] = []
    for L in layers:
        out.extend(Component(L, h) for h in range(n_heads))
        out.append(Component(L, MLP))
    return out


class ShapeMismatch(RuntimeError):
    """A patch source does not match the tensor it would overwrite. Raised, never skipped."""


@dataclass
class ComponentCache:
    """Per-layer recordings from one forward: heads `[T, n_heads, head_dim]`, MLPs `[T, d_model]`.

    Kept in the module's own dtype (bf16 on the GPU run) exactly as the hook received it — no
    fp32 round trip, so patching a run with its own cache is bit-identical (the SELF identity).
    """
    T: int = -1
    heads: dict[int, torch.Tensor] = field(default_factory=dict)
    mlps: dict[int, torch.Tensor] = field(default_factory=dict)

    def get(self, c: Component) -> torch.Tensor:
        if c.head == MLP:
            return self.mlps[c.layer]                      # [T, d_model]
        return self.heads[c.layer][:, c.head, :]           # [T, head_dim]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   list(self.heads.values()) + list(self.mlps.values()))


class ComponentPatcher:
    """Record or overwrite per-head `o_proj` inputs and MLP outputs on chosen layers.

    Recording and patching are mutually exclusive per forward, so a recorded tensor never aliases
    a patched one. `n_components_written` counts the components actually overwritten in the last
    forward — the caller asserts it equals `len(patches)` after every pass, the same habit
    `arm_guard` enforces for write positions: an arm that silently patched nothing must not score.
    """

    def __init__(self, model: Any, layers: Sequence[int] = SWEEP_LAYERS,
                 n_heads: int = N_HEADS, head_dim: int = HEAD_DIM):
        self.model = model
        self.layers = tuple(int(L) for L in layers)
        self.n_heads, self.head_dim = n_heads, head_dim
        self._cache: ComponentCache | None = None
        self._patches: dict[int, dict[int, torch.Tensor]] = {}    # L -> {head|MLP: src}
        self.n_patches = 0
        self.n_components_written = 0
        self._handles: list[Any] = []
        blocks = decoder_layers(model)
        for L in self.layers:
            assert 0 <= L < len(blocks), f"layer {L} out of range ({len(blocks)})"
            self._handles.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(
                self._pre_o_proj(L)))
            self._handles.append(blocks[L].mlp.register_forward_hook(self._post_mlp(L)))

    # ── recording ─────────────────────────────────────────────────────────────
    def record(self) -> ComponentCache:
        """The next forward fills and returns this cache. Refuses while patches are set."""
        if self._patches:
            raise RuntimeError("cannot record while patches are set — call set_patches(None)")
        self._cache = ComponentCache()
        return self._cache

    def stop_recording(self) -> None:
        self._cache = None

    # ── patching ──────────────────────────────────────────────────────────────
    def set_patches(self, patches: dict[Component, torch.Tensor] | None) -> None:
        """`{Component: source}` with sources `[T, head_dim]` / `[T, d_model]`. None = exact no-op."""
        if self._cache is not None and patches:
            raise RuntimeError("cannot patch while recording — call stop_recording()")
        self._patches = {}
        for c, src in (patches or {}).items():
            if c.layer not in self.layers:
                raise KeyError(f"{c.name}: layer {c.layer} has no hook (hooked: {self.layers})")
            self._patches.setdefault(c.layer, {})[c.head] = src
        self.n_patches = sum(len(v) for v in self._patches.values())
        self.reset()

    def reset(self) -> None:
        self.n_components_written = 0

    def assert_written(self) -> None:
        if self.n_components_written != self.n_patches:
            raise RuntimeError(f"patched {self.n_components_written} of {self.n_patches} "
                               "components — an arm that did not write must not score")

    # ── hooks ─────────────────────────────────────────────────────────────────
    def _check(self, x: torch.Tensor, where: str) -> int:
        if x.dim() != 3 or x.shape[0] != 1:
            raise ShapeMismatch(f"{where}: expected [1, T, *], got {tuple(x.shape)} — "
                                "single unpadded sequence only")
        T = int(x.shape[1])
        if self._cache is not None:
            if self._cache.T == -1:
                self._cache.T = T
            elif self._cache.T != T:
                raise ShapeMismatch(f"{where}: T={T} but cache was opened at T={self._cache.T}")
        return T

    def _pre_o_proj(self, L: int) -> Callable:
        H, D = self.n_heads, self.head_dim

        def hook(_module: nn.Module, args: tuple) -> tuple | None:
            x = args[0]
            T = self._check(x, f"L{L} o_proj")
            if x.shape[-1] != H * D:
                raise ShapeMismatch(f"L{L} o_proj: width {x.shape[-1]} != {H}*{D}")
            if self._cache is not None:
                self._cache.heads[L] = x.detach()[0].clone(
                    memory_format=torch.contiguous_format).view(T, H, D)
            per = self._patches.get(L)
            if not per:
                return None
            x = x.clone(memory_format=torch.contiguous_format)
            xv = x.view(1, T, H, D)                       # shares storage with x
            for h, src in per.items():
                if h == MLP:
                    continue
                if tuple(src.shape) != (T, D):
                    raise ShapeMismatch(f"L{L}H{h}: source {tuple(src.shape)} != {(T, D)}")
                xv[0, :, h, :] = src.to(x.dtype)
                self.n_components_written += 1
            return (x,) + tuple(args[1:])
        return hook

    def _post_mlp(self, L: int) -> Callable:
        def hook(_module: nn.Module, _args: tuple, out: torch.Tensor) -> torch.Tensor | None:
            T = self._check(out, f"L{L} mlp")
            if self._cache is not None:
                self._cache.mlps[L] = out.detach()[0].clone(
                    memory_format=torch.contiguous_format)
            src = self._patches.get(L, {}).get(MLP)
            if src is None:
                return None
            if tuple(src.shape) != (T, out.shape[-1]):
                raise ShapeMismatch(f"L{L}M: source {tuple(src.shape)} != {(T, out.shape[-1])}")
            self.n_components_written += 1
            return src.to(out.dtype)[None]
        return hook

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ComponentPatcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ── attention-read knockout ───────────────────────────────────────────────────

def build_layer_mask(T: int, sliding: bool, window: int = SLIDING_WINDOW,
                     device: Any = None) -> torch.Tensor:
    """Bool `[1, 1, T, T]`, True = query q may attend key k. Mirrors `masking_utils`:
    `causal_mask_function` is `kv <= q`; `sliding_window_overlay` adds `kv > q - window`."""
    q = torch.arange(T, device=device)[:, None]
    k = torch.arange(T, device=device)[None, :]
    m = k <= q
    if sliding:
        m = m & (k > q - window)
    return m[None, None]


class AttentionKnockout:
    """Forbid one query head at one layer from attending to chosen key positions.

    `set(layer, head, keys)` arms the hook; `set(layer, None, None, materialize_only=True)` installs
    the same mask with nothing knocked out — the reference every knockout score is taken against,
    because the materialised mask alone changes the sdpa kernel at that layer.
    """

    def __init__(self, model: Any, layers: Sequence[int] = SWEEP_LAYERS, n_heads: int = N_HEADS,
                 window: int = SLIDING_WINDOW,
                 global_fn: Callable[[int], bool] = is_global_layer):
        self.layers = tuple(int(L) for L in layers)
        self.n_heads, self.window, self.global_fn = n_heads, window, global_fn
        self.layer: int | None = None
        self.head: int | None = None
        self.keys: list[int] = []
        self.materialize_only = False
        self.n_masks_applied = 0
        self._handles: list[Any] = []
        blocks = decoder_layers(model)
        for L in self.layers:
            self._handles.append(blocks[L].self_attn.register_forward_pre_hook(
                self._hook(L), with_kwargs=True))

    def set(self, layer: int | None, head: int | None, keys: Sequence[int] | None,
            materialize_only: bool = False) -> None:
        if layer is not None and layer not in self.layers:
            raise KeyError(f"layer {layer} has no knockout hook (hooked: {self.layers})")
        if layer is not None and not materialize_only:
            if head is None or not keys:
                raise ValueError("a knockout needs a head and a non-empty key set")
            if not 0 <= head < self.n_heads:
                raise ValueError(f"head {head} out of range")
        self.layer, self.head = layer, head
        self.keys = sorted(int(k) for k in (keys or []))
        self.materialize_only = materialize_only
        self.n_masks_applied = 0

    def clear(self) -> None:
        self.set(None, None, None)

    def mask_for(self, L: int, T: int, device: Any = None) -> torch.Tensor:
        m = build_layer_mask(T, sliding=not self.global_fn(L), window=self.window,
                             device=device).expand(1, self.n_heads, T, T).clone()
        if not self.materialize_only:
            if max(self.keys) >= T:
                raise ShapeMismatch(f"knockout key {max(self.keys)} >= T={T}")
            m[0, self.head, :, self.keys] = False
        if not bool(m.any(-1).all()):
            raise RuntimeError("knockout emptied a query row — softmax would be NaN")
        return m

    def _hook(self, L: int) -> Callable:
        def hook(_module: nn.Module, args: tuple, kwargs: dict) -> tuple | None:
            if L != self.layer:
                return None
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            if hs.dim() != 3 or hs.shape[0] != 1:
                raise ShapeMismatch(f"L{L} knockout: expected [1, T, d], got {tuple(hs.shape)}")
            m = self.mask_for(L, int(hs.shape[1]), device=hs.device)
            self.n_masks_applied += 1
            if "attention_mask" in kwargs or len(args) < 3:
                return (args, {**kwargs, "attention_mask": m})
            new_args = list(args); new_args[2] = m          # positional (hidden, pos_emb, mask)
            return (tuple(new_args), kwargs)
        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "AttentionKnockout":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ── joint-pass selection helpers (pure; tested on CPU) ────────────────────────

def half_of(snippet_id: str) -> int:
    """Split-half membership, 0/1. Fixed by the id, not by row order, so it cannot drift."""
    return zlib.crc32(snippet_id.encode()) & 1


def rank_on_other_half(per_item: dict[str, dict[str, float]], eval_half: int,
                       names: Sequence[str]) -> list[str]:
    """Components in descending mean score over the items NOT in `eval_half`.

    The top-k evaluated on half H is chosen on half 1-H, so "how many heads suffice" is never
    selected and scored on the same items. Ties broken by name for reproducibility.
    """
    pool = [sid for sid in per_item if half_of(sid) != eval_half]
    if not pool:
        raise ValueError(f"no items in the selection half for eval_half={eval_half}")
    mean = {n: float(np.mean([per_item[sid][n] for sid in pool])) for n in names}
    return sorted(names, key=lambda n: (-mean[n], n))


def draw_uniform(pool: Sequence[Component], k: int, rng: np.random.Generator) -> list[Component]:
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idx]


def draw_layer_matched(top: Sequence[Component], pool: Sequence[Component],
                       rng: np.random.Generator) -> list[Component]:
    """Random set with the same per-layer multiset as `top` (a null for "it's the depth, not the
    head"). Drawn without replacement within each layer; heads and MLPs are both eligible."""
    need = Counter(c.layer for c in top)
    out: list[Component] = []
    for L, n in sorted(need.items()):
        cand = [c for c in pool if c.layer == L]
        if n > len(cand):
            raise ValueError(f"layer {L}: need {n} components, pool has {len(cand)}")
        out.extend(draw_uniform(cand, n, rng))
    return out
