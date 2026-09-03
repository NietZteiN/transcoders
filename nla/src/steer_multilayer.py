"""Close the KV bypass: write the correction at EVERY layer up to the target, during prefill.

THE DEFECT THIS ADDRESSES. `steer.ActivationSteerer` hooks one block. A write at layers[20],
position p, propagates only to layers 21-27 *for that position*. The K/V entries at layers 0-20
for p were computed before the hook fired and are left unedited — so every later token still
attends to the UN-EDITED decoy through the bottom 21 layers. That is a sufficient mechanical
explanation for B4's null which is independent of whether an item-level belief exists, and it has
never been tested.

BE PRECISE ABOUT HOW PARTIAL THIS IS (corrected 2026-09-03, see
log/nla-harness/2026-09-03_kv-bypass-claim-precision.md). The write DOES reach later tokens through
layers 21-27: those are downstream of the hook and are cached in corrected form. The bypass covers
21 of 28 layers -- 75% of the depth, not all of it. Do NOT write "the intervention never changes
what the model reads"; it is false, it contradicts the sentence before it, and a reviewer who
notices the top-7-layer path is entitled to discount the whole argument on the strength of it.

Writing at layers 0..L during prefill means the K/V cached at every one of those layers for p is
the corrected state. Later tokens then read the correction.

WHY PER-LAYER VECTORS, AND WHY THIS IS LICENSED. P0.1 returned HARD — cos(Δ₂₀, Δ_ℓ) = 0.277 over
layers 21-27 — so RE-USING the layer-20 vector elsewhere is not licensed. That verdict does not
forbid steering at other layers; it forbids steering with the wrong vector. V3 and V4 are
activation differences and are **defined at every layer**, so each layer gets its own Δ_ℓ derived
at that layer. No NLA is involved, which is also why V1/V2 cannot ride along: the AR is trained
at layer 20 only.

ENERGY IS THE CONFOUND TO CONTROL. Editing n layers at α each is n times the perturbation of
editing one, and edits compound — a change at layer ℓ alters the input to ℓ+1, which is then
edited again. P0.2 already showed what happens when coverage and magnitude move together:
widening to every reply position at unchanged α drove V1's parse rate to 0.100, destroying
generation rather than under-delivering. So `energy_matched=True` scales α by 1/sqrt(n_layers),
holding sum of squared coefficients constant against the single-layer baseline. Both settings are
run; the comparison is what separates coverage from magnitude.

alpha=0 must remain byte-identical to unsteered generation — the same identity gate steer.py
carries, and the first thing to check after any change here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from steer import ActivationSteerer  # reuse _layers() resolution so the two never drift


def model_dims(model: Any) -> tuple[int, int]:
    """(d_model, n_layers) for a plain causal LM or a multimodal wrapper.

    Gemma-3 loads as `Gemma3ForConditionalGeneration`, whose `Gemma3Config` carries neither
    `hidden_size` nor `num_hidden_layers` at the top level — both live under `text_config`.
    Reading them directly raises AttributeError, which is the good case; the bad case is code
    that falls back to a default and reports the wrong depth. `extract.ActivationExtractor`
    already resolves this the same way; this is the same rule for callers that hold a bare model.
    """
    cfg = model.config
    d = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
    n = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    return int(d), int(n)


@dataclass
class MultiLayerSpec:
    """Per-layer directions, one coefficient, one position rule.

    deltas: {layer_index: [d_model] direction}. Not required to be unit norm — each is normalized
            at write time, exactly as the single-layer hook does.
    alpha:  coefficient in units of the LOCAL activation norm at each layer, so one alpha means
            the same relative perturbation at every depth despite norms varying by ~2x across a
            network.
    prefill_only: the point of the exercise. The bypass is about what is cached during prefill;
            writing during decode as well would confound closing the bypass with steering the
            generation, which is axis A of the plan, not axis C.
    """
    deltas: dict[int, torch.Tensor]
    alpha: float = 1.0
    positions: str = "last_prompt"
    prefill_only: bool = True
    energy_matched: bool = False
    absolute: dict[int, torch.Tensor] | None = None
    """State REPLACEMENT: assign h_l := absolute[l] at the target position, ignoring `deltas`.

    This must be an assignment, not an addition, and the difference is not pedantic. Adding
    d_l = h_clean,l - h_obf,l at every layer does NOT yield h_clean at every layer: layer l+1
    receives the state layer l already corrected and adds d_{l+1} on top of it, so the error
    accumulates linearly with depth (verified: k*|d| at layer k). Only an absolute assignment is
    idempotent with respect to whatever propagated up from below.

    That makes this the one arm immune to the standing objection to all the others -- that
    per-layer directions derived at unsteered activations stop describing the state once the
    layer beneath has been edited. Here the position's state simply IS the clean run's state at
    every layer 0..L, so if the model still cannot answer, no vector-quality argument survives.

    It is a CEILING, not a deployable intervention: it requires the clean program. Alpha and
    energy matching do not apply and are refused rather than silently ignored."""

    def __post_init__(self) -> None:
        if self.absolute is not None and self.energy_matched:
            raise ValueError("absolute replacement is not scaled; energy_matched would be a "
                             "silent no-op on it")
        if self.absolute is not None and self.deltas:
            raise ValueError("pass either deltas or absolute, not both — one of them would be "
                             "silently ignored")
    _resolved: set[int] | None = field(default=None, repr=False)

    def layers_written(self) -> set[int]:
        return set(self.absolute) if self.absolute is not None else set(self.deltas)

    def effective_alpha(self) -> float:
        n = max(len(self.deltas), 1)
        return self.alpha / (n ** 0.5) if self.energy_matched else self.alpha


class MultiLayerSteerer:
    """Writes `alpha * ||h_l|| * delta_hat_l` at chosen positions, at every layer in `deltas`."""

    def __init__(self, model: Any, layers: Iterable[int]):
        self.model = model
        self._probe = ActivationSteerer.__new__(ActivationSteerer)   # borrow _layers() only
        self._probe.model = model
        blocks = self._probe._layers()
        self.n_layers = len(blocks)
        self.layers = sorted(set(layers))
        bad = [l for l in self.layers if not 0 <= l < self.n_layers]
        if bad:
            raise ValueError(f"layers out of range for this model ({self.n_layers}): {bad}")
        self.spec: MultiLayerSpec | None = None
        self._cursor: dict[int, int] = {}
        self.n_positions_written = 0
        self.n_hook_calls = 0
        self._handles = [blocks[l].register_forward_hook(self._make_hook(l)) for l in self.layers]

    def _make_hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                return output
            self.n_hook_calls += 1
            start = self._cursor.get(layer, 0)
            seq_len = hidden.shape[1]
            self._cursor[layer] = start + seq_len

            spec = self.spec
            if spec is None or layer not in spec.layers_written():
                return output
            # alpha=0 is inert for additive modes; a replacement has no alpha to zero, so it is
            # gated on `absolute` being present rather than on alpha.
            if spec.absolute is None and spec.alpha == 0.0:
                return output
            # Prefill is the first call at this layer; a decode step has seq_len == 1 and a
            # non-zero start. Restricting to prefill is what isolates the KV question.
            if spec.prefill_only and start != 0:
                return output

            if spec.positions == "last_prompt":
                local = [seq_len - 1] if start == 0 else []
            elif spec.positions == "all":
                local = list(range(seq_len))
            elif isinstance(spec._resolved, set):
                local = [p - start for p in spec._resolved if start <= p < start + seq_len]
            else:
                local = []
            if not local:
                return output

            idx = torch.tensor(sorted(local), device=hidden.device)
            h = hidden.index_select(1, idx)
            if spec.absolute is not None:
                tgt = spec.absolute[layer].to(device=hidden.device, dtype=hidden.dtype)
                upd = tgt.expand_as(h).clone()          # assign, do not add — see `absolute`
            else:
                d = spec.deltas[layer].to(device=hidden.device, dtype=torch.float32)
                dh = (d / d.norm().clamp_min(1e-12)).to(hidden.dtype)
                upd = h + spec.effective_alpha() * h.norm(dim=-1, keepdim=True) * dh
            hidden = hidden.index_copy(1, idx, upd)
            self.n_positions_written += len(local)
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def set_spec(self, spec: MultiLayerSpec | None, positions: Sequence[int] | None = None) -> None:
        self.reset()
        self.spec = spec
        if spec is not None and positions is not None:
            spec._resolved = {int(p) for p in positions}

    def reset(self) -> None:
        """MUST be called before each generation — the cursor is what distinguishes prefill."""
        self._cursor = {}
        self.n_positions_written = 0
        self.n_hook_calls = 0

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "MultiLayerSteerer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
