"""Activation steering: write a direction into the residual stream at chosen token positions.

This is the one mechanism the NLA release does not ship. The paper's protocol is:

    h  <-  h + alpha * ||h|| * delta / ||delta||

applied **at one token position, at the layer the NLA was trained on** (L20 for the Qwen pair).
Reported success is ~50% with "completions not always clean", so the control battery in
`steer_vectors.py` is not optional decoration — a claim rests on real-vs-foreign, not on
real-vs-nothing.

WHY A FORWARD HOOK ON `layers[K]`, AND THE ONE THING THAT MAKES IT SUBTLE.
We hook the same site `extract.ActivationExtractor` reads from, resolved through the same
path-probing helper, so a read and a write can never drift onto different tensors. The subtle
part is **position arithmetic under a KV cache**. During `generate()` the hook fires once on
prefill with `seq_len == len(prompt)`, then once per decode step with `seq_len == 1`. A hook
that targets absolute position *p* therefore cannot use the local index — it must know how many
tokens have already gone by. We keep an explicit cursor, require the caller to `reset()` before
each generation, and assert monotonic advance. Getting this wrong is silent: the edit lands on
the wrong token and the result looks like "steering doesn't work" rather than like a bug.

TWO RETURN SHAPES. Depending on config and version, a decoder block returns either a bare
tensor or a tuple whose first element is the hidden state. We rebuild whichever we were given.
Returning a bare tensor where a tuple was expected drops the attention/cache payload and fails
far from the cause — the same class of return-contract breakage that the G0
`apply_chat_template` patch had to fix.

Everything is deterministic: no RNG, no dropout (model is in eval), and alpha=0 must reproduce
the unsteered generation byte-for-byte. That identity is the first unit test and the only one
that really proves the hook is inert when it should be.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import torch

PositionSpec = Literal["last_prompt", "last_prefill", "all_reply", "all"] | Sequence[int]


@dataclass
class SteerSpec:
    """What to write, where, and how hard.

    delta: [d_model] direction. Not required to be unit norm — it is normalized here, so a
           caller can hand over a raw AR difference without pre-scaling.
    alpha: coefficient in units of the *local* activation norm. alpha=0 is a no-op by
           construction (see `apply_to`), which is what makes the identity test meaningful.
    positions: absolute token indices, or a symbolic spec resolved by the runner.
    """
    delta: torch.Tensor
    alpha: float
    positions: PositionSpec = "last_prompt"
    _resolved: set[int] | None = field(default=None, repr=False)

    def resolve(self, prompt_len: int, max_total: int | None = None) -> set[int]:
        """Turn a symbolic spec into absolute indices. Called once per generation."""
        if isinstance(self.positions, str):
            if self.positions == "last_prompt":
                out = {prompt_len - 1}
            elif self.positions == "last_prefill":
                # Resolved inside the hook, not here: the first hook call of a generation IS
                # the prefill, so its last index is the final prompt token. This lets a caller
                # steer that position WITHOUT knowing the prompt length in advance — which
                # matters when the prompt is built inside someone else's runner (the artifact's
                # ModelRunner), where re-tokenizing it here to recover the length would be a
                # second source of truth and would silently drift from the real one.
                out = set()
            elif self.positions == "all_reply":
                if max_total is None:
                    raise ValueError("'all_reply' needs max_total")
                out = set(range(prompt_len, max_total))
            elif self.positions == "all":
                if max_total is None:
                    raise ValueError("'all' needs max_total")
                out = set(range(max_total))
            else:
                raise ValueError(f"unknown position spec {self.positions!r}")
        else:
            out = {int(p) for p in self.positions}
        self._resolved = out
        return out


class ActivationSteerer:
    """Adds `alpha * ||h|| * delta_hat` to the residual stream at chosen absolute positions.

    Usage:
        st = ActivationSteerer(model, layer_index=20)
        st.set_spec(SteerSpec(delta=d, alpha=1.0, positions="last_prompt"), prompt_len=n)
        out = model.generate(...)          # hook fires; positions tracked across prefill+decode
        st.reset()                         # before the next generation
        st.close()
    """

    def __init__(self, model: Any, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self.spec: SteerSpec | None = None
        self._cursor = 0
        self.n_hook_calls = 0
        self.n_positions_written = 0
        self._targets: set[int] = set()
        layers = self._layers()
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer_index={layer_index} out of range ({len(layers)} layers)")
        self._handle = layers[layer_index].register_forward_hook(self._hook)

    # ── internals ────────────────────────────────────────────────────────────
    def _layers(self) -> Any:
        """Same resolution order as extract.ActivationExtractor._layers — keep in sync."""
        m = self.model
        # Gemma-3 loads as `Gemma3ForConditionalGeneration` — the multimodal wrapper — so its
        # decoder blocks are NOT at model.layers or model.model.layers. Both spellings are
        # listed because transformers exposes `language_model` at different depths across
        # versions. Order matters: the most-nested paths are tried last so a plain causal LM
        # still resolves on the first probe.
        #
        # extract.py and steer.py MUST list identical paths — a mismatch would put the read and
        # the write on different blocks and mislabel every depth result. nla/src/p04_gate.py
        # verifies they agree to 1e-3; run it after touching either file.
        for path in ("model.layers", "model.model.layers", "transformer.h",
                     "model.language_model.layers", "model.model.language_model.layers"):
            obj: Any = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"cannot locate decoder layers on {type(m).__name__}")

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output                       # not the shape we understand; leave untouched

        self.n_hook_calls += 1
        start = self._cursor
        seq_len = hidden.shape[1]
        self._cursor += seq_len

        spec = self.spec
        if spec is None or spec.alpha == 0.0 or (
                not self._targets and spec.positions != "last_prefill"):
            return output                       # alpha == 0 short-circuits before any math

        if spec.positions == "last_prefill":
            # cursor still at 0 => this is the prefill pass; target its final token.
            local = [seq_len - 1] if start == 0 else []
        else:
            local = [p - start for p in self._targets if start <= p < start + seq_len]
        if not local:
            return output

        # Normalize in fp32, THEN cast. Doing it in the activation dtype computes the norm of
        # a 3584-dim vector in bf16 (8 mantissa bits), which loses precision in the sum of
        # squares and makes the applied coefficient differ from the requested alpha. The AR
        # returns fp32 directions, so this costs nothing.
        d = spec.delta.to(device=hidden.device, dtype=torch.float32)
        d = (d / d.norm().clamp_min(1e-12)).to(hidden.dtype)
        idx = torch.tensor(sorted(local), device=hidden.device)
        # ||h|| is taken per (batch, position) so the edit scales with the local activation,
        # which is what makes one alpha transfer across positions with very different norms.
        h = hidden.index_select(1, idx)                                   # [b, k, d]
        norms = h.norm(dim=-1, keepdim=True)                              # [b, k, 1]
        hidden = hidden.index_copy(1, idx, h + spec.alpha * norms * d)
        self.n_positions_written += len(local)

        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    # ── public API ───────────────────────────────────────────────────────────
    def set_spec(self, spec: SteerSpec | None, prompt_len: int | None = None,
                 max_total: int | None = None) -> None:
        """Install a spec and resolve its positions. Also resets the cursor."""
        self.reset()
        self.spec = spec
        if spec is None:
            self._targets = set()
            return
        # "last_prefill" is the one symbolic spec that does NOT need prompt_len: it is
        # resolved inside the hook from the prefill call's own sequence length.
        if (prompt_len is None and isinstance(spec.positions, str)
                and spec.positions != "last_prefill"):
            raise ValueError("symbolic position specs need prompt_len")
        self._targets = (spec.resolve(prompt_len or 0, max_total)
                         if isinstance(spec.positions, str)
                         else {int(p) for p in spec.positions})

    def reset(self) -> None:
        """Rewind the position cursor. MUST be called before each new generation."""
        self._cursor = 0
        self.n_hook_calls = 0
        self.n_positions_written = 0

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ActivationSteerer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class PositionReplacer:
    """`h[p] <- ||h[p]|| * unit(v_p)` at chosen absolute positions, a DIFFERENT vector per position.

    Experiment W needs this and `ActivationSteerer` cannot express it: a `SteerSpec` carries one
    delta for all of its positions, whereas the NLA write-back gives every identifier span its own
    reconstructed target. (A `mode="replace"` flag on `SteerSpec` was written and tested first,
    then dropped -- it only covers the single-vector case, so keeping it would have left two
    replacement mechanisms in this module for one job.)

    Why replace rather than add, and why the norm is kept: the AR emits a whole state at one
    position, not a delta, so replacement is its native operation; and it is trained on
    `MSE = 2(1-cos)`, so its output norm carries no information and the LOCAL norm is the only
    defensible magnitude. That is also what removes alpha from Experiment W's design -- there is
    no coefficient to sweep, which is the point.

    Position arithmetic follows `ActivationSteerer` exactly: absolute indices over the whole
    sequence, tracked across the prefill call and each single-token decode call.
    """

    def __init__(self, model: Any, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self._targets: dict[int, torch.Tensor] = {}
        self._cursor = 0
        self.n_positions_written = 0
        layers = self._layers()
        assert 0 <= layer_index < len(layers), f"layer {layer_index} out of range"
        self._handle = layers[layer_index].register_forward_hook(self._hook)

    def _layers(self) -> Any:
        m = self.model
        for path in (("model", "layers"), ("model", "model", "layers"),
                     ("transformer", "h"), ("model", "language_model", "layers")):
            o = m
            try:
                for a in path:
                    o = getattr(o, a)
                return o
            except AttributeError:
                continue
        raise AttributeError("could not locate decoder layers")

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        start = self._cursor
        seq_len = hidden.shape[1]
        self._cursor += seq_len
        local = [(p - start, p) for p in self._targets if start <= p < start + seq_len]
        if not local:
            return output                       # empty targets => exact no-op, the identity test
        idx = torch.tensor([l for l, _ in local], device=hidden.device)
        h = hidden.index_select(1, idx)                                   # [b, k, d]
        # fp32 for the norm and the unit-normalisation, as in ActivationSteerer: a bf16 sum of
        # squares over 3840 dims loses enough precision to change the written magnitude.
        V = torch.stack([self._targets[p].to(torch.float32) for _, p in local])   # [k, d]
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        norms = h.to(torch.float32).norm(dim=-1, keepdim=True)            # [b, k, 1]
        new = (norms * V.to(hidden.device)[None]).to(hidden.dtype)
        hidden = hidden.index_copy(1, idx, new)
        self.n_positions_written += len(local)
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    def set_targets(self, targets: dict[int, torch.Tensor] | None) -> None:
        self._targets = dict(targets or {})
        self.reset()

    def reset(self) -> None:
        self._cursor = 0
        self.n_positions_written = 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "PositionReplacer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@torch.no_grad()
def generate_steered(model: Any, tokenizer: Any, input_ids: torch.Tensor,
                     steerer: ActivationSteerer, spec: SteerSpec | None,
                     max_new_tokens: int = 400) -> tuple[str, dict]:
    """Greedy generation with an optional steering spec installed.

    Greedy on purpose: with sampling, a steering effect and a sampling difference are not
    separable at n=1, and the alpha=0 identity test would be meaningless.
    """
    prompt_len = int(input_ids.shape[1])
    steerer.set_spec(spec, prompt_len=prompt_len,
                     max_total=prompt_len + max_new_tokens)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    diag = {"hook_calls": steerer.n_hook_calls,
            "positions_written": steerer.n_positions_written,
            "prompt_len": prompt_len,
            "new_tokens": int(out.shape[1]) - prompt_len}
    steerer.reset()
    return text, diag
