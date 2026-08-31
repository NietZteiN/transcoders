"""Residual-stream activation extraction for the NLA read harness.

Captures layer-K residual-stream vectors from a target model and writes them in
the parquet format the NLA reference tooling expects (`activation_vector`:
d_model-wide float list), plus provenance columns of our own.

Design notes (and the paths we rejected):

* **Forward hook on `model.model.layers[K]`, not `output_hidden_states`.** The
  reference is ambiguous in one place and wrong in another: the repo README's
  quick-start snippet uses `hidden_states[20]` for layer 20, but
  `nla/datagen/extractors.py` (the code that actually produced the training
  data) hooks `layers[K]` and documents this as `hidden_states[K+1]`, "their
  index 0 is the embedding output". `examples/qwen7b_layer20_step4200.txt`
  agrees: "layer 20 residual stream (= HF hidden_states[21])". Hooking the
  block directly is what datagen does, so it cannot be off by one relative to
  how the NLA was trained. An off-by-one here would silently degrade every
  downstream number rather than fail loudly, which is why we match datagen
  exactly instead of picking an index.

* **Raw, unnormalized vectors.** The reference's standing invariant is that
  data-gen never normalizes: normalization happens at injection time
  (`injection_scale`) and at loss time (`mse_scale`), both read from the
  sidecar. We store raw and keep `act_norm` as a diagnostic — outlier norms
  (Qwen layer-20 sits in a ~100-170 band; chat-template newlines can spike to
  ~14k) decode unreliably and should not be over-interpreted.

* **Right padding and right truncation.** Llama-family tokenizers default to
  left-padding for generation; with left padding the `[:seq_len]` slice would
  silently return pad-position activations. Left truncation would mean
  `token_ids[0]` is not the document start, shifting every position index.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The reference's datagen skips positions below this: layer-K needs enough
# left-context to carry signal, and earlier positions "decode to noise"
# (nla/datagen/stage0_extract.py:_MIN_POSITION). Reads below it are flagged
# rather than dropped, so callers can measure the degradation instead of
# assuming it.
MIN_POSITION = 50


@dataclass
class ExtractionResult:
    """Per-text extraction output. `activations` is [n_positions, d_model] fp32."""

    text_id: str
    token_ids: list[int]
    token_strs: list[str]
    positions: list[int]
    activations: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class ActivationExtractor:
    """Capture layer-K residual-stream activations from a causal LM.

    The hook fires on the K-th decoder block and captures its output — the
    residual stream entering block K+1, post-MLP and post-residual-add.
    """

    def __init__(
        self,
        model_name: str,
        layer_index: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ):
        self.model_name = model_name
        self.layer_index = layer_index
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # See module docstring: both must be "right" or positions/activations lie.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device
        ).eval()

        cfg = self.model.config
        self.d_model = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        self.n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
        assert 0 <= layer_index < self.n_layers, (
            f"layer_index={layer_index} out of range for {model_name} "
            f"({self.n_layers} layers)"
        )

        self._captured: torch.Tensor | None = None
        self._handle = self._layers()[layer_index].register_forward_hook(self._hook)

    # ─── internals ────────────────────────────────────────────────────────

    def _layers(self) -> Any:
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

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        # .clone() not just .detach(): detach shares storage and the buffer can
        # be reused before we move it to CPU.
        self._captured = h.detach().clone()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ─── public API ───────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(
        self,
        text: str,
        positions: Sequence[int] | None = None,
        text_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Forward `text` once and return activations at `positions`.

        `positions=None` returns every token position (the full sequence), which
        is what the per-token replication test needs. Causal masking means the
        activation at position t is identical whether or not tokens after t
        exist, so a single full-sequence forward pass gives the same vectors as
        forwarding each truncated prefix separately — much cheaper, and exactly
        equivalent.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        ids = enc["input_ids"].to(self.model.device)
        seq_len = ids.shape[1]

        self._captured = None
        self.model(input_ids=ids)
        assert self._captured is not None, (
            f"forward hook on layers[{self.layer_index}] did not fire — "
            f"model structure changed?"
        )
        hidden = self._captured[0, :seq_len]  # [T, d]

        if positions is None:
            positions = list(range(seq_len))
        positions = [p for p in positions if 0 <= p < seq_len]

        acts = hidden[positions].float().cpu().numpy()
        token_ids = ids[0].tolist()
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)

        return ExtractionResult(
            text_id=text_id,
            token_ids=token_ids,
            token_strs=token_strs,
            positions=list(positions),
            activations=acts,
            meta=dict(meta or {}),
        )

    @torch.no_grad()
    def extract_chat(
        self,
        user_content: str,
        assistant_reply: str | None = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract over a chat-templated sequence.

        `assistant_reply=None` stops after the generation prompt (prompt-only
        reads). Passing a reply appends it verbatim so reads can be taken over
        the model's own generated tokens — this is how the reference's worked
        example is laid out (prompt 34 tokens + reply 67 = 101).
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if assistant_reply is not None:
            text = text + assistant_reply
        return self.extract(text, **kwargs)


def to_parquet(
    results: Iterable[ExtractionResult],
    path: str | Path,
    *,
    model_name: str,
    layer_index: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write extraction results to the reference's parquet contract.

    `activation_vector` is the column `nla_inference.py` reads. Everything else
    is our provenance, which the reference tooling ignores.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols: dict[str, list[Any]] = {
        "activation_vector": [],
        "text_id": [],
        "position": [],
        "token_id": [],
        "token_str": [],
        "act_norm": [],
        "below_min_position": [],
    }
    meta_keys: set[str] = set()
    for r in results:
        meta_keys.update(r.meta.keys())
    meta_keys_sorted = sorted(meta_keys)
    for k in meta_keys_sorted:
        cols[k] = []

    for r in results:
        for i, pos in enumerate(r.positions):
            v = r.activations[i]
            cols["activation_vector"].append(v.astype(np.float32).tolist())
            cols["text_id"].append(r.text_id)
            cols["position"].append(int(pos))
            cols["token_id"].append(int(r.token_ids[pos]))
            cols["token_str"].append(r.token_strs[pos])
            cols["act_norm"].append(float(np.linalg.norm(v)))
            cols["below_min_position"].append(bool(pos < MIN_POSITION))
            for k in meta_keys_sorted:
                cols[k].append(r.meta.get(k))

    table = pa.table(cols)
    table = table.replace_schema_metadata(
        {
            b"model_name": model_name.encode(),
            b"layer_index": str(layer_index).encode(),
            b"extraction": b"forward_hook_on_decoder_block_K (== hidden_states[K+1])",
            b"normalized": b"none",
            **{k.encode(): str(v).encode() for k, v in (extra or {}).items()},
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def sha256_file(path: str | Path) -> str:
    """Script/artifact provenance hash (transcoders/CLAUDE.md section 4)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
