"""Run the AV (verbalizer) in-process with transformers, instead of via an sglang server.

Why this exists. Every AV call site in this project talks to `sglang.launch_server`, and sglang
is not installed in any env on juno — it did not survive the cluster migration (job 376102 died
in 40 s on `ModuleNotFoundError: No module named 'sglang'`). Two ways out:

  (a) install sglang — a heavy dependency (flashinfer, a pinned torch) added to a working env
      purely to serve one model to one process on the same GPU;
  (b) call the AV directly, which is what this module does.

(b) was chosen. sglang buys throughput for many concurrent requests; Experiment W issues a few
hundred sequential single-vector reads on the same card the subject model already occupies, so
the server was pure overhead plus a process-liveness failure mode (`ServerGuard` exists precisely
to nurse it). Dropping it also removes the restart-race hazard flagged in the Phase-0 handoff.

**The injection path is NOT reimplemented.** `NLAClient._build_embeds` does the whole delicate
part — tokenize the actor template, embed, apply the architecture's embedding scale (Gemma
multiplies by sqrt(d)), rescale the activation to `injection_scale`, and overwrite the embedding
rows at the marked `injection_char` positions with a found-count assertion. This module calls
that method and then runs `model.generate(inputs_embeds=...)` in place of the HTTP POST. Any
divergence from the served path would be a silent correctness bug in exactly the place the
vendored code warns about, so the vendored code stays the single source of truth.

Equivalence to the sglang path is therefore: same embeds, same greedy decode, same explanation
extraction. What differs is throughput, and (usefully) determinism — no server-side batching to
make one read depend on which reads it was batched with.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

_NLA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))

from nla_inference import EXPLANATION_RE, NLAClient  # noqa: E402


class LocalAV:
    """`.generate(activation, **sampling) -> str`, matching NLAClient's public shape.

    Drop-in for the AV half of `capture_core.ReadEngine`: the only method the read path uses is
    `generate`, with the same keyword arguments.
    """

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        checkpoint_dir = Path(checkpoint_dir)
        # sglang_url is never contacted; the client is here for its tokenizer, its sidecar config
        # and _build_embeds. Embedding lookup stays on CPU as the vendored default recommends.
        self._c = NLAClient(checkpoint_dir, device="cpu")
        self.tokenizer = self._c.tokenizer
        self.cfg = self._c.cfg
        self.model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), torch_dtype=dtype, device_map=device).eval()
        self.device = self.model.device
        self.n_no_tags = 0

    @torch.inference_mode()
    def generate(self, activation, *, prompt: str | None = None,
                 extract_explanation: bool = True, **sampling) -> str:
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model} — wrong host/AV pair")
        embeds_np, _ = self._c._build_embeds(v, prompt)          # vendored injection, verbatim
        emb = torch.from_numpy(np.asarray(embeds_np)).to(self.device, self.model.dtype)[None]
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)

        temp = float(sampling.get("temperature", 0.0))
        gen = dict(max_new_tokens=int(sampling.get("max_new_tokens", 200)),
                   do_sample=temp > 0.0,
                   pad_token_id=self.tokenizer.pad_token_id or 0)
        if temp > 0.0:
            gen["temperature"] = temp
        # With inputs_embeds and no input_ids, generate() returns ONLY the new tokens, so the
        # decode below must not try to strip a prompt that is not in the output.
        out = self.model.generate(inputs_embeds=emb, attention_mask=attn, **gen)
        text = self.tokenizer.decode(out[0], skip_special_tokens=False)

        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        if m is None:
            # Same failure mode the vendored client warns about: truncation or model drift.
            # Counted rather than only printed, so a run can report how often it happened.
            self.n_no_tags += 1
            return text
        return m.group(1).strip()
