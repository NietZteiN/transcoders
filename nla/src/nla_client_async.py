"""Async AV client — the throughput lever for N5.

The vendored `NLAClient.generate_batch` is a sequential for-loop (nla_inference.py:503-516);
its own docstring says "For real throughput, fire these in parallel via async httpx." At
concurrency 1 the sglang batcher sits idle between requests, which is why the measured cost is
3.83 s/read. N5 needs ~9.8k reads — 10.9 h sequential, ~3 h at concurrency 8.

CRITICAL: the injection path is reused VERBATIM from the vendored client (`_build_embeds`).
Injection is the one thing that must never be reimplemented — a subtly different embedding
lookup, scale or marker position produces plausible-looking but meaningless reads. Only the
HTTP call is replaced.

Gate before use (scripts/async_gate.py): 200 banked vectors, sequential vs async →
≥95% byte-identical text and max |Δrt_cos| < 0.01. Not bit-exactness: sglang's kernel
reductions vary with batch size even at temperature 0.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

_NLA_ROOT = Path(__file__).resolve().parent.parent
if str(_NLA_ROOT / "vendor" / "nla-repo") not in sys.path:
    sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))

from nla_inference import EXPLANATION_RE, NLAClient  # noqa: E402


class AsyncNLAClient:
    """Concurrent front-end over the same injection logic as NLAClient.

    Usage:
        cl = AsyncNLAClient(ckpt_dir, sglang_url, concurrency=8)
        texts = cl.generate_many([v1, v2, ...], temperature=0.0, max_new_tokens=180)
    """

    def __init__(self, checkpoint_dir, sglang_url: str = "http://localhost:30000",
                 concurrency: int = 8, timeout_s: float = 180.0):
        # the vendored client owns config/tokenizer/embedding table + _build_embeds
        self._sync = NLAClient(checkpoint_dir, sglang_url=sglang_url)
        self.url = sglang_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout_s = timeout_s

    # -- payload construction is delegated, never reimplemented -------------
    # Mirrors NLAClient._sglang_generate exactly: positional prompt arg, the (embeds, len)
    # tuple return, embeds-only body (never also input_ids — SGLang would mis-align logprob
    # bookkeeping against the forwarded embeds), and orjson's numpy fast path.
    def _payload(self, activation, prompt=None, **sampling) -> bytes:
        import orjson
        import torch
        v = activation if isinstance(activation, torch.Tensor) else torch.as_tensor(
            np.asarray(activation, dtype=np.float32))
        embeds_np, _ = self._sync._build_embeds(v, prompt)
        sp = {"temperature": 1.0, "max_new_tokens": 200, "skip_special_tokens": False}
        sp.update(sampling)
        return orjson.dumps({"input_embeds": embeds_np, "sampling_params": sp},
                            option=orjson.OPT_SERIALIZE_NUMPY)

    async def _one(self, client, sem, activation, extract: bool, **sampling) -> str:
        body = await asyncio.to_thread(self._payload, activation, None, **sampling)
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.post(f"{self.url}/generate", content=body,
                                          headers={"Content-Type": "application/json"},
                                          timeout=self.timeout_s)
                    r.raise_for_status()
                    out = r.json()
                    txt = (out[0] if isinstance(out, list) else out)["text"]
                    break
                except Exception:
                    if attempt == 2:
                        return ""
                    await asyncio.sleep(1.5 * (attempt + 1))
        if not extract:
            return txt
        m = EXPLANATION_RE.search(txt)          # same regex the sync client uses
        return m.group(1).strip() if m else txt  # no tags -> return raw, as the sync client does

    async def _run(self, activations, extract: bool, **sampling) -> list[str]:
        import httpx
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency + 4,
                              max_keepalive_connections=self.concurrency + 4)
        async with httpx.AsyncClient(limits=limits) as client:
            return await asyncio.gather(
                *(self._one(client, sem, v, extract, **sampling) for v in activations))

    def generate_many(self, activations, extract_explanation: bool = True, **sampling) -> list[str]:
        return asyncio.run(self._run(list(activations), extract_explanation, **sampling))

    # single-read parity with the sync client, for mixed call sites
    def generate(self, activation, **sampling) -> str:
        return self.generate_many([activation], **sampling)[0]
