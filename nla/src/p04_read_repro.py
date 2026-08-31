"""P0.4-read — is the reproducibility floor a GENERATION problem or a whole-pipeline problem?

Everything measured on 2026-08-29 concerns generated text: 0.80-0.95 per-item agreement, unmoved
by deterministic kernels. Generation is autoregressive over ~1,100 steps with a KV cache, so one
flipped token cascades. **Reads are not like that.** An activation is a single forward pass, no
cache, no sampling, no cascade — so it may be perfectly reproducible, and if it is, the floor
applies only to behavioural claims and leaves `rt_cos`, `act_norm` and the P0.1 layer curves
untouched. That would narrow the caveat from "the whole programme" to "the causal arm", which is
worth ten minutes of GPU time to find out.

Two comparisons, both on the activations `steer_run.last_tok_act` feeds to V3/V4:
  WITHIN-PROCESS   extract every pair twice in one process, back to back
  CROSS-PROCESS    run this script twice in one job (same card) and diff the saved arrays
Reported as max |delta|, mean cosine, and the fraction of pairs that are bit-identical.

No steering, no generation. ~120 forward passes, minutes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from steer_run import SEED, TARGET_MODEL, build_user, load_pairs  # noqa: E402


def summarize(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: [n, d] matched activation matrices."""
    d = np.abs(a - b)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = num / np.clip(den, 1e-12, None)
    identical = int((d.max(axis=1) == 0).sum())
    return {
        "n": int(a.shape[0]),
        "max_abs_delta": float(d.max()),
        "mean_abs_delta": float(d.mean()),
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
        "n_bit_identical": identical,
        "frac_bit_identical": round(identical / a.shape[0], 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tag", required=True, help="replicate tag, e.g. R1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(_PROJ / "data/nla/p0/p04/readrepro"))
    args = ap.parse_args()

    import torch
    from extract import ActivationExtractor

    torch.manual_seed(SEED)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.limit, random.Random(SEED))
    ex = ActivationExtractor(TARGET_MODEL, args.layer, device=args.device)

    def sweep() -> np.ndarray:
        rows = []
        for p in pairs:
            for code, call in ((p["code_l0"], p["call_l0"]), (p["code_l1b"], p["call_l1b"])):
                rows.append(ex.extract_chat(build_user(code, call), None,
                                            text_id="x").activations[-1])
        return np.stack(rows)

    a, b = sweep(), sweep()
    ex.close()

    np.save(out / f"{args.tag}.npy", a)
    rep = {"experiment": "p0.4_read_reproducibility", "tag": args.tag, "layer": args.layer,
           "model": TARGET_MODEL, "n_pairs": len(pairs), "n_activations": int(a.shape[0]),
           "within_process": summarize(a, b),
           "finished_utc": datetime.now(timezone.utc).isoformat()}
    (out / f"{args.tag}_within.json").write_text(json.dumps(rep, indent=2))
    w = rep["within_process"]
    print(f"[read-repro {args.tag}] within-process: max|d| {w['max_abs_delta']:.3e} · "
          f"cos {w['mean_cosine']:.9f} · bit-identical {w['n_bit_identical']}/{w['n']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
