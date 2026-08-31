"""S1 gate — does the async AV client produce the same reads as the sequential one?

Gate (from the plan): 200 banked vectors, sequential vs async →
  * ≥95% byte-identical text
  * max |Δrt_cos| < 0.01
  * ≥3× throughput
Byte-identity is the bar, not bit-exactness: sglang's kernel reductions vary with batch size
even at temperature 0, so a handful of divergent decodes is expected and acceptable — what
would NOT be acceptable is a systematic shift in rt_cos, which would mean the injected vector
differs.

Run (nla-mi, AV server up):
    python -m src.async_gate --n 200 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vendor" / "nla-repo"))

PROJ = _HERE.parent.parent
SEED = 20260724


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(PROJ / "data/nla/n5/async_gate.json"))
    args = ap.parse_args()

    import torch
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic
    from capture_core import align_reply, load_captures
    from nla_client_async import AsyncNLAClient
    from overnight_capture import build_tasks, build_user, task_key

    torch.manual_seed(SEED)
    caps = load_captures(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl")
    tasks = {task_key(t): t for t in build_tasks()[0]}

    # collect N banked (case, position) pairs, spread across cases
    picks: list[tuple[str, int]] = []
    for k, cap in caps.items():
        for rd in cap["reads"][:3]:
            picks.append((k, rd["position"]))
        if len(picks) >= args.n:
            break
    picks = picks[: args.n]

    ex = ActivationExtractor("Qwen/Qwen2.5-7B-Instruct", 20, device=args.device)
    ar = NLACritic(_HERE.parent / "data" / "checkpoints" / "ar", device=args.device)
    av_dir = _HERE.parent / "data" / "checkpoints" / "av"

    # gather the vectors once (shared by both paths)
    vecs, meta = [], []
    by_case: dict[str, list[int]] = {}
    for k, p in picks:
        by_case.setdefault(k, []).append(p)
    for k, ps in by_case.items():
        t = tasks[k]
        res = ex.extract_chat(build_user(t), caps[k].get("model_reply") or None)
        for p in ps:
            if p < len(res.positions):
                vecs.append(res.activations[p])
                meta.append((k, p))
    print(f"[S1] {len(vecs)} vectors from {len(by_case)} cases")

    sync = NLAClient(av_dir, sglang_url=args.sglang_url)
    t0 = time.time()
    seq_txt = [sync.generate(v, temperature=0.0, max_new_tokens=180) for v in vecs]
    t_seq = time.time() - t0

    acl = AsyncNLAClient(av_dir, sglang_url=args.sglang_url, concurrency=args.concurrency)
    t0 = time.time()
    asy_txt = acl.generate_many(vecs, temperature=0.0, max_new_tokens=180)
    t_asy = time.time() - t0

    identical = sum(1 for a, b in zip(seq_txt, asy_txt) if a == b)
    frac = identical / max(len(vecs), 1)
    d_rt = []
    for v, a, b in zip(vecs, seq_txt, asy_txt):
        _, ra = ar.score(a, v)
        _, rb = ar.score(b, v)
        d_rt.append(abs(float(ra) - float(rb)))
    speedup = t_seq / max(t_asy, 1e-9)
    ex.close()

    res = {"n": len(vecs), "identical": identical, "identical_frac": round(frac, 4),
           "max_abs_drt": round(max(d_rt), 5), "mean_abs_drt": round(float(np.mean(d_rt)), 5),
           "t_sequential_s": round(t_seq, 1), "t_async_s": round(t_asy, 1),
           "speedup": round(speedup, 2), "concurrency": args.concurrency,
           "reads_per_hour_async": round(3600 * len(vecs) / max(t_asy, 1e-9)),
           "gate": {"identical_ge_0.95": frac >= 0.95,
                    "max_drt_lt_0.01": max(d_rt) < 0.01,
                    "speedup_ge_3": speedup >= 3.0}}
    res["PASS"] = all(res["gate"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("\nS1 GATE:", "PASS" if res["PASS"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
