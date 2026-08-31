"""N7b(ii) — the pre-registered baseline the judge has to beat.

    ar_cos(item) = cos( AR.reconstruct(read), AR.reconstruct(window) )

Both texts are pushed through the SAME reconstructor the NLA pair ships with, into the L20
residual space they were trained on, and compared there. This is free (no judge, no external model,
no API) and it is not a strawman: if a dense text-similarity measure in activation space separates
real pairs from shuffled ones as well as an 8B judge does, then the judge is buying nothing and N7's
alignment score should just BE this number. Pre-registering it was the point — it stops "the judge
works" from meaning "the judge beats chance".

BLINDING applies identically: this opens `pack.jsonl` and nothing else.

Runs in `nla-mi`, needs ~16 GB on one GPU (AR backbone only; no AV server, no sglang).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vendor" / "nla-repo"))

PROJ = _HERE.parent.parent
ALLOWLIST = {"item_id", "read", "window"}
SEED = 20260724
MAX_CHARS = 1200          # windows are already capped here; reads are shorter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(PROJ / "data/nla/n7/2026-08-07/pack.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pack)]
    for r in rows[:200]:
        extra = set(r) - ALLOWLIST
        if extra:
            raise SystemExit(f"BLINDING VIOLATION: pack row carries {extra}; refusing to run.")
    if args.limit:
        rows = rows[: args.limit]
    out_p = Path(args.out) if args.out else Path(args.pack).parent / "ar_baseline.jsonl"

    done = set()
    if out_p.exists():
        for l in open(out_p):
            try:
                done.add(json.loads(l)["item_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["item_id"] not in done]
    print(f"[N7-AR] {len(rows)} items · {len(done)} done · {len(todo)} to go", flush=True)
    if not todo:
        return 0

    import torch
    torch.manual_seed(SEED)
    from nla_inference import NLACritic
    ar = NLACritic(_HERE.parent / "data" / "checkpoints" / "ar", device=args.device)

    # reconstruct() is one forward pass per string; cache by text since a window is reused by
    # every read that falls inside it (stride-8 strips hit the same sentence repeatedly).
    cache: dict[str, torch.Tensor] = {}

    def vec(text: str) -> torch.Tensor:
        t = text[:MAX_CHARS]
        v = cache.get(t)
        if v is None:
            with torch.no_grad():
                v = ar.reconstruct(t)
            v = v / v.norm().clamp_min(1e-12)
            if len(cache) < 20000:
                cache[t] = v
        return v

    t0 = 0.0
    with open(out_p, "a") as f:
        t0 = time.time()
        for i, r in enumerate(todo):
            try:
                c = float(vec(r["read"]) @ vec(r["window"]))
                f.write(json.dumps({"item_id": r["item_id"], "ar_cos": round(c, 5)}) + "\n")
            except Exception as e:
                f.write(json.dumps({"item_id": r["item_id"], "ar_cos": None,
                                    "error": repr(e)[:160]}) + "\n")
            if (i + 1) % 500 == 0:
                f.flush()
                rate = (i + 1) / (time.time() - t0)
                print(f"[N7-AR] {i+1}/{len(todo)} · {rate:.1f}/s · cache {len(cache)} "
                      f"· ETA {(len(todo)-i-1)/max(rate,1e-9)/60:.0f} min", flush=True)

    (out_p.parent / "ar_baseline_manifest.json").write_text(json.dumps({
        "experiment": "n7b_ar_baseline", "seed": SEED, "argv": sys.argv,
        "n_items": len(rows), "cache_size": len(cache),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"[N7-AR] done · wrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
