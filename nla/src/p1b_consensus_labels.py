"""Is the per-layer depth curve identifiable at n = 60, or is it label noise?

THE PROBLEM THIS ANSWERS. Two probes at the SAME position (`last_prompt`), with the same features,
folds and regularisation, produced opposite depth curves:

    2026-08-30a  0.6205 -> 0.7612, rising, argmax L27
    2026-08-30b  0.6942 -> 0.5737, falling, argmax L13

They differ only in which 8 of 60 correctness labels they used — the two label sets agree on
86.7%, exactly the reproducibility floor measured on 2026-08-29. Mean AUC across layers is nearly
identical (0.672 vs 0.654), so the LEVEL is stable and the SHAPE is not.

THE FIX, USING DATA ALREADY ON DISK. Ten independent unsteered generations of these same 60 items
exist from P0.4 and its replicates. A majority vote over ten draws is a much lower-variance label
than any single run. If the depth curve is real, denoised labels should sharpen it; if it is an
artefact of 8 coin flips, the consensus curve will not agree with either original.

Reported three ways: consensus labels on all items, the unanimous-only subset (where the label is
not in question at all), and the per-item vote distribution that says how much churn there is.

CPU only, minutes.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_read_probe import grouped_cv_auc  # noqa: E402

SOURCES = ["p04/L06", "p04/L13", "p04/L20", "p04/repro/A1", "p04/repro/A2", "p04/repro/B1",
           "p04/repro/D1", "p04/repro/D2", "p04/repro/E1", "p04/repro/E2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=str(PROJ / "data/nla/p0/p1b/pos_acts.npz"))
    ap.add_argument("--position", default="last_prompt")
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/consensus_labels.json"))
    args = ap.parse_args()

    votes: dict[str, list[int]] = {}
    used = []
    for src in SOURCES:
        p = PROJ / "data/nla/p0" / src / "baseline.jsonl"
        if not p.exists():
            continue
        used.append(src)
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                votes.setdefault(r["snippet_id"], []).append(int(bool(r["l1b_correct"])))

    z = np.load(args.acts, allow_pickle=True)
    positions = [str(x) for x in z["positions"]]
    pi = positions.index(args.position)
    acts, valid, groups = z["acts"], z["valid"], z["groups"]

    n_src = len(used)
    rows = []
    for i, sid in enumerate(groups):
        v = votes.get(str(sid), [])
        rows.append({"i": i, "sid": str(sid), "k": sum(v), "n": len(v)})
    dist = Counter(r["k"] for r in rows if r["n"] == n_src)

    rep = {"experiment": "p1b_consensus_label_stability", "sources": used, "n_draws": n_src,
           "position": args.position,
           "vote_distribution_k_of_n": {str(k): dist[k] for k in sorted(dist)},
           "n_unanimous": int(sum(c for k, c in dist.items() if k in (0, n_src))),
           "n_contested": int(sum(c for k, c in dist.items() if k not in (0, n_src)))}

    # consensus: majority of the draws; exact ties are dropped and counted rather than broken
    keep_c = [r for r in rows if r["n"] == n_src and r["k"] * 2 != n_src]
    keep_u = [r for r in rows if r["n"] == n_src and r["k"] in (0, n_src)]
    rep["n_ties_dropped"] = int(sum(1 for r in rows if r["n"] == n_src and r["k"] * 2 == n_src))

    for label, keep in (("consensus_all", keep_c), ("unanimous_only", keep_u)):
        idx = [r["i"] for r in keep if valid[r["i"], pi]]
        y = np.array([1 if next(x for x in keep if x["i"] == i)["k"] * 2 > n_src else 0
                      for i in idx])
        g = np.array([str(groups[i]) for i in idx])
        block = {"n": len(idx), "n_positive": int(y.sum())}
        if len(idx) < 30 or len(set(y.tolist())) < 2:
            block["note"] = "too few items or one class only"
            rep[label] = block
            continue
        curve = []
        for L in range(acts.shape[2]):
            a, _ = grouped_cv_auc(acts[idx, pi, L], y, g)
            curve.append({"layer": L, "auc": round(a, 4)})
        block["layers"] = curve
        block["argmax_layer"] = max(curve, key=lambda r: r["auc"])["layer"]
        block["argmax_auc"] = max(r["auc"] for r in curve)
        block["mean_auc"] = round(st.mean(r["auc"] for r in curve), 4)
        block["auc_L13"] = curve[13]["auc"]
        block["auc_L20"] = curve[20]["auc"]
        block["auc_L27"] = curve[27]["auc"]
        rep[label] = block
        print(f"[cons] {label:<16} n={block['n']:>3} pos={block['n_positive']:>3} "
              f"L13 {block['auc_L13']:.4f} L20 {block['auc_L20']:.4f} L27 {block['auc_L27']:.4f} "
              f"· argmax L{block['argmax_layer']} {block['argmax_auc']:.4f} "
              f"· mean {block['mean_auc']:.4f}", flush=True)

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("consensus_all", "unanimous_only")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
