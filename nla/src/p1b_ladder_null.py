"""Selection-free scoring of the ladder result, plus the honest permutation null.

TWO PROBLEMS WITH THE FIRST PASS, BOTH MINE.
 1. `beats_length_by` used the ARGMAX over 28 layers. Taking a maximum over 28 correlated
    statistics and then comparing it to a single-number baseline inflates the gap. The
    selection-free statistic is the MEAN over layers, which is reported here as primary.
 2. The permutation null fixed the layer at the observed argmax, so it asked "is this layer's rho
    unusual?" when the claim was "is the BEST of 28 layers unusual?". The null must therefore be
    taken over the same max-over-layers statistic. Both null distributions are computed here.

The ladder verdict is re-derived from these rather than from the first pass.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402
from p1b_ladder_score import TIERS, ROUTES, load_tier  # noqa: E402

SEED = 20260724
N_PERM = 200
BEATS = 0.10


def curve_for(X, y, g):
    return [spearman(y, oof_ridge(X[:, L], y, g)) for L in range(X.shape[1])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p1b/ladder"))
    ap.add_argument("--perm-tiers", default="L1b,L2,L3")
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/ladder_null.json"))
    args = ap.parse_args()
    root = Path(args.root)

    rep = {"experiment": "p1b_ladder_selection_free", "seed": SEED, "n_perm": N_PERM,
           "primary_statistic": "mean rho over all 28 layers (no argmax selection)",
           "tiers": {}}

    perm_want = {t.strip() for t in args.perm_tiers.split(",")}
    for tier in TIERS:
        if not (root / tier / "acts.npy").exists():
            continue
        X, y, ln, g = load_tier(root, tier)
        base = spearman(y, oof_ridge(ln.reshape(-1, 1), y, g))
        c = curve_for(X, y, g)
        mean_rho, max_rho = st.mean(c), max(c)
        block = {
            "n": int(len(y)), "length_baseline_rho": round(base, 4),
            "mean_rho": round(mean_rho, 4), "max_rho": round(max_rho, 4),
            "beats_length_mean": round(mean_rho - base, 4),
            "beats_length_max": round(max_rho - base, 4),
        }
        if tier in perm_want:
            rng = np.random.default_rng(SEED)
            nm, nx = [], []
            for _ in range(N_PERM):
                cc = curve_for(X, rng.permutation(y), g)
                nm.append(st.mean(cc))
                nx.append(max(cc))
            block["null_mean_stat"] = {
                "mean": round(st.mean(nm), 4), "sd": round(st.pstdev(nm), 4),
                "max": round(max(nm), 4),
                "p": round((sum(1 for v in nm if v >= mean_rho) + 1) / (N_PERM + 1), 5)}
            block["null_max_stat"] = {
                "mean": round(st.mean(nx), 4), "sd": round(st.pstdev(nx), 4),
                "max": round(max(nx), 4),
                "p": round((sum(1 for v in nx if v >= max_rho) + 1) / (N_PERM + 1), 5)}
            print(f"[null] {tier:<4} mean-stat {mean_rho:+.4f} vs null "
                  f"{block['null_mean_stat']['mean']:+.4f}±{block['null_mean_stat']['sd']:.4f} "
                  f"p={block['null_mean_stat']['p']} · max-stat {max_rho:+.4f} vs null "
                  f"{block['null_max_stat']['mean']:+.4f}±{block['null_max_stat']['sd']:.4f} "
                  f"(null max {block['null_max_stat']['max']:+.4f}) "
                  f"p={block['null_max_stat']['p']}", flush=True)
        rep["tiers"][tier] = block
        print(f"[sel-free] {tier:<4} len-base {base:+.4f} · mean {mean_rho:+.4f} "
              f"(beats {block['beats_length_mean']:+.4f}) · max {max_rho:+.4f} "
              f"(beats {block['beats_length_max']:+.4f})", flush=True)

    ok = list(rep["tiers"])
    rep["routes"] = {r: {"tiers": [t for t in ts if t in ok],
                         "mean_beats_length_mean": round(
                             st.mean(rep["tiers"][t]["beats_length_mean"] for t in ts if t in ok), 4)}
                     for r, ts in ROUTES.items() if any(t in ok for t in ts)}
    top = max((rep["tiers"][t]["beats_length_mean"] for t in ok), default=0.0)
    rep["verdict"] = (
        "RELATIONAL TIERS CLEAR THE BAR on the selection-free statistic"
        if top >= BEATS else
        f"NO TIER CLEARS THE BAR selection-free (max {top:+.4f})")
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["routes"], indent=2))
    print(f"[sel-free] {rep['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
