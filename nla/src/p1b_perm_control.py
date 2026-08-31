"""P1b — the CORRECTED permutation control.

WHY THIS EXISTS. The pre-registered control in `p1b_read_probe.py` ran ONE label permutation per
layer and required the MAXIMUM over 28 layers to stay below 0.60. That rule is mis-specified, and
the error is mine: a threshold appropriate for a single draw was applied to a max-over-28
statistic. The 28 observed draws have mean 0.4827 and sd 0.0616 — a null centred on 0.5 — so the
expected maximum of 28 such draws is around 0.5 + 2*sd = 0.62, and the observed 0.6533 is
unremarkable. A single draw cannot distinguish a leak from ordinary variance; a null DISTRIBUTION
can, which is what this does.

This is a correction to a badly-built control, NOT a renegotiation of the primary decision rule.
The primary verdict is unchanged by it: the tier probe sits at ceiling in 28 of 28 layers, so the
pre-registered UNINFORMATIVE branch applies on the substance whatever the control says.

Runs N_PERM permutations through the identical grouped-CV pipeline at the two contrast layers and
reports the null mean, sd, and the empirical p-value of the observed AUC.

CPU only.
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
from p1b_read_probe import SEED, grouped_cv_auc  # noqa: E402

N_PERM = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=str(PROJ / "data/nla/p0/p1b/acts.npz"))
    ap.add_argument("--layers", default="13,20")
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/perm_control.json"))
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    A, y, groups = z["acts"], z["y_tier"], z["groups"]
    rep = {"experiment": "p1b_permutation_control_corrected", "n_perm": N_PERM,
           "note": ("Corrects a mis-specified control: one draw per layer with a max-over-28 "
                    "threshold. See the script docstring."),
           "layers": {}}

    for L in [int(x) for x in args.layers.split(",")]:
        X = A[:, L, :]
        obs, _ = grouped_cv_auc(X, y, groups)
        null = []
        rng = np.random.default_rng(SEED + L)
        for i in range(N_PERM):
            a, _ = grouped_cv_auc(X, rng.permutation(y), groups)
            null.append(a)
            if (i + 1) % 50 == 0:
                print(f"[perm] L{L:02d} {i+1}/{N_PERM}", flush=True)
        null.sort()
        # One-sided: how often does a shuffled-label pipeline reach the observed AUC?
        ge = sum(1 for v in null if v >= obs)
        rep["layers"][f"L{L:02d}"] = {
            "observed_auc": round(obs, 4),
            "null_mean": round(st.mean(null), 4), "null_sd": round(st.pstdev(null), 4),
            "null_p05": round(null[int(0.05 * N_PERM)], 4),
            "null_p95": round(null[int(0.95 * N_PERM)], 4),
            "null_max": round(max(null), 4),
            "p_empirical": round((ge + 1) / (N_PERM + 1), 5),
            "leaks": bool(st.mean(null) > 0.60),
        }
        print(f"[perm] L{L:02d} observed {obs:.4f} · null {st.mean(null):.4f} "
              f"+/- {st.pstdev(null):.4f} (max {max(null):.4f}) · p={((ge+1)/(N_PERM+1)):.5f}",
              flush=True)

    rep["verdict"] = ("NO LEAK — null distributions are centred near 0.5"
                      if not any(v["leaks"] for v in rep["layers"].values())
                      else "LEAK — shuffled labels are decodable")
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
