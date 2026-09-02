"""Does the residual stream encode dispatcher complexity DIRECTLY?

The 2026-08-31 mechanism test inferred this: the residual stream beat reply length on L2 but added
only +0.0245 over a baseline that knew the code's static shape, so the correctness signal it
carries is static complexity. That is an inference from a correctness probe. This measures the
thing head-on — regress the residual stream on `n_dispatcher_spans` itself.

THE CONTROL THAT MATTERS. Span count correlates with code size, and code size is trivially
readable from the token sequence — a probe that predicts spans by counting characters has
demonstrated nothing. The test is therefore incremental over `code_chars` and `n_lines`, exactly
as the correctness probe was made incremental over reply length.

Reported on L2 (which has 2-12 spans) with L1b as a floor (0 spans everywhere, so nothing to find
and the probe should be unable to find it).

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
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402
from p1b_l2_mechanism import stimuli  # noqa: E402
from p1b_ladder import repetition_features  # noqa: E402

SEED, N_PERM, BAR = 20260724, 200, 0.10




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="L2,L3,L1b")
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p1b/ladder"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/span_probe.json"))
    args = ap.parse_args()
    root = Path(args.root)

    rep = {"experiment": "p1b_dispatcher_span_probe", "seed": SEED, "bar": BAR,
           "target": "n_dispatcher_spans", "tiers": {}}

    for tier in [t.strip() for t in args.tiers.split(",")]:
        d = root / tier
        if not (d / "acts.npy").exists():
            continue
        acts = np.load(d / "acts.npy")
        sids = json.loads((d / "items.json").read_text())
        stim = stimuli(tier)
        keep = [i for i, s in enumerate(sids) if s in stim]
        y = np.array([float(len(stim[sids[i]].get("dispatcher_spans") or [])) for i in keep])
        g = np.array([sids[i] for i in keep])
        X = acts[keep]
        size = np.array([[float(len(stim[sids[i]]["code"])),
                          float(stim[sids[i]]["code"].count("\n") + 1)] for i in keep])

        block = {"n": len(keep), "span_mean": round(float(y.mean()), 2),
                 "span_sd": round(float(y.std()), 2),
                 "span_range": [float(y.min()), float(y.max())]}
        if y.std() == 0:
            block["note"] = "no variation in span count — nothing to decode"
            rep["tiers"][tier] = block
            print(f"[span] {tier:<4} n={len(keep):>3} · no span variation (all {y[0]:.0f})",
                  flush=True)
            continue

        reps = np.array([repetition_features(stim[sids[i]]["code"]) for i in keep])
        comb = np.hstack([size, reps])
        rho_size = spearman(y, oof_ridge(size, y, g))
        rho_rep = spearman(y, oof_ridge(reps, y, g))
        rho_comb = spearman(y, oof_ridge(comb, y, g))
        curve = [spearman(y, oof_ridge(X[:, L], y, g)) for L in range(X.shape[1])]
        mean_rho = st.mean(curve)
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(N_PERM):
            yy = rng.permutation(y)
            null.append(st.mean(spearman(yy, oof_ridge(X[:, L], yy, g))
                                for L in range(X.shape[1])))
        p = (sum(1 for v in null if v >= mean_rho) + 1) / (N_PERM + 1)
        block.update({
            "rho_code_size_baseline": round(rho_size, 4),
            "rho_repetition_only": round(rho_rep, 4),
            "rho_size_plus_repetition": round(rho_comb, 4),
            "beats_size_and_repetition_by": round(mean_rho - rho_comb, 4),
            "rho_residual_mean": round(mean_rho, 4),
            "rho_residual_max": round(max(curve), 4),
            "beats_size_by": round(mean_rho - rho_size, 4),
            "perm_p": round(p, 5), "perm_null_mean": round(st.mean(null), 4),
            "verdict": ("SPANS ENCODED BEYOND SIZE AND REPETITION"
                        if mean_rho - rho_comb >= BAR and p < 0.05
                        else "NOT BEYOND SIZE AND REPETITION"),
        })
        rep["tiers"][tier] = block
        print(f"[span] {tier:<4} n={len(keep):>3} spans {y.mean():.1f}±{y.std():.1f} · "
              f"size {rho_size:+.4f} · rep {rho_rep:+.4f} · size+rep {rho_comb:+.4f} · "
              f"residual {mean_rho:+.4f} · beats {mean_rho - rho_comb:+.4f} "
              f"(p={p:.4f}) -> {block['verdict']}", flush=True)

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["tiers"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
