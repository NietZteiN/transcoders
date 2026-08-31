"""Two closing controls on the dispatcher-span result.

(1) MID-REASONING, PROBED FOR SOMETHING ELSE. Correctness is undecodable in the middle of the
    reply — five nulls across two tiers and three instruments. That says the region does not encode
    *correctness*; it does not say the region is empty. `n_dispatcher_spans` is a target we know is
    encoded at `last_prompt` (rho +0.8842), so asking whether it is still encoded at reply_q50
    distinguishes "the residual stream goes quiet mid-reasoning" from "it stays informative about
    the code but not about the outcome".

(2) REPETITION CONTROL. A dispatcher object repeats a lexical pattern N times, so a probe that
    predicts span count may be counting repeats rather than representing structure. The baseline is
    therefore extended with repetition features and the span probe must still clear it.

CPU only.
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
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402
from p1b_l2_mechanism import stimuli  # noqa: E402

SEED, N_PERM, BAR = 20260724, 200, 0.10


def repetition_features(code: str) -> list[float]:
    """How repetitive is this text, independent of how many dispatcher sites it has?"""
    toks = code.split()
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    tc, lc = Counter(toks), Counter(lines)
    return [
        float(tc.most_common(1)[0][1]) if tc else 0.0,      # max token frequency
        float(len(tc)),                                      # distinct tokens
        float(len(toks)),                                    # total tokens
        float(sum(v for v in lc.values() if v > 1)),         # duplicated lines
        float(lc.most_common(1)[0][1]) if lc else 0.0,       # max line frequency
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-acts", default=str(PROJ / "data/nla/p0/p1b/pos_acts_L2.npz"))
    ap.add_argument("--ladder", default=str(PROJ / "data/nla/p0/p1b/ladder/L2"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/span_positions.json"))
    args = ap.parse_args()

    stim = stimuli("L2")
    rep = {"experiment": "p1b_span_positions_and_repetition", "seed": SEED, "bar": BAR,
           "positions": {}, "repetition_control": {}}

    # ── (1) spans across reply positions ──────────────────────────────────────
    z = np.load(args.pos_acts, allow_pickle=True)
    acts, valid, groups = z["acts"], z["valid"], z["groups"]
    positions = [str(x) for x in z["positions"]]
    keep0 = [i for i in range(len(groups)) if str(groups[i]) in stim]
    y_all = np.array([float(len(stim[str(groups[i])].get("dispatcher_spans") or []))
                      for i in keep0])
    size_all = np.array([[float(len(stim[str(groups[i])]["code"])),
                          float(stim[str(groups[i])]["code"].count("\n") + 1)] for i in keep0])
    g_all = np.array([str(groups[i]) for i in keep0])

    for pi, tag in enumerate(positions):
        m = np.array([valid[i, pi] for i in keep0])
        if m.sum() < 30:
            rep["positions"][tag] = {"n": int(m.sum()), "note": "too few valid"}
            continue
        X = acts[np.array(keep0)[m]][:, pi]
        y, g, size = y_all[m], g_all[m], size_all[m]
        base = spearman(y, oof_ridge(size, y, g))
        curve = [spearman(y, oof_ridge(X[:, L], y, g)) for L in range(X.shape[1])]
        mean_rho = st.mean(curve)
        rep["positions"][tag] = {
            "n": int(m.sum()), "size_baseline_rho": round(base, 4),
            "span_rho_mean": round(mean_rho, 4), "span_rho_max": round(max(curve), 4),
            "beats_size_by": round(mean_rho - base, 4)}
        print(f"[pos] {tag:<12} n={int(m.sum()):>3} · size-base {base:+.4f} · "
              f"spans {mean_rho:+.4f} · beats {mean_rho - base:+.4f}", flush=True)

    # ── (2) repetition control at last_prompt, on the ladder acts ─────────────
    acts2 = np.load(Path(args.ladder) / "acts.npy")
    sids = json.loads((Path(args.ladder) / "items.json").read_text())
    keep = [i for i, s in enumerate(sids) if s in stim]
    y = np.array([float(len(stim[sids[i]].get("dispatcher_spans") or [])) for i in keep])
    g = np.array([sids[i] for i in keep])
    X = acts2[keep]
    size = np.array([[float(len(stim[sids[i]]["code"])),
                      float(stim[sids[i]]["code"].count("\n") + 1)] for i in keep])
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
        null.append(st.mean(spearman(yy, oof_ridge(X[:, L], yy, g)) for L in range(X.shape[1])))
    p = (sum(1 for v in null if v >= mean_rho) + 1) / (N_PERM + 1)
    beats = mean_rho - rho_comb
    rep["repetition_control"] = {
        "n": len(keep), "rho_size_only": round(rho_size, 4),
        "rho_repetition_only": round(rho_rep, 4),
        "rho_size_plus_repetition": round(rho_comb, 4),
        "rho_residual_mean": round(mean_rho, 4),
        "beats_combined_by": round(beats, 4), "perm_p": round(p, 5),
        "verdict": ("SPANS ENCODED BEYOND SIZE AND REPETITION" if beats >= BAR and p < 0.05
                    else "NOT BEYOND SIZE AND REPETITION")}
    r = rep["repetition_control"]
    print(f"\n[rep-ctl] size {rho_size:+.4f} · repetition {rho_rep:+.4f} · combined "
          f"{rho_comb:+.4f} · residual {mean_rho:+.4f} · beats {beats:+.4f} (p={p:.4f})")
    print(f"[rep-ctl] {r['verdict']}")

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
