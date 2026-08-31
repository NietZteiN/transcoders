"""Control B, done properly: does the residual stream beat reply LENGTH on clean code but only
tie it under adversarial renaming?

The 2026-08-30 version could not answer this. `baseline.jsonl` recorded `reply_chars` for the L1b
generation only, so L0 had no length baseline and its rho was compared against L1b's number. That
limitation is now closed: `steer_run.py --baseline-only` records `l0_reply_chars` and
`l1b_reply_chars`, and one pass at the same 2048 budget gives both tiers a baseline computed under
the same generation regime.

WHY THE COMPARISON IS "BEATS ITS OWN BASELINE" AND NOT "HIGHER RHO". The two tiers differ in
accuracy (0.633 vs 0.550) and in how much their replies vary in length, so raw rho is not
comparable across them. The quantity that is comparable is how much the residual stream adds OVER
a token count within each tier.

Labels are the established 10-draw consensus so the numbers stay comparable with the entry that
raised the limitation; the new pass is reported as an 11th draw in a robustness line rather than
folded into the primary.

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
from p1b_consensus_labels import SOURCES  # noqa: E402
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402

SEED = 20260724


def votes_for(field: str, sources: list[str]) -> tuple[dict[str, list[int]], int]:
    v: dict[str, list[int]] = {}
    n = 0
    for src in sources:
        p = PROJ / "data/nla/p0" / src / "baseline.jsonl"
        if not p.exists():
            continue
        n += 1
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                v.setdefault(r["snippet_id"], []).append(int(bool(r[field])))
    return v, n


def analyse(tacts, y_tier, tgroups, flag, votes, n_draws, lens, label):
    m = np.array([y_tier[i] == flag and str(tgroups[i]) in votes
                  and len(votes[str(tgroups[i])]) == n_draws
                  and str(tgroups[i]) in lens for i in range(len(y_tier))])
    y = np.array([st.mean(votes[str(tgroups[i])]) for i in range(len(y_tier)) if m[i]])
    g = np.array([str(tgroups[i]) for i in range(len(y_tier)) if m[i]])
    Ln = np.array([lens[str(tgroups[i])] for i in range(len(y_tier)) if m[i]], float)
    X = tacts[m]
    base = round(spearman(y, oof_ridge(Ln.reshape(-1, 1), y, g)), 4)
    curve = [round(spearman(y, oof_ridge(X[:, L], y, g)), 4) for L in range(tacts.shape[1])]
    best = int(np.argmax(curve))
    out = {"tier": label, "n": int(m.sum()), "n_draws": n_draws,
           "target_mean": round(float(y.mean()), 4),
           "length_baseline_rho": base,
           "argmax_layer": best, "argmax_rho": curve[best],
           "mean_rho": round(st.mean(curve), 4),
           "beats_length_by": round(curve[best] - base, 4),
           "mean_reply_chars": round(float(Ln.mean()), 1),
           "rho_by_layer": curve}
    print(f"[tier] {label:<4} n={out['n']:>3} len-baseline {base:+.4f} · "
          f"argmax L{best} {curve[best]:+.4f} · beats length by {out['beats_length_by']:+.4f} "
          f"· mean reply {out['mean_reply_chars']:.0f} chars", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-acts", default=str(PROJ / "data/nla/p0/p1b/acts.npz"))
    ap.add_argument("--baselen", default=str(PROJ / "data/nla/p0/p1b/baselen/baseline.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/tier_control.json"))
    args = ap.parse_args()

    lens0, lens1 = {}, {}
    for l in open(args.baselen):
        if l.strip():
            r = json.loads(l)
            if "l0_reply_chars" in r:
                lens0[r["snippet_id"]] = r["l0_reply_chars"]
                lens1[r["snippet_id"]] = r["l1b_reply_chars"]
    if not lens0:
        raise SystemExit("baseline file has no l0_reply_chars — rerun with the patched steer_run")

    z = np.load(args.tier_acts, allow_pickle=True)
    tacts, y_tier, tgroups = z["acts"], z["y_tier"], z["groups"]

    rep = {"experiment": "p1b_tier_control_with_length_baseline", "seed": SEED,
           "closes": "the missing-L0-length-baseline limitation in 2026-08-30_p1b-readout-controls.md",
           "baseline_source": args.baselen, "tiers": {}}

    for tier, field, flag, lens in (("L0", "l0_correct", 0, lens0),
                                    ("L1b", "l1b_correct", 1, lens1)):
        v, n = votes_for(field, SOURCES)
        rep["tiers"][tier] = analyse(tacts, y_tier, tgroups, flag, v, n, lens, tier)

    a, b = rep["tiers"]["L0"], rep["tiers"]["L1b"]
    rep["contrast"] = {
        "delta_argmax_rho": round(a["argmax_rho"] - b["argmax_rho"], 4),
        "delta_beats_length": round(a["beats_length_by"] - b["beats_length_by"], 4),
        "verdict": ("OBFUSCATION DEGRADES THE READ — the residual stream adds more over a token "
                    "count on clean code than on renamed code"
                    if a["beats_length_by"] - b["beats_length_by"] >= 0.10 else
                    "NO TIER DIFFERENCE IN INCREMENTAL VALUE — the raw rho gap is not an "
                    "incremental-information gap"),
    }
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n[tier] L0 beats length by {a['beats_length_by']:+.4f} · "
          f"L1b by {b['beats_length_by']:+.4f} · delta {rep['contrast']['delta_beats_length']:+.4f}")
    print(f"[tier] {rep['contrast']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
