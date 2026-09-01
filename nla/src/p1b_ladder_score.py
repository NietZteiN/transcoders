"""The ladder, scored: does the residual stream beat a token count at ANY tier?

Each tier is measured against ITS OWN length baseline — the lesson of the 2026-08-30 tier
retraction, where a +0.1383 raw-rho gap turned out to be a +0.1567 baseline gap. Raw rho is not
comparable across tiers that differ in accuracy and in reply-length variance; the incremental value
over a token count is.

Route grouping follows the study's frame: ATOM-level interference (L1 nonsense renaming, L1b
adversarial renaming) versus RELATIONAL overload (L2 control-flow flattening, L3 stacked), with L0
as the clean anchor.

A permutation DISTRIBUTION is run at the tier with the largest increment — single draws have
misled three times this week.

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
from p1b_ladder import read_draws  # noqa: E402
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402

SEED = 20260724
N_PERM = 200
TIERS = ["L0", "L1", "L1b", "L2", "L3"]
ROUTES = {"clean": ["L0"], "atom": ["L1", "L1b"], "relational": ["L2", "L3"]}
BEATS = 0.10          # what would count as the residual stream carrying something length does not


def load_tier(root: Path, tier: str, draws: set[int] | None = None):
    d = root / tier
    acts = np.load(d / "acts.npy")
    sids = json.loads((d / "items.json").read_text())
    rows = read_draws(d)          # every shard, not just draws.jsonl
    if draws is not None:
        rows = [r for r in rows if r["draw"] in draws]
    corr: dict[str, list[int]] = {}
    chars: dict[str, list[int]] = {}
    for r in rows:
        corr.setdefault(r["snippet_id"], []).append(int(r["correct"]))
        chars.setdefault(r["snippet_id"], []).append(int(r["reply_chars"]))
    keep = [i for i, s in enumerate(sids) if s in corr]
    y = np.array([st.mean(corr[sids[i]]) for i in keep])
    ln = np.array([st.mean(chars[sids[i]]) for i in keep], float)
    g = np.array([sids[i] for i in keep])
    return acts[keep], y, ln, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p1b/ladder"))
    # WITHOUT this the script silently changes meaning when draws are added later. The Qwen
    # ladder was scored as a 5-draw discovery run, then the replication appended draws 5-9 to the
    # same files — so a re-run now pools ten draws and produces different numbers under the same
    # filename, with no log entry describing them. The discovery artifact is `--draws 0-4`.
    ap.add_argument("--draws", default=None,
                    help="draw indices to score, e.g. '0-4' or '0,1,2'. Default: all present.")
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/ladder_score.json"))
    args = ap.parse_args()
    root = Path(args.root)
    sel = None
    if args.draws:
        sel = set()
        for part in args.draws.split(","):
            if "-" in part:
                a, b = part.split("-"); sel.update(range(int(a), int(b) + 1))
            else:
                sel.add(int(part))

    rep = {"experiment": "p1b_ladder_read_probe", "seed": SEED, "beats_threshold": BEATS,
           "draws_scored": sorted(sel) if sel else "all present",
           "note": "each tier against its own length baseline; see the 2026-08-30 tier retraction",
           "tiers": {}}
    store = {}

    for tier in TIERS:
        if not (root / tier / "acts.npy").exists():
            rep["tiers"][tier] = {"note": "missing"}
            continue
        X, y, ln, g = load_tier(root, tier, sel)
        base = round(spearman(y, oof_ridge(ln.reshape(-1, 1), y, g)), 4)
        curve = [round(spearman(y, oof_ridge(X[:, L], y, g)), 4) for L in range(X.shape[1])]
        best = int(np.argmax(curve))
        rep["tiers"][tier] = {
            "n": int(len(y)), "mean_correct": round(float(y.mean()), 4),
            "mean_reply_chars": round(float(ln.mean()), 1),
            "length_baseline_rho": base,
            "argmax_layer": best, "argmax_rho": curve[best],
            "mean_rho": round(st.mean(curve), 4),
            "beats_length_by": round(curve[best] - base, 4),
            "rho_by_layer": curve,
        }
        store[tier] = (X, y, ln, g)
        b = rep["tiers"][tier]
        print(f"[ladder] {tier:<4} n={b['n']:>3} acc {b['mean_correct']:.3f} · "
              f"len-base {base:+.4f} · argmax L{best} {curve[best]:+.4f} · "
              f"beats length {b['beats_length_by']:+.4f}", flush=True)

    ok = [t for t in TIERS if "beats_length_by" in rep["tiers"].get(t, {})]
    rep["routes"] = {
        r: {"tiers": ts,
            "mean_beats_length": round(st.mean(rep["tiers"][t]["beats_length_by"]
                                               for t in ts if t in ok), 4),
            "mean_length_baseline": round(st.mean(rep["tiers"][t]["length_baseline_rho"]
                                                  for t in ts if t in ok), 4)}
        for r, ts in ROUTES.items() if any(t in ok for t in ts)}

    best_tier = max(ok, key=lambda t: rep["tiers"][t]["beats_length_by"]) if ok else None
    if best_tier:
        X, y, ln, g = store[best_tier]
        L = rep["tiers"][best_tier]["argmax_layer"]
        obs = rep["tiers"][best_tier]["argmax_rho"]
        rng = np.random.default_rng(SEED)
        null = [spearman(rng.permutation(y), oof_ridge(X[:, L], y, g)) for _ in range(N_PERM)]
        rep["permutation_control"] = {
            "tier": best_tier, "layer": L, "n_perm": N_PERM, "observed": obs,
            "null_mean": round(st.mean(null), 4), "null_sd": round(st.pstdev(null), 4),
            "null_max": round(max(null), 4),
            "p_empirical": round((sum(1 for x in null if x >= obs) + 1) / (N_PERM + 1), 5)}

    top = max((rep["tiers"][t]["beats_length_by"] for t in ok), default=0.0)
    rep["verdict"] = (
        "RESIDUAL STREAM CARRIES SOMETHING LENGTH DOES NOT — at least one tier clears the bar"
        if top >= BEATS else
        "`last_prompt` ~ REPLY LENGTH AT EVERY TIER — no tier's residual stream adds "
        f"{BEATS:+.2f} over a token count (max {top:+.4f})")
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps({k: v for k, v in rep.items() if k != "tiers"}, indent=2))
    print(f"\n[ladder] {rep['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
