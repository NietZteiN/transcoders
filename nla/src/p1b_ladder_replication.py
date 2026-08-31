"""Replication of the relational read effect, scored by the frozen rule.

Pre-registration: log/nla-harness/2026-08-31_ladder-replication-prereg.md.

  PRIMARY  Delta = mean(beats-length over L2,L3) - mean(beats-length over L1,L1b),
           computed on the REPLICATION draws only (5-9), selection-free (mean rho over all 28
           layers, no argmax anywhere). REPLICATED iff Delta >= +0.15 AND the relational mean > 0.
           Discovery value was +0.3178; the bar is half of it, set in advance.

The discovery draws (0-4) are deliberately NOT pooled into the primary: a replication that includes
its own discovery data is not a replication. They are used only for the pooled split-half
label-noise check, which is reported as secondary.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402
from p1b_ladder_score import ROUTES, TIERS  # noqa: E402

SEED = 20260724
N_PERM = 200
DELTA_BAR = 0.15
DISCOVERY_DRAWS = set(range(5))


def load(root: Path, tier: str, draws: set[int] | None):
    d = root / tier
    acts = np.load(d / "acts.npy")
    sids = json.loads((d / "items.json").read_text())
    corr: dict[str, list[int]] = {}
    chars: dict[str, list[int]] = {}
    for l in open(d / "draws.jsonl"):
        if not l.strip():
            continue
        r = json.loads(l)
        if draws is not None and r["draw"] not in draws:
            continue
        corr.setdefault(r["snippet_id"], []).append(int(r["correct"]))
        chars.setdefault(r["snippet_id"], []).append(int(r["reply_chars"]))
    keep = [i for i, s in enumerate(sids) if corr.get(s)]
    y = np.array([st.mean(corr[sids[i]]) for i in keep])
    ln = np.array([st.mean(chars[sids[i]]) for i in keep], float)
    g = np.array([sids[i] for i in keep])
    return acts[keep], y, ln, g, {sids[i]: corr[sids[i]] for i in keep}


def beats(X, y, ln, g):
    base = spearman(y, oof_ridge(ln.reshape(-1, 1), y, g))
    c = [spearman(y, oof_ridge(X[:, L], y, g)) for L in range(X.shape[1])]
    return st.mean(c) - base, st.mean(c), base, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p1b/ladder"))
    ap.add_argument("--rep-draws", default="5,6,7,8,9")
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/ladder_replication.json"))
    args = ap.parse_args()
    root = Path(args.root)
    rep_draws = {int(x) for x in args.rep_draws.split(",")}

    rep = {"experiment": "p1b_ladder_replication",
           "prereg": "log/nla-harness/2026-08-31_ladder-replication-prereg.md",
           "replication_draws": sorted(rep_draws), "delta_bar": DELTA_BAR,
           "discovery_delta": 0.3178, "tiers": {}}

    store = {}
    for tier in TIERS:
        if not (root / tier / "acts.npy").exists():
            continue
        X, y, ln, g, corr = load(root, tier, rep_draws)
        n_draws = st.mean(len(v) for v in corr.values()) if corr else 0
        if not len(y) or n_draws < len(rep_draws):
            rep["tiers"][tier] = {"note": f"incomplete: mean {n_draws} draws per item"}
            continue
        b, mrho, base, curve = beats(X, y, ln, g)
        rep["tiers"][tier] = {"n": int(len(y)), "draws_per_item": n_draws,
                              "mean_correct": round(float(y.mean()), 4),
                              "length_baseline_rho": round(base, 4),
                              "mean_rho": round(mrho, 4),
                              "beats_length_mean": round(b, 4),
                              "max_rho": round(max(curve), 4)}
        store[tier] = (X, y, ln, g)
        print(f"[rep] {tier:<4} n={len(y):>3} len-base {base:+.4f} · mean {mrho:+.4f} · "
              f"beats {b:+.4f}", flush=True)

    ok = [t for t in TIERS if "beats_length_mean" in rep["tiers"].get(t, {})]
    rel = [t for t in ROUTES["relational"] if t in ok]
    atom = [t for t in ROUTES["atom"] if t in ok]
    if rel and atom:
        rel_m = st.mean(rep["tiers"][t]["beats_length_mean"] for t in rel)
        atom_m = st.mean(rep["tiers"][t]["beats_length_mean"] for t in atom)
        delta = rel_m - atom_m
        rep["primary"] = {
            "relational_mean": round(rel_m, 4), "atom_mean": round(atom_m, 4),
            "delta": round(delta, 4), "bar": DELTA_BAR,
            "verdict": ("REPLICATED" if delta >= DELTA_BAR and rel_m > 0 else "NOT REPLICATED"),
        }
    else:
        rep["primary"] = {"verdict": "NEEDS_DATA"}

    # secondary: per-tier permutation nulls on the mean statistic
    rep["permutation"] = {}
    for tier in ("L1b", "L2", "L3"):
        if tier not in store:
            continue
        X, y, ln, g = store[tier]
        _, mrho, _, _ = beats(X, y, ln, g)
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(N_PERM):
            yy = rng.permutation(y)
            null.append(st.mean(spearman(yy, oof_ridge(X[:, L], yy, g))
                                for L in range(X.shape[1])))
        rep["permutation"][tier] = {
            "observed_mean_rho": round(mrho, 4), "null_mean": round(st.mean(null), 4),
            "null_sd": round(st.pstdev(null), 4), "null_max": round(max(null), 4),
            "p": round((sum(1 for v in null if v >= mrho) + 1) / (N_PERM + 1), 5)}
        print(f"[rep:null] {tier:<4} {mrho:+.4f} vs null "
              f"{st.mean(null):+.4f}±{st.pstdev(null):.4f} "
              f"p={rep['permutation'][tier]['p']}", flush=True)

    # secondary: pooled split-half label-noise check on L2/L3
    rep["split_half_pooled"] = {}
    for tier in ("L2", "L3"):
        if not (root / tier / "acts.npy").exists():
            continue
        X, y_all, ln, g, corr = load(root, tier, None)
        ks = sorted({d for s in corr for d in range(len(corr[s]))})
        rng = random.Random(SEED)
        vals = []
        V = np.array([corr[s] for s in g])
        for _ in range(10):
            idx = list(range(V.shape[1]))
            rng.shuffle(idx)
            h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
            b1, _, _, _ = beats(X, V[:, h1].mean(1), ln, g)
            b2, _, _, _ = beats(X, V[:, h2].mean(1), ln, g)
            vals.append((b1, b2))
        rep["split_half_pooled"][tier] = {
            "n_splits": 10,
            "mean_half_beats": [round(st.mean(v[0] for v in vals), 4),
                                round(st.mean(v[1] for v in vals), 4)],
            "min_half_beats": round(min(min(v) for v in vals), 4),
            "both_halves_positive_frac": round(
                sum(1 for v in vals if v[0] > 0 and v[1] > 0) / len(vals), 4)}
        print(f"[rep:split] {tier} halves {rep['split_half_pooled'][tier]['mean_half_beats']} "
              f"· both positive on "
              f"{rep['split_half_pooled'][tier]['both_halves_positive_frac']:.0%} of splits",
              flush=True)

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["primary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
