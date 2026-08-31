"""Graded correctness labels: is the depth curve identifiable once the label noise is reduced?

THE MOVE. A single binary grade is a Bernoulli draw whose noise flipped a depth curve's argmax
from L27 to L13 (2026-08-30). Ten independent unsteered generations of these same 60 items exist
on disk, so the fraction correct `k/10` is available at zero GPU cost. It is continuous, it is
about sqrt(10) less noisy than one grade, and on this corpus it is also a natural *difficulty*
measure: 19 items are never right, 26 always, and 15 sit in between.

WHAT IS PRE-DECLARED HERE. This is exploratory analysis of data already inspected, so it is not a
hypothesis test and is not presented as one. But one rule is frozen before running, because it is
the question that matters and it is answerable with the draws on hand:

    SPLIT-HALF IDENTIFIABILITY. Partition the 10 draws into two disjoint sets of 5, build a
    per-layer curve from each, and compare. Repeated over N_SPLITS random partitions.
      IDENTIFIABLE      mean Spearman between the two curves >= 0.70 AND the argmax layer
                        agrees on >= 50% of splits.
      NOT IDENTIFIABLE  otherwise — report the level and refuse the shape.
    0.70 is the conventional reliability line, the same one used for N10b on 2026-08-29.

Ridge rather than logistic because the target is continuous; alpha frozen at 1.0, not tuned, for
the reason C was frozen in the binary probe (p >> n at d = 3584, n = 60).

Controls carried over: GroupKFold on snippet_id, a length-only baseline, and a permutation
DISTRIBUTION rather than a single draw.

CPU only, minutes.
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
from p1b_consensus_labels import SOURCES  # noqa: E402

SEED = 20260724
ALPHA_FROZEN = 1.0
N_FOLDS, N_SPLITS, N_PERM = 5, 20, 200
RHO_STABLE, ARGMAX_STABLE = 0.70, 0.50


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j + 2) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    sx, sy = rx.std(), ry.std()
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy)) if sx and sy else 0.0


def oof_ridge(X, y, groups):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=ALPHA_FROZEN).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return oof


def curve(acts, pi, y, groups, n_layers):
    return [round(spearman(y, oof_ridge(acts[:, pi, L], y, groups)), 4) for L in range(n_layers)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=str(PROJ / "data/nla/p0/p1b/pos_acts.npz"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/graded_labels.json"))
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
    acts, valid, groups = z["acts"], z["valid"], z["groups"]
    lengths = z["lengths"].astype(float)
    positions = [str(x) for x in z["positions"]]
    n_layers = acts.shape[2]
    n_draws = len(used)

    keep = [i for i in range(len(groups)) if len(votes.get(str(groups[i]), [])) == n_draws]
    g = np.array([str(groups[i]) for i in keep])
    V = np.array([votes[str(groups[i])] for i in keep])          # [n_items, n_draws]
    y = V.mean(1)                                                 # k/10

    rep = {"experiment": "p1b_graded_labels", "sources": used, "n_draws": n_draws,
           "n_items": len(keep), "alpha": ALPHA_FROZEN, "target": "fraction correct over draws",
           "note": "exploratory analysis of already-inspected data; only the split-half "
                   "identifiability rule was frozen before running",
           "target_mean": round(float(y.mean()), 4),
           "target_sd": round(float(y.std()), 4),
           "positions": {}}

    for pi, tag in enumerate(positions):
        v = np.array([valid[i, pi] for i in keep])
        if v.sum() < 30:
            rep["positions"][tag] = {"n": int(v.sum()), "note": "too few valid items"}
            continue
        A = acts[np.array(keep)[v]]
        yy, gg = y[v], g[v]
        c = curve(A, pi, yy, gg, n_layers)
        len_rho = round(spearman(yy, oof_ridge(lengths[np.array(keep)[v]].reshape(-1, 1), yy, gg)), 4)
        best = int(np.argmax(c))
        rep["positions"][tag] = {
            "n": int(v.sum()), "length_only_rho": len_rho,
            "rho_by_layer": c, "argmax_layer": best, "argmax_rho": c[best],
            "rho_L13": c[13], "rho_L20": c[20], "rho_L27": c[27],
            "mean_rho": round(st.mean(c), 4),
        }
        print(f"[grad] {tag:<12} n={int(v.sum()):>3} len {len_rho:+.4f} · "
              f"L13 {c[13]:+.4f} L20 {c[20]:+.4f} L27 {c[27]:+.4f} · "
              f"argmax L{best} {c[best]:+.4f}", flush=True)

    # ── the frozen rule: split-half identifiability of the CURVE ──────────────
    rng = random.Random(SEED)
    stab = {}
    for tag in ("last_prompt", "answer_line"):
        pi = positions.index(tag)
        v = np.array([valid[i, pi] for i in keep])
        A = acts[np.array(keep)[v]]
        gg, VV = g[v], V[v]
        rhos, argmax_hit, full_best = [], 0, rep["positions"][tag]["argmax_layer"]
        for _ in range(N_SPLITS):
            idx = list(range(n_draws))
            rng.shuffle(idx)
            h1, h2 = idx[:n_draws // 2], idx[n_draws // 2:]
            c1 = curve(A, pi, VV[:, h1].mean(1), gg, n_layers)
            c2 = curve(A, pi, VV[:, h2].mean(1), gg, n_layers)
            rhos.append(spearman(c1, c2))
            argmax_hit += int(int(np.argmax(c1)) == int(np.argmax(c2)))
        mean_rho = st.mean(rhos)
        frac = argmax_hit / N_SPLITS
        stab[tag] = {
            "n_splits": N_SPLITS, "mean_curve_spearman": round(mean_rho, 4),
            "sd_curve_spearman": round(st.pstdev(rhos), 4),
            "argmax_agreement_frac": round(frac, 4),
            "full_sample_argmax": full_best,
            "verdict": ("IDENTIFIABLE" if mean_rho >= RHO_STABLE and frac >= ARGMAX_STABLE
                        else "NOT IDENTIFIABLE"),
        }
        print(f"[grad] split-half {tag:<12} curve rho {mean_rho:+.4f} · "
              f"argmax agree {frac:.2f} · {stab[tag]['verdict']}", flush=True)
    rep["split_half_identifiability"] = stab
    rep["thresholds"] = {"curve_spearman": RHO_STABLE, "argmax_agreement": ARGMAX_STABLE}

    # permutation distribution at last_prompt / L20
    pi = positions.index("last_prompt")
    v = np.array([valid[i, pi] for i in keep])
    A, yy, gg = acts[np.array(keep)[v]], y[v], g[v]
    obs = spearman(yy, oof_ridge(A[:, pi, 20], yy, gg))
    nprng = np.random.default_rng(SEED)
    null = [spearman(nprng.permutation(yy), oof_ridge(A[:, pi, 20], yy, gg)) for _ in range(N_PERM)]
    rep["permutation_control"] = {
        "cell": "last_prompt/L20", "observed": round(obs, 4),
        "null_mean": round(st.mean(null), 4), "null_sd": round(st.pstdev(null), 4),
        "null_max": round(max(null), 4),
        "p_empirical": round((sum(1 for x in null if x >= obs) + 1) / (N_PERM + 1), 5)}

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps({k: v for k, v in rep.items() if k != "positions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
