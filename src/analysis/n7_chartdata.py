"""Precompute the chart series for the results browser.

Kept OUT of the page builder deliberately: these are the same numbers the N6/N7/N8 verdicts rest on,
so they belong in version control next to the analyses that produced them, not inside a rendering
script. The builder reads this file and draws; it never derives a statistic itself.

Emits data/nla/n7/2026-08-07/charts.json with six series:

  judge_dist      score histogram (0-3) per condition -- what "the judge separates them" looks like
  roc             ROC for judge score AND the AR-space baseline, matched vs shuffled
  confusion       Llama x Phi 4x4 agreement counts -- why the kappa gate failed
  align_by_pos    mean judge alignment across relative trace position, correct vs wrong (HT13, exploratory)
  forest          HT12's decomposition arms with bootstrap CIs -- where the effect went
  onset_excess    HT14's own-minus-foreign answer rate per trace decile, correct vs wrong

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
N7 = PROJ / "data/nla/n7/2026-08-07"
N_BINS = 10
SEED = 20260724

sys.path.insert(0, str(PROJ))


def roc_points(pos: np.ndarray, neg: np.ndarray, max_pts: int = 60) -> list[dict]:
    """ROC by sweeping every distinct threshold. Ties are handled by scoring >= t, which is what
    makes the integer 0-3 judge scale produce a 5-point curve rather than a smooth one -- that
    coarseness is a real property of the instrument and is left visible."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    ts = np.unique(np.concatenate([pos, neg]))
    ts = np.concatenate([[ts.max() + 1], ts[::-1]])
    if len(ts) > max_pts:
        idx = np.unique(np.linspace(0, len(ts) - 1, max_pts).astype(int))
        ts = ts[idx]
    out = []
    for t in ts:
        out.append({"fpr": round(float((neg >= t).mean()), 4),
                    "tpr": round(float((pos >= t).mean()), 4)})
    out.append({"fpr": 1.0, "tpr": 1.0})
    return out


def main() -> int:
    from src.analysis.n7_alignment import auc, load

    d = load("llama")
    real = d[d.condition == "real"]
    out: dict = {}

    # ---- 1. score distribution per condition ----
    dist = {}
    for cond in ("real", "distant", "shuffled"):
        c = Counter(d[d.condition == cond]["score"].astype(int))
        n = max(sum(c.values()), 1)
        dist[cond] = {"n": n, "pct": [round(100 * c.get(s, 0) / n, 2) for s in range(4)],
                      "counts": [int(c.get(s, 0)) for s in range(4)],
                      "mean": round(float(d[d.condition == cond]["score"].mean()), 3)}
    out["judge_dist"] = dist

    # ---- 2. ROC: judge vs the free AR-space baseline, on the same comparison ----
    shuf = d[d.condition == "shuffled"]
    a_j, _ = auc(real["score"], shuf["score"])
    a_ar, _ = auc(real["ar_cos"], shuf["ar_cos"])
    out["roc"] = {
        "judge": {"points": roc_points(real["score"], shuf["score"]), "auc": round(a_j, 4)},
        "ar": {"points": roc_points(real["ar_cos"], shuf["ar_cos"]), "auc": round(a_ar, 4)},
        "spearman": float(real[["score", "ar_cos"]].dropna().corr(method="spearman").iloc[0, 1]),
    }

    # ---- 3. Llama x Phi agreement ----
    p = load("phi").set_index("item_id")["score"]
    j = d.set_index("item_id")["score"]
    common = j.index.intersection(p.index)
    m = np.zeros((4, 4), int)
    for a, b in zip(j.loc[common].astype(int), p.loc[common].astype(int)):
        m[a, b] += 1
    out["confusion"] = {"n": int(len(common)), "matrix": m.tolist(),
                        "llama_mean": round(float(j.loc[common].mean()), 2),
                        "phi_mean": round(float(p.loc[common].mean()), 2),
                        "agree_pct": round(100 * float(np.trace(m)) / max(len(common), 1), 1)}

    # ---- 4. alignment across the trace, correct vs wrong (HT13, exploratory) ----
    r = real[real.u_rel.notna()].copy()
    r["bin"] = np.minimum((r["u_rel"] * N_BINS).astype(int), N_BINS - 1)
    series = {}
    rng = np.random.default_rng(SEED)
    for corr in (1, 0):
        g = r[r.correct == corr]
        pts = []
        for b in range(N_BINS):
            gb = g[g.bin == b]
            if not len(gb):
                pts.append(None)
                continue
            # cluster bootstrap over cases: reads within a case are not independent
            cases = gb["case"].unique()
            per = gb.groupby("case")["score"].mean()
            bs = np.array([per.loc[rng.choice(cases, len(cases), True)].mean() for _ in range(400)])
            pts.append({"bin": b, "mean": round(float(per.mean()) / 3, 4),
                        "lo": round(float(np.percentile(bs, 2.5)) / 3, 4),
                        "hi": round(float(np.percentile(bs, 97.5)) / 3, 4),
                        "n": int(len(gb)), "cases": int(len(cases))})
        series["correct" if corr else "wrong"] = pts
    out["align_by_pos"] = series

    # ---- 5. HT12 decomposition as a forest ----
    dec = json.load(open(PROJ / "data/nla/n6/2026-08-07/ht12_decomposition.json"))
    label = {"banked/all-cases": "First pass, every case",
             "banked/matched-cases-only": "First pass, length-matched cases only",
             "dense/all": "Dense pass (length-matched)",
             "dense/sweep-only": "Dense pass, uniform sweep only",
             "dense/strip-only": "Dense pass, late strip only"}
    out["forest"] = [{"key": k, "label": label.get(k, k),
                      "diff": v["case_level"]["raw_diff"],
                      "lo": v["case_level"]["boot_ci95"][0], "hi": v["case_level"]["boot_ci95"][1],
                      "p": v["case_level"]["p"], "n_cases": v["n_cases"]}
                     for k, v in dec.items() if "case_level" in v]

    # ---- 6. HT14 excess over the foreign-answer null ----
    h = json.load(open(PROJ / "data/nla/n8/2026-08-07/ht14.json"))["primary"]
    out["onset_excess"] = {
        "correct": [h["excess_by_bin"]["1"].get(str(b), h["excess_by_bin"]["1"].get(b)) for b in range(N_BINS)],
        "wrong": [h["excess_by_bin"]["0"].get(str(b), h["excess_by_bin"]["0"].get(b)) for b in range(N_BINS)],
        "own_rate": h["own_rate_overall"], "foreign_rate": h["foreign_rate_overall"],
        "mean_excess": h["mean_excess_over_null"],
    }

    p_out = N7 / "charts.json"
    p_out.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {p_out}")
    for k in out:
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
