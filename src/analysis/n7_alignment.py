"""N7 / HT13 — judge-scored CoT<->NLA alignment: instrument gate first, hypothesis second.

Confirmatory. Rules frozen in log/nla-harness/2026-08-06_n6-n8-preregistration.md, with the
analyzable-n and power declared in 2026-08-07_n7-prerun-power-addendum.md BEFORE the judge ran.

THE GATE RUNS FIRST AND CAN VETO EVERYTHING. A judge that cannot tell a real (read, window) pair
from a shuffled one is not measuring alignment, and its HT13 verdict would be noise dressed as a
result. So, in order:

  G1  shuffled-null AUC >= 0.65        judge separates real pairs from different-case pairs
  G2  judge AUC > AR-baseline AUC      the 8B judge buys something over cos(AR(read), AR(window)),
                                       which is free -- pre-registered so "beats chance" is not
                                       allowed to masquerade as "works"
  G3  kappa(Llama, Phi-3.5) >= 0.40    the score is a property of the pair, not of one model

  (reported, not gating) distant-null AUC -- same case, |du| > 0.4. A judge scoring mere topicality
  scores these as high as real pairs; separation here is what makes the score *local*.

ANY GATE FAILS -> N7 is reported as an INSTRUMENT NULL, HT13 is NOT adjudicated, and the artifact's
alignment sort stays disabled. That is a pre-registered outcome, not a fallback.

HT13 (only if the gate passes), EXACTLY as frozen:
    align = score / 3
    after_error = 1[u_rel > 0.70]          <- the FIXED STRIP, on every case (pre-reg clause 3)
    align ~ correct * after_error + C(tier) + log(read_len) + u_rel + u_rel^2   [cluster = case]
  confirm = negative `correct:after_error` interaction, p < .05, AND the before-error simple effect's
  CI includes 0 (the drop must LOCALIZE rather than track correctness globally).
  refute  = the interaction CI includes 0, OR the before-error effect matches the after-error effect
            in magnitude (a different, weaker claim -- reported as such, never as support).

  Sensitivity arm = the per-case *localized* first-error boundary, which clause 3 restricts to the
  trusted detectors because localization failed validation on 2026-08-06. This is the arm the
  2026-08-07 power addendum sized (n=23 / 8 trusted, 0.90 / 0.54 power at 0.4 SD). The addendum
  labelled it "primary" -- that was an error against the frozen text and is corrected here: the
  fixed strip is the primary, and it runs on all 182 cases, so the primary is far better powered
  than the addendum implied.

DEVIATION, same one declared for HT12: the frozen formula carries `(1|case)`. `correct` is constant
within case, so a per-case random intercept annihilates it (HT12: singular fit / beta=0 with NaN se).
GEE with an exchangeable working correlation clustered on case is used instead. The INTERACTION does
vary within case and would survive a random intercept, but the model must estimate both terms, and
GEE estimates both. Nothing else about the rule changes.

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
N7 = PROJ / "data/nla/n7/2026-08-07"
SEED = 20260724
GATE_AUC = 0.65
GATE_KAPPA = 0.40
TRUSTED = {"D0", "D2-call"}


def load(tag: str = "llama") -> pd.DataFrame:
    keys = {json.loads(l)["item_id"]: json.loads(l) for l in open(N7 / "keys.jsonl")}
    # read/window lengths come from the pack; keys.jsonl deliberately does not duplicate the text
    plen = {}
    for l in open(N7 / "pack.jsonl"):
        p = json.loads(l)
        plen[p["item_id"]] = (len(p["read"]), len(p["window"]))
    rows = []
    for l in open(N7 / f"verdicts_{tag}.jsonl"):
        v = json.loads(l)
        k = keys.get(v["item_id"])
        if k is None or v.get("score") is None:
            continue
        rl, wl = plen.get(v["item_id"], (0, 0))
        rows.append({**k, "score": float(v["score"]), "category": v.get("category", ""),
                     "read_len": rl, "window_len": wl})
    d = pd.DataFrame(rows)
    ar_p = N7 / "ar_baseline.jsonl"
    if ar_p.exists():
        ar = {}
        for l in open(ar_p):
            r = json.loads(l)
            if r.get("ar_cos") is not None:
                ar[r["item_id"]] = r["ar_cos"]
        d["ar_cos"] = d["item_id"].map(ar)
    return d


def auc(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Tie-corrected AUC via Mann-Whitney U, with a normal-approx SE.

    Scores are integers 0-3, so ties are the common case, not the exception -- a naive
    rank-without-ties AUC would be biased here.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()          # average ranks -> tie correction
    u = r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    a = u / (len(pos) * len(neg))
    se = np.sqrt(a * (1 - a) / min(len(pos), len(neg)))
    return float(a), float(se)


def cohen_kappa(a, b, weights: str | None = None) -> float:
    """Cohen's kappa. weights=None -> unweighted (categories); 'quadratic' -> ordinal 0-3 scores.

    The pre-registration asks for BOTH: unweighted on `category` and quadratic-weighted on `score`.
    Quadratic weighting is the right one for an ordinal scale -- a 2-vs-3 disagreement should not
    cost the same as a 0-vs-3.
    """
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return float("nan")
    obs = m / n
    exp = np.outer(m.sum(1), m.sum(0)) / n ** 2
    if weights == "quadratic":
        w = np.array([[((i - j) / max(k - 1, 1)) ** 2 for j in range(k)] for i in range(k)])
    else:
        w = 1.0 - np.eye(k)
    den = (w * exp).sum()
    return float(1.0 - (w * obs).sum() / den) if den > 0 else float("nan")


def gate(d: pd.DataFrame) -> dict:
    real = d[d.condition == "real"]
    shuf = d[d.condition == "shuffled"]
    dist = d[d.condition == "distant"]

    # G1/G2 are measured on the SAME comparison so they are directly comparable: the null items
    # reuse a real item's read text, so read quality is held fixed and only the pairing changes.
    a_j, se_j = auc(real["score"], shuf["score"])
    a_d, se_d = auc(real["score"], dist["score"])
    out = {
        "n": {c: int((d.condition == c).sum()) for c in ("real", "shuffled", "distant")},
        "mean_score": {c: round(float(d[d.condition == c]["score"].mean()), 3)
                       for c in ("real", "shuffled", "distant")},
        "G1_shuffled_auc": round(a_j, 4), "G1_se": round(se_j, 4),
        "distant_auc": round(a_d, 4), "distant_se": round(se_d, 4),
    }
    if "ar_cos" in d and d["ar_cos"].notna().any():
        a_ar, se_ar = auc(real["ar_cos"], shuf["ar_cos"])
        a_ard, _ = auc(real["ar_cos"], dist["ar_cos"])
        out["G2_ar_shuffled_auc"] = round(a_ar, 4)
        out["G2_ar_distant_auc"] = round(a_ard, 4)
        # difference of two AUCs on overlapping samples; independent-SE approx is conservative
        # enough for a gate that only asks "is the judge ahead of free?"
        out["G2_judge_minus_ar"] = round(a_j - a_ar, 4)
        out["G2_pass"] = bool(a_j > a_ar)
        out["judge_vs_ar_spearman"] = round(float(
            real[["score", "ar_cos"]].dropna().corr(method="spearman").iloc[0, 1]), 4)
    out["G1_pass"] = bool(a_j >= GATE_AUC)

    kp = N7 / "verdicts_phi.jsonl"
    if kp.exists():
        other = load("phi").set_index("item_id")["score"]
        j = d.set_index("item_id")["score"]
        common = j.index.intersection(other.index)
        if len(common) >= 100:
            jj = j.loc[common].astype(int).to_numpy()
            oo = other.loc[common].astype(int).to_numpy()
            kq = cohen_kappa(jj, oo, weights="quadratic")
            ku = cohen_kappa(jj, oo)
            cj = d.set_index("item_id")["category"].loc[common].fillna("").to_numpy()
            co = load("phi").set_index("item_id")["category"].reindex(common).fillna("").to_numpy()
            out["G3_kappa_quadratic_score"] = round(kq, 4)
            out["G3_kappa_unweighted_score"] = round(ku, 4)
            out["G3_pearson_score"] = round(float(np.corrcoef(jj, oo)[0, 1]), 4)
            out["G3_mean_llama"] = round(float(jj.mean()), 3)
            out["G3_mean_phi"] = round(float(oo.mean()), 3)
            out["G3_n"] = int(len(common))
            out["G3_pass"] = bool(kq >= GATE_KAPPA)
    out["gate_verdict"] = ("PASS" if all(out.get(g, False) for g in ("G1_pass", "G2_pass", "G3_pass"))
                           else "FAIL/INCOMPLETE")
    return out


STRIP_U = 0.70          # pre-registered: after_error = 1[u_read > 0.70]


def ht13(d: pd.DataFrame, arm: str = "primary") -> dict:
    """arm='primary' -> fixed 0.70 strip on every case (the frozen rule).
    arm='localized' -> per-case first-error boundary, trusted detectors only (clause-3 sensitivity).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    r = d[(d.condition == "real") & d.u_rel.notna()].copy()
    if arm == "primary":
        r["after_error"] = (r["u_rel"] > STRIP_U).astype(int)
    else:
        r = r[r.err_detector.isin(TRUSTED) & r.u_err.notna()]
        r["after_error"] = (r["u_rel"] >= r["u_err"]).astype(int)
    # a case that is all-before or all-after contributes nothing to a within-case contrast
    ok = r.groupby("case")["after_error"].nunique()
    r = r[r.case.isin(ok[ok > 1].index)]
    if r.case.nunique() < 6:
        return {"arm": arm, "n_cases": int(r.case.nunique()),
                "verdict": "NOT ADJUDICATED (too few cases)"}
    r["align"] = r["score"] / 3.0
    r["log_read_len"] = np.log(r["read_len"].clip(lower=1))
    r["u2"] = r["u_rel"] ** 2

    f = "align ~ correct * after_error + C(tier) + log_read_len + u_rel + u2"
    m = smf.gee(f, groups=r["case"], data=r, cov_struct=sm.cov_struct.Exchangeable()).fit()
    inter = "correct:after_error"
    b = float(m.params.get(inter, np.nan)); se = float(m.bse.get(inter, np.nan))
    p = float(m.pvalues.get(inter, np.nan))

    # before-error simple effect = the `correct` main effect at after_error == 0
    bb = float(m.params.get("correct", np.nan)); bse = float(m.bse.get("correct", np.nan))
    before_ci = (bb - 1.96 * bse, bb + 1.96 * bse)

    cells = (r.groupby(["correct", "after_error"])["align"].agg(["mean", "size"])
             .round(4).reset_index().to_dict("records"))
    ci = (b - 1.96 * se, b + 1.96 * se)
    ok_inter = (p < 0.05) and (b < 0) and not (ci[0] <= 0 <= ci[1])
    ok_local = before_ci[0] <= 0 <= before_ci[1]
    # pre-registered second refute trigger: "the before-error effect matches the after-error effect
    # in magnitude" -> the drop is global, not localized, and is a weaker claim
    after_effect = bb + b
    global_like = abs(bb) > 0 and abs(after_effect - bb) < 0.25 * abs(bb)
    return {"arm": arm, "n_reads": int(len(r)), "n_cases": int(r.case.nunique()),
            "beta_interaction": round(b, 4), "se": round(se, 4), "p": p,
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "before_error_correct_effect": round(bb, 4),
            "before_error_ci95": [round(before_ci[0], 4), round(before_ci[1], 4)],
            "after_error_correct_effect": round(after_effect, 4),
            "cells": cells,
            "verdict": ("CONFIRM" if (ok_inter and ok_local) else
                        "REFUTE (before- and after-error effects match in magnitude — global, not localized)"
                        if global_like else
                        "EQUIVOCAL (interaction holds but the drop does not localize)" if ok_inter
                        else "REFUTE (interaction CI includes 0)")}


def main() -> int:
    d = load("llama")
    print(f"loaded {len(d)} scored items\n")
    g = gate(d)
    print("=== N7 INSTRUMENT GATE ===")
    print(json.dumps(g, indent=1))

    out = {"gate": g}
    if g["gate_verdict"] == "PASS":
        out["ht13_primary"] = ht13(d, "primary")
        out["ht13_localized"] = ht13(d, "localized")
        for k in ("ht13_primary", "ht13_localized"):
            print(f"\n=== HT13 / {k} ===")
            print(json.dumps(out[k], indent=1))
    else:
        # still compute it, clearly marked, so the log can say what it WOULD have been --
        # never as an adjudication
        out["ht13_not_adjudicated"] = ht13(d, "primary")
        print("\n=== GATE DID NOT PASS -> HT13 NOT ADJUDICATED ===")
        print("(exploratory value below, explicitly NOT a verdict)")
        print(json.dumps(out["ht13_not_adjudicated"], indent=1))

    p = N7 / "n7_results.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
