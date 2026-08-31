"""B6, judge-free — does `rt_cos` add anything over reply LENGTH in predicting correctness?

WHY THIS WAY. B6 ("does read/CoT agreement predict correctness") was never adjudicated because it
inherited HT13's judge instrument null (kappa = 0.049). But the useful half of the question needs
no judge at all: `rt_cos` is a continuous per-read number, correctness is binary, and the one
thing already known about `rt_cos` is that it correlates +0.28-0.31 with how much text the
verbalizer produced. That makes reply length the mandatory covariate, not an afterthought — a
"faithfulness predicts correctness" result that a token count reproduces is a length result.

N6/HT12 refuted the marginal relationship (beta = -0.0002 [-0.0051, +0.0047], p = .93). This asks
the incremental question instead, which is the one that decides whether `rt_cos` earns its place
in any model: fit correctness on length alone, then on length + faithfulness, and ask what the
second buys.

PRE-STATED RULE, frozen before the fit:
  `rt_cos` ADDS SOMETHING iff the likelihood-ratio test of (length + rt_cos) vs (length) has
  p < 0.05 AND cross-validated AUC improves (delta > 0). Both, not either: an LRT can reach
  significance on 512 cases for an effect that predicts nothing out of sample, and a CV AUC can
  drift upward by chance.
Otherwise: `rt_cos` ADDS NOTHING over length, and every model in the programme that carries it
should carry the token count instead.

`act_norm` is fitted the same way as a second candidate, since N13 found instability and
describability are nearly the same measurement (r = -0.958) and it is worth knowing whether either
survives the length control.

No GPU. Seconds. Env `nla-mi` (sklearn, scipy).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
SEED = 20260724
N_FOLDS = 5
ALPHA = 0.05


def build(readings_p: Path, cases_p: Path, read_class: str | None) -> tuple[np.ndarray, ...]:
    """One row per case: correctness, reply length, and the case's mean rt_cos / act_norm."""
    agg: dict[str, list[dict]] = defaultdict(list)
    for l in open(readings_p):
        if not l.strip():
            continue
        r = json.loads(l)
        if read_class and r.get("read_class") != read_class:
            continue
        agg[r["case_id"]].append(r)

    y, feats, ids = [], [], []
    for l in open(cases_p):
        if not l.strip():
            continue
        c = json.loads(l)
        rs = agg.get(c["case_id"]) or []
        # A case needs an outcome, a length, and at least one read of the requested class.
        if c.get("correct") is None or not c.get("reply_tokens") or not rs:
            continue
        rt = [x["rt_cos"] for x in rs if x.get("rt_cos") is not None]
        an = [x["act_norm"] for x in rs if x.get("act_norm") is not None]
        if not rt or not an:
            continue
        y.append(int(c["correct"]))
        feats.append([float(c["reply_tokens"]), st.mean(rt), st.mean(an)])
        ids.append(c["case_id"])
    return np.array(y), np.array(feats), ids


def loglik(model, X, y) -> float:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_compare(y: np.ndarray, X_base: np.ndarray, X_full: np.ndarray) -> dict[str, Any]:
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    def cv_auc(X: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(penalty=None, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, oof))

    sc = StandardScaler().fit(X_base)
    mb = LogisticRegression(penalty=None, max_iter=2000).fit(sc.transform(X_base), y)
    sf = StandardScaler().fit(X_full)
    mf = LogisticRegression(penalty=None, max_iter=2000).fit(sf.transform(X_full), y)

    ll0, ll1 = loglik(mb, sc.transform(X_base), y), loglik(mf, sf.transform(X_full), y)
    df = X_full.shape[1] - X_base.shape[1]
    lr = 2 * (ll1 - ll0)
    p = float(chi2.sf(lr, df)) if lr > 0 else 1.0
    null_ll = float(np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())))
    a0, a1 = cv_auc(X_base), cv_auc(X_full)
    return {
        "lr_stat": round(lr, 4), "df": df, "lrt_p": round(p, 5),
        "mcfadden_r2_base": round(1 - ll0 / null_ll, 5),
        "mcfadden_r2_full": round(1 - ll1 / null_ll, 5),
        "delta_mcfadden_r2": round((1 - ll1 / null_ll) - (1 - ll0 / null_ll), 5),
        "cv_auc_base": round(a0, 4), "cv_auc_full": round(a1, 4),
        "delta_cv_auc": round(a1 - a0, 4),
        "coefs_full": {n: round(float(c), 4)
                       for n, c in zip(("length", "candidate"), mf.coef_[0])},
        "verdict": "ADDS SOMETHING" if (p < ALPHA and a1 - a0 > 0) else "ADDS NOTHING",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default=str(PROJ / "data/nla/unified/readings.jsonl"))
    ap.add_argument("--cases", default=str(PROJ / "data/nla/unified/cases.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/unified/b6_length_vs_faithfulness.json"))
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    rep: dict[str, Any] = {"experiment": "b6_length_vs_faithfulness_judgefree",
                           "seed": SEED, "folds": N_FOLDS, "alpha": ALPHA, "strata": {}}

    # All reads pooled is the primary; the per-class fits are secondary and exist because a
    # signal confined to one read class would be invisible in the pooled mean.
    for label, rc in (("all_reads", None), ("reasoning", "reasoning"),
                      ("answer_line", "answer_line"), ("code_token", "code_token")):
        y, X, ids = build(Path(args.readings), Path(args.cases), rc)
        if len(y) < 50 or len(set(y.tolist())) < 2:
            rep["strata"][label] = {"n": int(len(y)), "note": "too few cases or one class only"}
            continue
        block = {"n": int(len(y)), "accuracy": round(float(y.mean()), 4),
                 "length_alone_auc": round(float(roc_auc_score(y, X[:, 0])), 4),
                 "rt_cos_alone_auc": round(float(roc_auc_score(y, X[:, 1])), 4),
                 "act_norm_alone_auc": round(float(roc_auc_score(y, X[:, 2])), 4)}
        block["rt_cos_over_length"] = fit_compare(y, X[:, [0]], X[:, [0, 1]])
        block["act_norm_over_length"] = fit_compare(y, X[:, [0]], X[:, [0, 2]])
        rep["strata"][label] = block

    # Multiplicity. The pre-registered primary is the pooled `all_reads` stratum; the per-class
    # fits are secondary and there are several of them, so their LRT p-values carry BH-FDR. The
    # primary is excluded from its own correction — it is one declared test, not a discovery
    # among many. Without this, "rt_cos helps at the answer line" is a garden-of-forking-paths
    # claim over four read classes.
    sec = [(lab, cand) for lab, b in rep["strata"].items() if "note" not in b and lab != "all_reads"
           for cand in ("rt_cos_over_length", "act_norm_over_length")]
    ps = [rep["strata"][lab][cand]["lrt_p"] for lab, cand in sec]
    if ps:
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        adj, prev = [0.0] * len(ps), 1.0
        for rank, i in enumerate(reversed(order), start=1):
            k = len(ps) - rank + 1
            prev = min(prev, ps[i] * len(ps) / k)
            adj[i] = round(min(prev, 1.0), 6)
        for (lab, cand), q in zip(sec, adj):
            rep["strata"][lab][cand]["lrt_q_bh"] = q
            # A secondary stratum only counts if it survives correction AND still helps out of
            # sample; the verdict is rewritten so the JSON cannot be quoted without the caveat.
            if rep["strata"][lab][cand]["verdict"] == "ADDS SOMETHING" and q >= ALPHA:
                rep["strata"][lab][cand]["verdict"] = "ADDS NOTHING (fails BH-FDR)"
            elif rep["strata"][lab][cand]["verdict"] == "ADDS SOMETHING":
                rep["strata"][lab][cand]["verdict"] = "SECONDARY LEAD (survives BH-FDR)"
    rep["primary_stratum"] = "all_reads"
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
