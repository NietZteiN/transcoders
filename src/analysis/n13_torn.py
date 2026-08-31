"""N13 Stage 4 — does any internal signal track TORN-ness rather than incorrectness?

THE CELL HT12 COULD NOT SEE. HT12 refuted `rt_cos ~ correctness` (beta = -0.00022, p = .93, and
on a length-matched corpus the raw gap collapsed from +0.0145 to +0.0035 because wrong traces run
~1.6x longer and rt_cos correlates +0.277 with read length). But `correct = False` merges two
opposite internal states:

                    consistent          inconsistent
        right       confidently right   lucky/unstable
        wrong       CONFIDENTLY WRONG   TORN            <- the contrast

Stage 1 supplies the missing axis by sampling K answers per item, so "torn" is now measurable and
the bottom row can be split. A metric that merely tracks correctness cannot separate that row;
one that tracks torn-ness can.

THREE ANALYSES, IN DECREASING ORDER OF HOW MUCH THEY COULD SURPRISE US:

1. **The horse race** (item level, one row per case, so no clustering is needed — entropy is an
   item property). Nested models: entropy ~ length, then + rt_cos, + act_norm, + instability.
   Reported as incremental R^2 with a case bootstrap, because the whole question is whether any
   internal signal beats the length meter that explained away HT12.

2. **The 2x2**, matched on reply length. Torn and confidently-wrong items are compared on every
   internal metric via AUC. Length matching is not optional: torn items are the ones the model
   thrashed on, so they run longer by construction, and an unmatched AUC would rediscover length.
   The `length_auc_confound_check` idiom from N10b is applied to the matched set — if length
   still separates the groups, nothing else in the table can be trusted.

3. **The engineered-ambiguity contrast** (read level, clustered on case): `adversarial` vs
   `l1_neutral` instability. This is the sharpest test in the whole experiment because the two
   strata are matched in token form and both carry renamed identifiers, differing only in whether
   the name asserts a confident wrong meaning. It also separates the two ways instability can
   arise — superposition (two readings) vs emptiness (no reading) — which the aggregate cannot.

EXPLORATORY THROUGHOUT. No pre-registration, and this is deliberately NOT added to the
{HT12, HT13, HT14} family whose BH-FDR correction is still unapplied. Effect sizes with
case-clustered CIs; any survivor earns a frozen confirmatory test on held-out data, it does not
get to claim significance here.

Env `transcoders-mi`. CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJ = _HERE.parent.parent
sys.path.insert(0, str(PROJ / "nla" / "src"))

from n10b_stats import auc, auc_ci  # noqa: E402  (reuse: ties-averaged AUC + bootstrap CI)

SEED = 20260724
MAX_LOG_RATIO = math.log(1.5)   # same hard length-matching threshold as match_controls.py
N_BOOT = 2000

# Torn vs committed-wrong, defined on modal SHARE rather than distinct-answer count.
#
# The first draft asked for n_distinct <= 1 (the same answer all 8 times) as "confidently wrong"
# and that cell came back EMPTY: among 43 modal-wrong items, n_distinct ran 2-8 with a median
# modal share of 0.375, while correct items sat at 0.875 with 28 of them perfectly consistent.
# **At T = 0.8 this model is almost never firmly and wrongly committed** — being wrong and being
# unstable are nearly the same condition here. That is a finding about the corpus, not a
# threshold artifact, and it is reported as one below.
#
# So the contrast is necessarily RELATIVE: the most-committed wrong items against the most-split
# wrong items. modal_share is the right axis because it directly encodes "kept giving the same
# wrong answer" = one reading, firmly held, mistaken. The middle is left unlabelled rather than
# forced into a bin.
TORN_MAX_MODAL_SHARE = 0.25       # no single answer won more than a quarter of the samples
COMMITTED_MIN_MODAL_SHARE = 0.5   # one wrong answer took at least half


def load_stage1(p: Path) -> dict[str, dict]:
    out = {}
    for line in open(p):
        r = json.loads(line)
        if "error" not in r:
            out[r["task_key"]] = r
    return out


def load_act_norm(p: Path) -> dict[str, list[dict]]:
    """case -> per-position records (act_norm joined to banked cls/rt_cos)."""
    out: dict[str, list[dict]] = {}
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if "error" not in r:
            out[r["case"]] = r["norms"]
    return out


def load_instability(p: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if "error" not in r:
            out[r["case"]].append(r)
    return out


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with intercept; returns (beta, R^2)."""
    A = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def horse_race(rows: list[dict], preds: list[str], n_perm: int = 1000) -> dict[str, Any]:
    """Nested incremental R^2, tested by PERMUTATION rather than by a bootstrap CI on itself.

    The baseline is log reply length ALONE, because that is the variable that explained away
    HT12. Every internal signal has to beat it, not merely correlate with the outcome.

    **Why not a bootstrap CI on delta R^2.** The first version of this function bootstrapped the
    increment and asked whether the CI excluded zero. That test is vacuous: adding a predictor to
    an OLS fit can never LOWER in-sample R^2, so delta R^2 >= 0 by construction and its
    resampling distribution is bounded below by ~0. The 2.5th percentile is therefore always >= 0
    and the CI "excludes zero" no matter what. A dress rehearsal on synthetic data confirmed it —
    a predictor generated independently of the outcome still returned [1e-05, 0.019] and would
    have been reported as a real improvement.

    Two honest tests replace it:
      * **permutation p-value** — shuffle the added column and refit. Under the null, an
        irrelevant predictor still buys some R^2 by chance, and this measures exactly how much,
        so the observed increment is judged against the right reference distribution.
      * **coefficient bootstrap CI** — a slope CAN be negative, so this interval is free to
        contain zero and is meaningful on its own terms.
    """
    rows = [r for r in rows if all(r.get(p) is not None for p in preds)
            and r.get("entropy") is not None]
    if len(rows) < 20:
        return {"n": len(rows), "note": "too few complete rows for a horse race"}
    y = np.array([r["entropy"] for r in rows], float)
    cols = {p: np.array([r[p] for r in rows], float) for p in preds}

    ladder, prev = [], []
    rng = np.random.default_rng(SEED)
    n = len(rows)
    for p in preds:
        cur = prev + [p]
        beta, r2_cur = ols(np.column_stack([cols[c] for c in cur]), y)
        _, r2_prev = (ols(np.column_stack([cols[c] for c in prev]), y) if prev else (None, 0.0))
        inc = r2_cur - r2_prev
        coef = float(beta[-1])          # the added predictor is the last column

        # permutation null: break only the added predictor's link to y, keep the rest intact
        hits = 0
        for _ in range(n_perm):
            shuffled = cols[p][rng.permutation(n)]
            mats = [cols[c] for c in prev] + [shuffled]
            _, r2_p = ols(np.column_stack(mats), y)
            if (r2_p - r2_prev) >= inc:
                hits += 1
        p_perm = (hits + 1) / (n_perm + 1)

        # bootstrap the COEFFICIENT (unbounded in sign, so its CI can contain zero)
        cboots = []
        for _ in range(400):
            idx = rng.integers(0, n, n)
            try:
                b, _ = ols(np.column_stack([cols[c][idx] for c in cur]), y[idx])
                cboots.append(float(b[-1]))
            except Exception:
                continue
        cboots.sort()
        ladder.append({
            "added": p, "model": " + ".join(cur),
            "r2": round(r2_cur, 4), "delta_r2": round(inc, 5),
            "coef": round(coef, 5),
            "coef_ci95": [round(cboots[int(0.025 * len(cboots))], 5),
                          round(cboots[int(0.975 * len(cboots))], 5)] if cboots else None,
            "p_perm": round(p_perm, 4),
            "beats_baseline": bool(p_perm < 0.05),
        })
        prev = cur
    return {"n": len(rows), "n_perm": n_perm, "ladder": ladder}


def match_on_length(a: list[dict], b: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy 1-1 length matching, same |log ratio| <= log(1.5) rule as match_controls.py.

    Torn items ran longer by construction — the model thrashed — so without this every AUC
    below would be rediscovering reply length.
    """
    used, pairs = set(), []
    for x in sorted(a, key=lambda r: r["task_key"]):
        best, bestd = None, None
        for y in b:
            if y["task_key"] in used:
                continue
            lx = max(x.get("banked_reply_tokens") or 1, 1)
            ly = max(y.get("banked_reply_tokens") or 1, 1)
            d = abs(math.log(lx / ly))
            if d <= MAX_LOG_RATIO and (bestd is None or d < bestd):
                best, bestd = y, d
        if best is not None:
            used.add(best["task_key"])
            pairs.append((x, best))
    return pairs


def cluster_bootstrap_diff(groups: dict[str, list[tuple[str, float]]],
                           n_boot: int = N_BOOT) -> dict[str, Any]:
    """Mean difference between two strata, resampling CASES not reads.

    Reads are nested in cases; resampling reads would give CIs that are too narrow exactly where
    within-case correlation is highest. This is the caveat the 2026-08-06 token-type analysis
    logged against its own bootstrap, so it is fixed here rather than repeated.
    """
    (na, ra), (nb, rb) = list(groups.items())
    by_case_a, by_case_b = defaultdict(list), defaultdict(list)
    for c, v in ra:
        by_case_a[c].append(v)
    for c, v in rb:
        by_case_b[c].append(v)
    ca, cb = list(by_case_a), list(by_case_b)
    if not ca or not cb:
        return {"note": "a stratum is empty"}
    obs = float(np.mean([v for vs in by_case_a.values() for v in vs]) -
                np.mean([v for vs in by_case_b.values() for v in vs]))
    rng = random.Random(SEED)
    boots = []
    for _ in range(n_boot):
        sa = [v for c in (rng.choice(ca) for _ in ca) for v in by_case_a[c]]
        sb = [v for c in (rng.choice(cb) for _ in cb) for v in by_case_b[c]]
        if sa and sb:
            boots.append(float(np.mean(sa) - np.mean(sb)))
    boots.sort()
    return {
        f"mean_{na}": round(float(np.mean([v for vs in by_case_a.values() for v in vs])), 4),
        f"mean_{nb}": round(float(np.mean([v for vs in by_case_b.values() for v in vs])), 4),
        f"n_{na}": sum(len(v) for v in by_case_a.values()), f"n_cases_{na}": len(ca),
        f"n_{nb}": sum(len(v) for v in by_case_b.values()), f"n_cases_{nb}": len(cb),
        "diff": round(obs, 4),
        "diff_ci95": [round(boots[int(0.025 * len(boots))], 4),
                      round(boots[int(0.975 * len(boots))], 4)] if boots else None,
        "excludes_zero": bool(boots and (boots[int(0.025 * len(boots))] > 0
                                         or boots[int(0.975 * len(boots))] < 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy", default=str(PROJ / "data/nla/n13/answer_entropy.jsonl"))
    ap.add_argument("--act-norm", default=str(PROJ / "data/nla/n13/act_norm.jsonl"))
    ap.add_argument("--instability", default=str(PROJ / "data/nla/n13/read_instability.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n13/n13_torn.json"))
    args = ap.parse_args()

    s1 = load_stage1(Path(args.entropy))
    an = load_act_norm(Path(args.act_norm))
    inst = load_instability(Path(args.instability))
    rep: dict[str, Any] = {"experiment": "n13_torn", "seed": SEED, "exploratory": True,
                           "n_items_stage1": len(s1), "n_cases_act_norm": len(an),
                           "n_cases_instability": len(inst)}

    # ── Gate 1a: is there entropy to explain at all? ──────────────────────────────
    ents = [r["entropy"] for r in s1.values()]
    frac_zero = sum(e == 0 for e in ents) / max(len(ents), 1)
    rep["gate_1a_entropy_spread"] = {
        "n": len(ents), "frac_perfectly_consistent": round(frac_zero, 4),
        "mean_entropy": round(float(np.mean(ents)), 4) if ents else None,
        "median_entropy": round(float(np.median(ents)), 4) if ents else None,
        "pass": bool(frac_zero < 0.80),
        "rule": "stop if >=80% of items are perfectly consistent — no variance to explain",
    }

    # ── Gate 1b: does the trap tier raise entropy vs matched L0? ──────────────────
    by_tier = defaultdict(list)
    for r in s1.values():
        by_tier[r.get("tier")].append(r)
    l1b, l0 = by_tier.get("L1b", []), by_tier.get("L0", [])
    pairs = match_on_length(l1b, l0)
    if pairs:
        d = [p[0]["entropy"] - p[1]["entropy"] for p in pairs]
        rng = random.Random(SEED)
        bs = sorted(float(np.mean([rng.choice(d) for _ in d])) for _ in range(N_BOOT))
        rep["gate_1b_trap_raises_entropy"] = {
            "n_pairs": len(pairs),
            "mean_L1b": round(float(np.mean([p[0]["entropy"] for p in pairs])), 4),
            "mean_L0": round(float(np.mean([p[1]["entropy"] for p in pairs])), 4),
            "mean_diff": round(float(np.mean(d)), 4),
            "diff_ci95": [round(bs[int(0.025 * N_BOOT)], 4), round(bs[int(0.975 * N_BOOT)], 4)],
            "pass": bool(np.mean(d) > 0),
            "note": "length-matched so the trap effect is not a trace-length effect",
        }
    else:
        rep["gate_1b_trap_raises_entropy"] = {"note": "no length-matched L1b/L0 pairs"}

    # ── per-item internal signals ─────────────────────────────────────────────────
    items = []
    for tk, r in s1.items():
        norms = [x for x in an.get(tk, []) if not x.get("norm_outlier")]
        ii = inst.get(tk, [])
        row = dict(r)
        row["task_key"] = tk
        row["log_reply"] = math.log(max(r.get("banked_reply_tokens") or 1, 1))
        row["mean_rt_cos"] = (float(np.mean([x["rt_cos"] for x in norms if x.get("rt_cos")]))
                              if any(x.get("rt_cos") for x in norms) else None)
        row["mean_act_norm"] = float(np.mean([x["act_norm"] for x in norms])) if norms else None
        row["mean_instability"] = (float(np.mean([1 - x["ar_cos_mean"] for x in ii
                                                  if x.get("ar_cos_mean") is not None]))
                                   if ii else None)
        items.append(row)

    rep["horse_race"] = {
        "baseline": "log reply tokens — the variable that explained away HT12",
        "rt_cos": horse_race(items, ["log_reply", "mean_rt_cos"]),
        "rt_cos_then_act_norm": horse_race(items, ["log_reply", "mean_rt_cos", "mean_act_norm"]),
        "full_with_instability": horse_race(
            items, ["log_reply", "mean_rt_cos", "mean_act_norm", "mean_instability"]),
    }

    # ── the 2x2: torn vs confidently wrong, length matched ────────────────────────
    wrong = [r for r in items if not r["modal_correct"]]
    right = [r for r in items if r["modal_correct"]]
    torn = [r for r in wrong if r["modal_share"] <= TORN_MAX_MODAL_SHARE]
    conf = [r for r in wrong if r["modal_share"] >= COMMITTED_MIN_MODAL_SHARE]
    mp = match_on_length(torn, conf)

    # The premise check: how full is the cell the whole 2x2 depends on?
    rep["confidently_wrong_cell_occupancy"] = {
        "n_wrong": len(wrong),
        "n_wrong_perfectly_consistent": sum(r["n_distinct"] == 1 for r in wrong),
        "n_right_perfectly_consistent": sum(r["n_distinct"] == 1 for r in right),
        "mean_entropy_wrong": round(float(np.mean([r["entropy"] for r in wrong])), 4)
                              if wrong else None,
        "mean_entropy_right": round(float(np.mean([r["entropy"] for r in right])), 4)
                              if right else None,
        # Self-consistency as an accuracy signal: how well does answer entropy alone separate
        # wrong from right items? This is the behavioural anchor for everything below — and if
        # it is high, entropy is partly a correctness proxy, so the horse race is predicting
        # something correlated with correctness rather than something orthogonal to it.
        "entropy_auc_wrong_vs_right": (
            round(auc([0] * len(right) + [1] * len(wrong),
                      [r["entropy"] for r in right] + [r["entropy"] for r in wrong]), 4)
            if right and wrong else None),
        "note": ("If n_wrong_perfectly_consistent is ~0, the 'confidently wrong' state barely "
                 "exists at this sampling temperature: on this corpus being wrong and being "
                 "unstable nearly coincide, so the 2x2 below is a relative contrast between the "
                 "most-committed and most-split wrong items, not an absolute one."),
    }

    cell = {
        "n_wrong": len(wrong), "n_torn": len(torn), "n_committed_wrong": len(conf),
        "n_matched_pairs": len(mp),
        "definition": {"torn": f"modal answer wrong AND modal_share <= {TORN_MAX_MODAL_SHARE}",
                       "committed_wrong": f"modal wrong AND modal_share >= "
                                          f"{COMMITTED_MIN_MODAL_SHARE}"},
    }
    if mp:
        lab = [1] * len(mp) + [0] * len(mp)
        for metric in ("mean_rt_cos", "mean_act_norm", "mean_instability"):
            sc_t = [p[0].get(metric) for p in mp]
            sc_c = [p[1].get(metric) for p in mp]
            keep = [i for i in range(len(mp)) if sc_t[i] is not None and sc_c[i] is not None]
            if len(keep) < 8:
                cell[metric] = {"note": f"only {len(keep)} complete pairs"}
                continue
            sc = [sc_t[i] for i in keep] + [sc_c[i] for i in keep]
            lb = [1] * len(keep) + [0] * len(keep)
            cell[metric] = {"n_pairs": len(keep), "auc": round(auc(lb, sc), 4),
                            "auc_ci95": auc_ci(lb, sc),
                            "mean_torn": round(float(np.mean([sc_t[i] for i in keep])), 4),
                            "mean_committed": round(float(np.mean([sc_c[i] for i in keep])), 4)}
        # the N10b idiom: matching must leave length uninformative
        lens = ([math.log(max(p[0].get("banked_reply_tokens") or 1, 1)) for p in mp] +
                [math.log(max(p[1].get("banked_reply_tokens") or 1, 1)) for p in mp])
        cell["length_auc_confound_check"] = round(auc(lab, lens), 4)
        cell["confound_note"] = ("Torn items thrash and therefore run longer; if this is far "
                                 "from 0.5 the matching failed and no AUC above is meaningful.")
    rep["torn_vs_confidently_wrong"] = cell

    # ── the engineered-ambiguity contrast (read level, clustered on case) ─────────
    #
    # Run on TWO measures with very different scaling. Early rows showed AR-space cosine spanning
    # only ~0.971-0.994 — the documented AR-clustering caveat, where reconstructions of different
    # sentences sit close together regardless of content. Relative ordering can still be
    # informative on a compressed scale, but a conclusion resting on it alone would be fragile,
    # and absolute effect sizes are not comparable to anything. The lexical Jaccard has a much
    # wider range (~0.37 mean) and depends on no learned component at all, so agreement between
    # the two is what makes a class difference credible rather than an artifact of AR geometry.
    MEASURES = {
        "ar_space": ("ar_cos_mean", lambda x: 1 - x),      # 1 - cosine  = instability
        "lexical": ("jaccard_mean", lambda x: 1 - x),      # 1 - Jaccard = instability
    }
    rep["instability_by_class"] = {}
    for mname, (field, to_instability) in MEASURES.items():
        strata: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for case, rs in inst.items():
            for r in rs:
                if r.get(field) is None:
                    continue
                strata[r["cls"]].append((case, to_instability(r[field])))
        rep["instability_by_class"][mname] = {
            c: {"mean": round(float(np.mean([v for _, v in vs])), 4), "n": len(vs),
                "n_cases": len({c_ for c_, _ in vs}),
                "min": round(float(np.min([v for _, v in vs])), 4),
                "max": round(float(np.max([v for _, v in vs])), 4)}
            for c, vs in sorted(strata.items())
        }
        if "adversarial" in strata and "l1_neutral" in strata:
            rep.setdefault("adversarial_vs_l1_neutral", {})[mname] = cluster_bootstrap_diff(
                {"adversarial": strata["adversarial"], "l1_neutral": strata["l1_neutral"]})
        if "fn_orig" in strata and "adversarial" in strata:
            rep.setdefault("floor_gate", {})[mname] = cluster_bootstrap_diff(
                {"adversarial": strata["adversarial"], "fn_orig": strata["fn_orig"]})
    if "adversarial_vs_l1_neutral" in rep:
        rep["adversarial_vs_l1_neutral"]["interpretation"] = (
            "adversarial > l1_neutral: instability tracks SUPERPOSITION (a decoy asserting a "
            "second reading). adversarial ~= l1_neutral: it is an EMPTINESS meter — vacant "
            "names give the decoder nothing, which is a different internal state entirely. "
            "The claim needs BOTH measures to agree.")
    if "floor_gate" in rep:
        rep["floor_gate"]["rule"] = ("adversarial must be MORE unstable than the fn_orig floor, "
                                     "else the measure is decoder stochasticity. Confounded with "
                                     "tier and token form — see read_instability.py docstring.")

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
