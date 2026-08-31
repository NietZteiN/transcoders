"""N8 / HT14 — does the answer surface in the NLA reads earlier in correct runs?

Confirmatory. Rules frozen in log/nla-harness/2026-08-06_n6-n8-preregistration.md before this ran.
Measured with the NLA itself (the user's explicit choice over a logit-lens); the text-based onset is
kept strictly as CALIBRATION and never substituted for the primary.

"The answer" is the model's OWN eventual answer (`model_answer`), not the ground truth. That is what
makes the correct-vs-wrong comparison meaningful: it measures when the eventual output begins to
show up internally, not whether that output was right.

    mentions_answer(read, answer) = literal word-boundary match
                                  | numeric-value match (tol 1e-9)
                                  | container match (>= 2/3 of elements, in order)

PRIMARY STATISTIC IS THE EXCESS OVER A FOREIGN-ANSWER NULL. For every read we also test 20 foreign
answers drawn from the same tier with matched type and digit count, and report
`f_own(bin) - f_foreign(bin)`. Without this, "the read is full of small integers" counts as a hit and
every trajectory looks the same. The null is the whole point of the measurement.

ONSET AND CENSORING. onset = the first of 10 equal-width `u` bins where the case's own hit rate
>= 0.5 AND stays >= 0.5 in the next bin. Never reached -> RIGHT-CENSORED, not dropped. This is how
the ~58% of wrong traces that never state their answer stay in the analysis instead of biasing the
group by their absence. Test: tier-stratified log-rank on the censored onset.

CONFIRM = log-rank p < 0.05 (BH-FDR within the family) AND the correct group's median onset is
>= 0.10 of relative position earlier. REFUTE = p > 0.05 or the direction reverses.

GROUP STRUCTURE (the HT12 lesson, applied in advance): onset is ONE number per case, so `correct` is
again constant within case. The frozen primary is already a CASE-LEVEL survival test with no per-case
random effect, so it is immune -- and it must stay that way. Do not "upgrade" this to a read-level
mixed model; that is exactly the move that broke HT12.

`lifelines` is not in the env, so Kaplan-Meier and the stratified log-rank are implemented here
directly (~40 lines) rather than adding a dependency mid-project.

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
SEED = 20260724
N_BINS = 10
HIT_THRESH = 0.5
N_FOREIGN = 20
N_BOOT = 4000
CHEAP = {"0", "1", "2", "-1", "true", "false", "none", "null", "[]", "{}"}
MIN_ANSWER_CHARS = 3

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _as_number(s: str):
    try:
        return float(s.strip())
    except Exception:
        return None


def _as_container(s: str):
    try:
        v = ast.literal_eval(s.strip())
        return list(v) if isinstance(v, (list, tuple)) else None
    except Exception:
        return None


def answer_type(ans: str) -> str:
    if _as_container(ans) is not None:
        return "container"
    if _as_number(ans) is not None:
        return "numeric"
    return "string"


def mentions_answer(read: str, ans: str) -> bool:
    """Frozen matcher: literal | numeric-value | container (>= 2/3 elements, in order)."""
    if not read or not ans:
        return False
    low = read.lower()
    a = ans.strip()

    # 1. literal, word-boundary (escaped: answers contain brackets/commas)
    if re.search(rf"(?<!\w){re.escape(a.lower())}(?!\w)", low):
        return True

    # 2. numeric value, tolerance 1e-9 -- catches 3 vs 3.0 vs 3.00
    av = _as_number(a)
    if av is not None:
        for m in _NUM.finditer(read):
            v = _as_number(m.group(0))
            if v is not None and abs(v - av) <= 1e-9:
                return True
        return False

    # 3. container: >= 2/3 of the elements present IN ORDER
    els = _as_container(a)
    if els:
        pos, hit = 0, 0
        for e in els:
            t = str(e).strip().strip("'\"")
            if not t:
                continue
            m = re.search(rf"(?<!\w){re.escape(t.lower())}(?!\w)", low[pos:])
            if m:
                hit += 1
                pos += m.end()
        return hit >= math.ceil(len(els) * 2 / 3)
    return False


# ------------------------------------------------------------------ survival
def km(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier. times = onset (or censoring time), events = 1 if onset observed."""
    order = np.argsort(times, kind="mergesort")
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    n = len(t)
    ts, s, surv = [], 1.0, []
    at_risk = n
    i = 0
    while i < n:
        tt = t[i]
        j = i
        d = c = 0
        while j < n and t[j] == tt:
            d += int(e[j] == 1)
            c += int(e[j] == 0)
            j += 1
        if d > 0 and at_risk > 0:
            s *= (1 - d / at_risk)
        ts.append(tt)
        surv.append(s)
        at_risk -= (d + c)
        i = j
    return np.array(ts), np.array(surv)


def median_survival(times: np.ndarray, events: np.ndarray) -> float:
    """First time S(t) <= 0.5; nan if the curve never reaches it (heavy censoring)."""
    ts, s = km(times, events)
    idx = np.where(s <= 0.5)[0]
    return float(ts[idx[0]]) if len(idx) else float("nan")


def logrank_stratified(df: pd.DataFrame, group: str = "correct",
                       strata: str = "tier") -> tuple[float, float, float]:
    """Tier-stratified log-rank. Returns (chi2, p, O1-E1). Handles ties (only 10 distinct
    onset values exist, so tied event times are the norm here, not an edge case)."""
    O1 = E1 = V = 0.0
    for _, sdf in df.groupby(strata):
        t = sdf["onset"].to_numpy(float)
        e = sdf["event"].to_numpy(int)
        g = (sdf[group].to_numpy(int) == 1)
        for tt in np.unique(t[e == 1]):
            at_risk = t >= tt
            n_t = at_risk.sum()
            n1 = (at_risk & g).sum()
            d_t = ((t == tt) & (e == 1)).sum()
            d1 = ((t == tt) & (e == 1) & g).sum()
            if n_t <= 1 or d_t == 0:
                continue
            O1 += d1
            E1 += d_t * n1 / n_t
            V += (d_t * (n_t - d_t) * n1 * (n_t - n1)) / (n_t ** 2 * (n_t - 1))
    if V <= 0:
        return float("nan"), float("nan"), float("nan")
    chi2 = (O1 - E1) ** 2 / V
    from scipy.stats import chi2 as chi2_dist
    return float(chi2), float(chi2_dist.sf(chi2, 1)), float(O1 - E1)


def _len_bucket(a: str) -> int:
    """Coarse length bucket so length-matching does not shatter the pool."""
    return min(int(math.log2(max(len(a), 1))), 6)


def _key(tier: str, a: str, match_length: bool) -> tuple:
    k = (tier, answer_type(a), len(re.findall(r"\d", a)))
    return k + ((_len_bucket(a),) if match_length else ())


# ------------------------------------------------------------------ pipeline
def build(dense_path: Path, caps_path: Path, match_length: bool = False
          ) -> tuple[pd.DataFrame, dict]:
    """match_length=True adds answer LENGTH to the foreign-null matching key.

    EXPLORATORY (not the frozen rule). The frozen null matches on (tier, type, digit count) only.
    But wrong-case answers turn out to be far rarer and longer than correct-case ones (median 0 vs 5
    other cases sharing the string; 16 vs 7 chars; Mann-Whitney p = 1.3e-14), so a null drawn from
    the same digit bucket contains systematically SHORTER, COMMONER strings for wrong cases -- which
    inflates the foreign rate for exactly the group whose own rate is low. That mechanism alone can
    manufacture the negative excess. This arm tests whether the correct-vs-wrong gap survives once
    the null is length-matched too.
    """
    caps = {}
    for l in open(caps_path):
        r = json.loads(l)
        caps[r["task_key"]] = r

    rng = np.random.default_rng(SEED)
    # foreign-answer pool, keyed by (tier, type, digit count) so the null is matched on the
    # surface features that make a spurious match likely in the first place
    pool: dict[tuple, list[str]] = defaultdict(list)
    for tk, c in caps.items():
        a = (c.get("model_answer") or "").strip()
        if a:
            pool[_key(c.get("tier") or "slice", a, match_length)].append(a)

    rows, excl = [], defaultdict(int)
    for l in open(dense_path):
        r = json.loads(l)
        if "error" in r:
            continue
        for side in ("wrong", "control"):
            s = r[side]
            c = caps[s["task_key"]]
            ans = (c.get("model_answer") or "").strip()
            if not ans:
                excl["no_parsed_answer"] += 1
                continue
            if c.get("truncated"):
                excl["truncated"] += 1
                continue
            cheap = len(ans) < MIN_ANSWER_CHARS or ans.lower() in CHEAP
            tier = c.get("tier") or "slice"
            key = _key(tier, ans, match_length)
            cands = [x for x in pool.get(key, []) if x != ans]
            if len(cands) < 5:                      # relax the digit match before the tier match
                cands = [x for k2, v in pool.items() if k2[0] == tier and k2[1] == key[1]
                         for x in v if x != ans]
            foreign = list(rng.choice(cands, size=min(N_FOREIGN, len(cands)), replace=False)) \
                if cands else []
            for rd in s["reads"]:
                u = rd.get("u_rel")
                if u is None or not (0.0 <= u <= 1.0):
                    continue
                txt = rd["read"]
                own = mentions_answer(txt, ans)
                fh = [mentions_answer(txt, f) for f in foreign]
                rows.append({"case": s["task_key"], "tier": tier,
                             "correct": int(side == "control"), "u": float(u),
                             "bin": min(int(u * N_BINS), N_BINS - 1),
                             "own": int(own),
                             "foreign_rate": float(np.mean(fh)) if fh else np.nan,
                             "n_foreign": len(foreign), "cheap": int(cheap),
                             "answer": ans})
    return pd.DataFrame(rows), dict(excl)


def onsets(d: pd.DataFrame) -> pd.DataFrame:
    """Per-case onset with right-censoring, per the frozen rule."""
    out = []
    for (case, tier, corr), g in d.groupby(["case", "tier", "correct"]):
        rate = g.groupby("bin")["own"].mean()
        onset, event = 1.0, 0
        for b in range(N_BINS - 1):
            if rate.get(b, 0.0) >= HIT_THRESH and rate.get(b + 1, 0.0) >= HIT_THRESH:
                onset, event = (b + 0.5) / N_BINS, 1
                break
        else:                                   # last bin alone cannot "stay" high; treat as censored
            if rate.get(N_BINS - 1, 0.0) >= HIT_THRESH:
                onset, event = (N_BINS - 0.5) / N_BINS, 1
        out.append({"case": case, "tier": tier, "correct": corr, "onset": onset, "event": event,
                    "n_reads": len(g), "cheap": int(g["cheap"].max())})
    return pd.DataFrame(out)


def main() -> int:
    d, excl = build(PROJ / "data/nla/n5/2026-08-06/dense_reads.jsonl",
                    PROJ / "data/nla/overnight/2026-08-04/captures.jsonl")
    print(f"reads: {len(d)} · cases: {d.case.nunique()} · exclusions: {excl}")

    res = {"n_reads": int(len(d)), "n_cases": int(d.case.nunique()), "exclusions": excl}

    for arm, dd in (("primary", d[d.cheap == 0]), ("with_cheap_answers", d)):
        o = onsets(dd)
        # --- excess-over-null trajectory, case-weighted ---
        per_case = (dd.groupby(["case", "correct", "bin"])[["own", "foreign_rate"]]
                    .mean().reset_index())
        traj = (per_case.groupby(["correct", "bin"])[["own", "foreign_rate"]].mean()
                .assign(excess=lambda x: x["own"] - x["foreign_rate"]).round(4))
        chi2, p, oe = logrank_stratified(o)
        med = {int(c): median_survival(g["onset"].to_numpy(), g["event"].to_numpy())
               for c, g in o.groupby("correct")}
        # cluster bootstrap over cases on the median difference
        rng = np.random.default_rng(SEED)
        oc, ow = o[o.correct == 1], o[o.correct == 0]
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            a = oc.iloc[rng.integers(0, len(oc), len(oc))]
            b = ow.iloc[rng.integers(0, len(ow), len(ow))]
            diffs[i] = (median_survival(b["onset"].to_numpy(), b["event"].to_numpy())
                        - median_survival(a["onset"].to_numpy(), a["event"].to_numpy()))
        lo, hi = (np.nanpercentile(diffs, [2.5, 97.5]) if np.isfinite(diffs).any()
                  else (np.nan, np.nan))

        # cluster bootstrap on the MEAN EXCESS over the foreign null, per group. This is the
        # pre-registered primary statistic and it survives censoring, unlike the median onset.
        pc = (dd.groupby(["case", "correct"])[["own", "foreign_rate"]].mean()
              .assign(excess=lambda x: x["own"] - x["foreign_rate"]).reset_index())
        exc = {}
        for cval, gg in pc.groupby("correct"):
            v = gg["excess"].to_numpy()
            bs = np.array([rng.choice(v, v.size, True).mean() for _ in range(N_BOOT)])
            exc[int(cval)] = {"mean": round(float(v.mean()), 4),
                              "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                       round(float(np.percentile(bs, 97.5)), 4)],
                              "n_cases": int(len(v))}
        vc, vw = pc[pc.correct == 1]["excess"].to_numpy(), pc[pc.correct == 0]["excess"].to_numpy()
        bd = np.array([rng.choice(vc, vc.size, True).mean() - rng.choice(vw, vw.size, True).mean()
                       for _ in range(N_BOOT)])
        exc["correct_minus_wrong"] = {"mean": round(float(vc.mean() - vw.mean()), 4),
                                      "ci95": [round(float(np.percentile(bd, 2.5)), 4),
                                               round(float(np.percentile(bd, 97.5)), 4)]}
        earlier = (med.get(0, np.nan) - med.get(1, np.nan))   # wrong minus correct
        ok = (p < 0.05) and (earlier >= 0.10)
        arm_res = {
            "n_cases": int(len(o)),
            "events": {int(c): int(g["event"].sum()) for c, g in o.groupby("correct")},
            "censored_frac": round(float(1 - o["event"].mean()), 3),
            "median_onset": {k: (None if np.isnan(v) else round(v, 3)) for k, v in med.items()},
            "correct_earlier_by": None if np.isnan(earlier) else round(float(earlier), 3),
            "boot_ci95": [None if np.isnan(lo) else round(float(lo), 3),
                          None if np.isnan(hi) else round(float(hi), 3)],
            "logrank_chi2": None if np.isnan(chi2) else round(chi2, 3), "logrank_p": p,
            "O_minus_E": None if np.isnan(oe) else round(oe, 2),
            "excess_by_bin": {f"{int(c)}": {int(b): float(v) for (cc, b), v
                                            in traj["excess"].items() if cc == c}
                              for c in (0, 1)},
            "own_rate_overall": round(float(dd["own"].mean()), 4),
            "foreign_rate_overall": round(float(dd["foreign_rate"].mean()), 4),
            "mean_excess_over_null": exc,
            "onset_reached_frac": {int(c): round(float(g["event"].mean()), 3)
                                   for c, g in o.groupby("correct")},
            # The frozen CONFIRM needs BOTH log-rank p<.05 AND median >= 0.10 earlier. Under heavy
            # censoring the median is undefined, so CONFIRM is unreachable however strong the
            # log-rank is -- say that explicitly instead of implying the log-rank failed.
            "verdict": ("CONFIRM" if ok else
                        "REFUTE (direction reversed)" if (not np.isnan(earlier) and earlier < 0)
                        else f"REFUTE (log-rank p={p:.4g} but median onset UNDEFINED — "
                             f"{round(float(1 - o['event'].mean()) * 100)}% of cases never reach "
                             f"onset, so the >=0.10-earlier criterion is unmeasurable)"
                        if p < 0.05 and np.isnan(earlier)
                        else "REFUTE (log-rank n.s. or gap < 0.10)"),
        }
        res[arm] = arm_res
        print(f"\n=== HT14 / {arm} ===")
        print(json.dumps(arm_res, indent=1, default=str))

    # ---- exploratory: does the gap survive a length-matched null? ----
    d2, _ = build(PROJ / "data/nla/n5/2026-08-06/dense_reads.jsonl",
                  PROJ / "data/nla/overnight/2026-08-04/captures.jsonl", match_length=True)
    d2 = d2[d2.cheap == 0]
    rng = np.random.default_rng(SEED)
    pc = (d2.groupby(["case", "correct"])[["own", "foreign_rate"]].mean()
          .assign(excess=lambda x: x["own"] - x["foreign_rate"]).reset_index())
    vc, vw = pc[pc.correct == 1]["excess"].to_numpy(), pc[pc.correct == 0]["excess"].to_numpy()
    bd = np.array([rng.choice(vc, vc.size, True).mean() - rng.choice(vw, vw.size, True).mean()
                   for _ in range(N_BOOT)])
    o2 = onsets(d2)
    chi2b, pb, _ = logrank_stratified(o2)
    res["exploratory_length_matched_null"] = {
        "note": "EXPLORATORY, not the frozen rule — foreign null additionally matched on answer "
                "length bucket, because wrong answers are rarer/longer than correct ones "
                "(p=1.3e-14) and the frozen null does not control for that.",
        "excess_correct": round(float(vc.mean()), 4),
        "excess_wrong": round(float(vw.mean()), 4),
        "correct_minus_wrong": round(float(vc.mean() - vw.mean()), 4),
        "boot_ci95": [round(float(np.percentile(bd, 2.5)), 4),
                      round(float(np.percentile(bd, 97.5)), 4)],
        "logrank_p": pb,
        "onset_reached_frac": {int(c): round(float(g["event"].mean()), 3)
                               for c, g in o2.groupby("correct")},
    }
    print("\n=== HT14 / exploratory: length-matched null ===")
    print(json.dumps(res["exploratory_length_matched_null"], indent=1, default=str))

    outp = PROJ / "data/nla/n8/2026-08-07/ht14.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
