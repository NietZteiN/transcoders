"""N6 / HT12 — does reconstruction faithfulness predict correctness, properly?

Confirmatory. The decision rule was frozen in
log/nla-harness/2026-08-06_n6-n8-preregistration.md BEFORE this ran. Nothing here may be
changed to chase a result; deviations must be logged as deviations.

Primary model (statsmodels MixedLM, groups = task_key):
    rt_cos ~ correct + tier + read_kind + log(read_len) + log(reply_tokens) + u_rel + u_rel^2
             + (1 | case)

Co-primaries, ALL THREE required to confirm:
  1. mixed-model Wald p < 0.05 on beta_correct (BH-FDR within {HT12, HT13, HT14})
  2. case-level permutation test p < 0.05 (labels shuffled at CASE level, within tier)
  3. case-level cluster-bootstrap 95% CI excludes 0
REFUTE if the mixed-model CI includes 0, OR |beta_correct| < 0.005.

Runs on BOTH corpora, reported separately and never pooled (different sampling designs):
  * banked   — the 2026-08-04 corpus (5,090 reads). The post-hoc finding lives here; N6
               sharpens it, it cannot independently confirm it.
  * dense    — the N5 matched corpus (4,653 reads), which is the length-matched replication.

Env: transcoders-mi (statsmodels + patsy). No GPU.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
SEED = 20260724
N_PERM = 10000
N_BOOT = 4000
REFUTE_BETA = 0.005


def load_banked() -> pd.DataFrame:
    cases = json.load(open(PROJ / "data/nla/overnight/2026-08-04/enriched.json"))
    rows = []
    for c in cases:
        if c.get("correct") is None:
            continue
        L = max(len(c["model_reply"]), 1)
        for rd in c["reads"]:
            a = rd.get("anchor") or {}
            u = (a.get("rs", 0) / L) if a.get("in_reply") else np.nan
            w = rd["where"]
            kind = ("cot" if w.startswith("cot@") else "answer" if w.startswith("ans@")
                    else "dispatcher" if w.startswith("disp:") else "code_id")
            rows.append({"case": c["task_key"], "snippet": c["snippet_id"], "tier": c["tier"] or "slice",
                         "correct": int(bool(c["correct"])), "rt_cos": rd["rt_cos"],
                         "read_len": len(rd["read"]), "reply_tokens": max(c["reply_tokens"] or 1, 1),
                         "read_kind": kind, "u_rel": u})
    return pd.DataFrame(rows)


def load_dense() -> pd.DataFrame:
    rows = []
    for line in open(PROJ / "data/nla/n5/2026-08-06/dense_reads.jsonl"):
        r = json.loads(line)
        if "error" in r:
            continue
        for side in ("wrong", "control"):
            s = r[side]
            for rd in s["reads"]:
                rows.append({"case": s["task_key"], "snippet": s["task_key"], "tier": s["tier"] or "slice",
                             "correct": int(side == "control"), "rt_cos": rd["rt_cos"],
                             "read_len": len(rd["read"]), "reply_tokens": max(s["reply_tokens"] or 1, 1),
                             "read_kind": rd["kind"], "u_rel": rd.get("u_rel", np.nan),
                             "pair": r["task_key"]})
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame, label: str) -> dict:
    import statsmodels.formula.api as smf

    d = df.dropna(subset=["u_rel"]).copy()
    d["log_read_len"] = np.log(d["read_len"].clip(lower=1))
    d["log_reply"] = np.log(d["reply_tokens"])
    d["u2"] = d["u_rel"] ** 2
    formula = ("rt_cos ~ correct + C(tier) + C(read_kind) + log_read_len + log_reply + u_rel + u2")
    # PRE-REGISTRATION DEVIATION (logged): the frozen rule specified MixedLM with (1|case).
    # That model is structurally unusable here — `correct` is CONSTANT within every case
    # (verified: 0/330 banked and 0/182 dense groups vary), so a per-case random intercept
    # absorbs exactly the between-case variance the predictor lives in. It returned beta=0
    # with NaN se on the dense corpus and a singular fit on the banked one. GEE with an
    # exchangeable working correlation clustered on case is the correct tool for a
    # group-constant predictor: it keeps read-level data, honours the clustering, and
    # estimates a between-group effect. The case-level co-primary is unchanged.
    import statsmodels.api as sm
    try:
        m = smf.gee(formula, groups=d["case"], data=d,
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        beta = float(m.params.get("correct", np.nan))
        se = float(m.bse.get("correct", np.nan))
        p = float(m.pvalues.get("correct", np.nan))
        ci = (beta - 1.96 * se, beta + 1.96 * se)
        conv = True
    except Exception as e:                     # keep the failure visible rather than silent
        return {"corpus": label, "error": repr(e)[:300]}

    # --- case-level aggregation: the version immune to pseudo-replication ---
    cl = (d.groupby(["case", "tier", "correct"], as_index=False)
            .agg(mean_rt=("rt_cos", "mean"), mean_len=("read_len", "mean"),
                 reply_tokens=("reply_tokens", "first"), n=("rt_cos", "size")))
    import statsmodels.formula.api as smf2
    cl["log_len"] = np.log(cl["mean_len"].clip(lower=1))
    cl["log_reply"] = np.log(cl["reply_tokens"])
    om = smf2.ols("mean_rt ~ correct + C(tier) + log_len + log_reply", cl).fit()
    beta_case = float(om.params.get("correct", np.nan))
    p_case = float(om.pvalues.get("correct", np.nan))

    # --- (2) case-level permutation, shuffling WITHIN tier ---
    rng = random.Random(SEED)
    obs = abs(beta_case)
    hits = 0
    by_tier = {t: cl.index[cl.tier == t].tolist() for t in cl.tier.unique()}
    lab = cl["correct"].to_numpy().copy()
    for _ in range(N_PERM):
        perm = lab.copy()
        for t, idx in by_tier.items():
            vals = [lab[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                perm[i] = v
        tmp = cl.assign(correct=perm)
        b = np.polyfit(tmp["correct"], tmp["mean_rt"], 1)[0]     # simple slope; tier-balanced by design
        hits += abs(b) >= obs
    p_perm = (hits + 1) / (N_PERM + 1)

    # --- (3) case-level cluster bootstrap on the raw difference ---
    rngb = np.random.default_rng(SEED)
    cases_c = cl[cl.correct == 1]["mean_rt"].to_numpy()
    cases_w = cl[cl.correct == 0]["mean_rt"].to_numpy()
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs[i] = (rngb.choice(cases_c, cases_c.size, True).mean()
                    - rngb.choice(cases_w, cases_w.size, True).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    return {"corpus": label, "n_reads": int(len(d)), "n_cases": int(cl.shape[0]),
            "model": "GEE(exchangeable, cluster=case)", "beta_correct": round(beta, 5), "se": round(se, 5), "p_wald": p,
            "ci95": [round(ci[0], 5), round(ci[1], 5)], "converged": conv,
            "case_level": {"beta": round(beta_case, 5), "p": p_case,
                           "raw_diff": round(float(cases_c.mean() - cases_w.mean()), 5),
                           "boot_ci95": [round(float(lo), 5), round(float(hi), 5)]},
            "p_permutation": round(p_perm, 5)}


def verdict(r: dict) -> str:
    if "error" in r:
        return "ERROR"
    ci = r["ci95"]
    if ci[0] <= 0 <= ci[1]:
        return "REFUTE (mixed-model CI includes 0)"
    if abs(r["beta_correct"]) < REFUTE_BETA:
        return f"REFUTE (|beta| {abs(r['beta_correct']):.5f} < {REFUTE_BETA} — too small to be usable)"
    ok = (r["p_wald"] < 0.05 and r["p_permutation"] < 0.05
          and not (r["case_level"]["boot_ci95"][0] <= 0 <= r["case_level"]["boot_ci95"][1]))
    return "CONFIRM" if ok else "EQUIVOCAL (not all three co-primaries hold)"


def main() -> int:
    out = {}
    for label, df in (("banked", load_banked()), ("dense", load_dense())):
        r = fit(df, label)
        r["verdict"] = verdict(r)
        out[label] = r
        print(f"\n=== HT12 / {label} ===")
        print(json.dumps(r, indent=1, default=str))
    outp = PROJ / "data/nla/n6/2026-08-07/ht12.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
