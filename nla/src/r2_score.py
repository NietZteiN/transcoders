#!/usr/bin/env python
"""R2 — damage-controlled clean-half readout. CPU only; reads banked rows.

Rule frozen in log/nla-harness/2026-09-03_clean-half-prereg.md BEFORE this script ran.
Nothing here may be tuned after seeing its output; a changed rule goes in a new dated entry.

Why this exists: Experiment R's metric M was a likelihood *ratio* whose denominator is the model's
own greedy corrupt reply, so any intervention that degrades the model raises M (the clean half moved
the wrong way in 21 of 28 arms). R2 drops the corrupt term entirely and vetoes damage instead:

    M2_i(a) = [log P(y_clean_i | x_l1b_i, theta_a) - log P(y_clean_i | x_l1b_i)] / |y_clean_i|
    G_i     = [log P(y_clean_i | x_l0_i)          - log P(y_clean_i | x_l1b_i)] / |y_clean_i|

G is the unit: what swapping in the un-obfuscated prompt achieves. An arm must win a fixed fraction
of G, separate from its matched-alpha random control by NON-overlapping CI (R's floor-based test was
too permissive), and not damage banked greedy behaviour.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

BELIEF = {"V1_gloss", "V2_wordedit", "V3_taskvec", "V4_oracle", "V5_replace"}  # same set as trace_llr.py
FLOOR_FALLBACK = 0.01
SUPPORT_FRAC = 0.10      # mover threshold = max(SUPPORT_FRAC * mean G, F2)
VETO_DROP = 0.05         # V-behav: max tolerated absolute drop in parse rate / accuracy
SEED = 20260724
N_BOOT = 10000
_PROJ = pathlib.Path(__file__).resolve().parents[2]


def per_token(row):
    return row["logp_sum"] / row["n_tok"]


def boot_ci(x, rng, n=N_BOOT):
    """Percentile bootstrap over items. Returns (lo, hi) of the mean."""
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_clean(rows_p):
    """key -> per-token logp of the CLEAN trace only. The corrupt half is never read."""
    out = {}
    with open(rows_p) as f:
        for line in f:
            r = json.loads(line)
            sid, *tag, which = r["key"].split("|")
            if which != "clean":
                continue
            out[(sid, "|".join(tag))] = per_token(r)
    return out


def load_behav(paths):
    """Banked greedy rows -> arm -> (accuracy, parse rate, n). Damage veto, no GPU."""
    agg = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                k = (f"{r['condition']}|{r['alpha']}|{r.get('positions')}|"
                     f"{'ML' if r.get('multilayer') else 'SL'}")
                a = agg.setdefault(k, {"correct": 0, "parsed": 0, "n": 0})
                a["correct"] += bool(r.get("correct")); a["parsed"] += bool(r.get("parsed")); a["n"] += 1
    return {k: {"acc": v["correct"] / v["n"], "parse": v["parsed"] / v["n"], "n": v["n"]}
            for k, v in agg.items() if v["n"]}


def baseline_behav(baseline_p):
    n = corr = parsed = 0
    with open(baseline_p) as f:
        for line in f:
            r = json.loads(line)
            n += 1; corr += bool(r.get("l1b_correct")); parsed += bool(r.get("l1b_parsed"))
    return {"acc": corr / n, "parse": parsed / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/llr_rows.jsonl"))
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/r2_stats.json"))
    ap.add_argument("--spec-rows", default=None, help="V-spec pass output; omit to leave V-spec pending")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    p0 = _PROJ / "data/nla/p0"
    clean = load_clean(pathlib.Path(a.rows))
    sids = sorted({s for s, _ in clean})
    arms = sorted({t for _, t in clean} - {"noop#1", "noop#2", "l0prompt"})

    def vec(tag):
        """Paired per-item series vs noop#1; None if any item is missing (no ragged denominators)."""
        v = []
        for s in sids:
            x, b = clean.get((s, tag)), clean.get((s, "noop#1"))
            if x is None or b is None:
                return None
            v.append(x - b)
        return np.array(v)

    # --- floor and unit, both from unsteered rows, computed before any arm is scored ---
    noop_d = np.array([clean[(s, "noop#1")] - clean[(s, "noop#2")] for s in sids
                       if (s, "noop#2") in clean])
    F2 = float(2 * noop_d.std(ddof=1)) if len(noop_d) > 2 else FLOOR_FALLBACK
    if not np.isfinite(F2) or F2 <= 0:
        F2 = FLOOR_FALLBACK
    G = vec("l0prompt")
    mean_G = float(G.mean())
    sanity = {"n": len(sids), "mean_G": mean_G, "frac_G_positive": float((G > 0).mean()),
              "G_ci95": boot_ci(G, rng), "F2": F2, "n_floor_items": len(noop_d)}
    s2_ok = sanity["frac_G_positive"] >= 0.90 and mean_G > 10 * F2
    threshold = max(SUPPORT_FRAC * mean_G, F2)

    behav = load_behav([p0 / d / "gemma12b/steer_results.jsonl"
                        for d in ("nla_steer", "coverage", "steerv2/run")])
    base_b = baseline_behav(p0 / "steerv2/gemma12b/run/baseline.jsonl")
    spec = json.load(open(a.spec_rows)) if a.spec_rows else {}

    per_arm = {}
    for k in arms:
        v = vec(k)
        if v is None:
            continue
        cond, alpha, pos, ml = k.split("|")
        lo, hi = boot_ci(v, rng)
        b = behav.get(k)
        # V-behav: an arm with no banked greedy rows is 'unavailable' and can never be supported.
        veto_behav = (None if b is None else
                      (b["acc"] >= base_b["acc"] - VETO_DROP and b["parse"] >= base_b["parse"] - VETO_DROP))
        sp = spec.get(k)
        veto_spec = None if sp is None else (sp >= -SUPPORT_FRAC * mean_G)
        per_arm[k] = {"cond": cond, "alpha": float(alpha), "positions": pos, "ml": ml == "ML",
                      "n": len(v), "mean_M2": float(v.mean()), "ci95": [lo, hi],
                      "sd": float(v.std(ddof=1)), "frac_G": float(v.mean() / mean_G) if mean_G else None,
                      "mover": bool(v.mean() >= threshold and lo > 0),
                      "behav": b, "veto_behav": veto_behav, "spec": sp, "veto_spec": veto_spec}

    # --- separation from the matched-alpha random control: CIs must not overlap ---
    rand = {(v["alpha"], v["positions"], v["ml"]): v for k, v in per_arm.items() if v["cond"] == "R_random"}
    for k, v in per_arm.items():
        r = rand.get((v["alpha"], v["positions"], v["ml"]))
        v["matched_random_mean"] = r["mean_M2"] if r else None
        v["separates_from_random"] = (None if r is None else bool(v["ci95"][0] > r["ci95"][1]))

    supported = [k for k, v in per_arm.items()
                 if v["cond"] in BELIEF and v["mover"] and v["separates_from_random"]
                 and v["veto_behav"] and v["veto_spec"]]
    # Arms that clear everything except the GPU pass -> that pass is worth running; otherwise it isn't.
    pending = [k for k, v in per_arm.items()
               if v["cond"] in BELIEF and v["mover"] and v["separates_from_random"]
               and v["veto_behav"] and v["veto_spec"] is None]
    movers = [k for k, v in per_arm.items() if v["mover"]]

    if not s2_ok:
        verdict = "R2-VOID"
    elif supported:
        verdict = "R2-BELIEF"
    elif pending:
        verdict = "R2-PENDING-SPEC"      # not a result: the staged GPU veto decides
    elif movers:
        verdict = "R2-GENERIC"
    else:
        verdict = "R2-NULL"

    res = {"experiment": "R2_clean_half", "seed": SEED, "n_boot": N_BOOT, "rows": a.rows,
           "sanity": sanity, "sanity_ok": bool(s2_ok), "support_frac": SUPPORT_FRAC,
           "threshold": threshold, "baseline_behav": base_b, "per_arm": per_arm,
           "movers": movers, "supported": supported, "pending_spec": pending, "verdict": verdict}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"F2 {F2:.4f} · mean G {mean_G:+.4f} [{sanity['G_ci95'][0]:+.4f},{sanity['G_ci95'][1]:+.4f}] "
          f"· G>0 on {sanity['frac_G_positive']:.0%} · sanity {'OK' if s2_ok else 'FAIL'} "
          f"· mover threshold {threshold:+.4f}")
    print(f"{'arm':<38}{'meanM2':>9}{'ci95':>22}{'%G':>7}{'mov':>5}{'sep':>5}{'beh':>5}{'acc':>6}")
    for k, v in sorted(per_arm.items(), key=lambda kv: -kv[1]["mean_M2"]):
        f = lambda x: {True: "y", False: "n", None: "-"}[x]
        print(f"{k:<38}{v['mean_M2']:>+9.4f}"
              f"{f'[{v[chr(99)+chr(105)+chr(57)+chr(53)][0]:+.4f},{v[chr(99)+chr(105)+chr(57)+chr(53)][1]:+.4f}]':>22}"
              f"{(v['frac_G'] or 0)*100:>6.0f}%{f(v['mover']):>5}{f(v['separates_from_random']):>5}"
              f"{f(v['veto_behav']):>5}{(v['behav']['acc'] if v['behav'] else float('nan')):>6.2f}")
    print(f"\nbaseline acc {base_b['acc']:.3f} parse {base_b['parse']:.3f} (n={base_b['n']})")
    print(f"VERDICT {verdict} · movers {movers or 'none'} · supported {supported or 'none'} "
          f"· pending V-spec {pending or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
