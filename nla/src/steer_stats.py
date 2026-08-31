"""Score the B4 steering gate with the pre-declared statistic.

**Balanced Δaccuracy, not flip rate.** A direction that pushes the model toward the true
semantics on every item scores brilliantly on "wrong → right" flips while quietly destroying
answers it already had. The statistic is therefore accuracy change across ALL items, and
`recovered` / `damaged` are reported separately so the trade is visible rather than netted away.

**The gate is V1 > V3, not V1 > 0.** V3 is a contrastive task vector — `mean(h_L0 − h_L1b)` over
other programs — needing no verbalizer, reconstructor or explanation. Beating "no steering"
proves only that a large perturbation changes outputs, which the random control already shows.

Paired tests throughout: every condition is evaluated on the same programs, so the informative
comparison is per-item discordance (exact McNemar), not a difference of marginal rates.

**Everything is keyed by (condition, alpha).** The first version grouped by condition alone,
which was correct for the banked single-alpha run and silently wrong for a sweep: `flags[sid]`
was assigned once per row, so an item's accuracy became whichever alpha appeared last in the
file, while `deltas` pooled every alpha into one mean. That produces a plausible-looking number
with no defined meaning. A sweep therefore needs this module, not just `--alphas` on the runner.

**The alpha family is a multiplicity problem.** Testing the gate at every alpha and reporting
the best one is the garden of forking paths, and this programme has twice been fooled by
partial results (N11's 3/3 → 3/10; B4's n=34 interim → an exact tie at n=60). So: the gate at
the pre-registered `--primary-alpha` is the verdict, every other alpha is secondary, and the
secondary p-values carry a BH-FDR adjustment across the alpha family. N7 recorded the FDR
family as vacuous because no p-value ever entered it; this is the first real one.

Env `transcoders-mi` (scipy). CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parent.parent.parent
SEED = 20260724


def mcnemar_exact(b: int, c: int) -> float:
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def boot_ci(vals: list[float], n_boot: int = 5000) -> tuple[float, float]:
    """Percentile CI on a mean of per-item deltas (paired, so items are the unit)."""
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(SEED)
    n = len(vals)
    means = sorted(st.mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up. Returns adjusted p-values in the input order."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        prev = min(prev, pvals[i] * n / k)
        adj[i] = round(min(prev, 1.0), 5)
    return adj


def gate(v1: dict[str, int], v3: dict[str, int], note: str) -> dict[str, Any]:
    """V1 vs V3 paired on the programs both conditions scored."""
    shared = sorted(set(v1) & set(v3))
    b_only = sum(1 for s in shared if v1[s] and not v3[s])
    c_only = sum(1 for s in shared if v3[s] and not v1[s])
    return {
        "n_paired": len(shared),
        "V1_acc": round(st.mean(v1[s] for s in shared), 4) if shared else None,
        "V3_acc": round(st.mean(v3[s] for s in shared), 4) if shared else None,
        "delta": round(st.mean(v1[s] - v3[s] for s in shared), 4) if shared else None,
        "delta_ci95": boot_ci([v1[s] - v3[s] for s in shared]),
        "discordant_V1_only": b_only, "discordant_V3_only": c_only,
        "mcnemar_p": round(mcnemar_exact(b_only, c_only), 5),
        "verdict": ("V1 > V3" if b_only > c_only else
                    "V3 >= V1" if c_only > b_only else "tie"),
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(PROJ / "data/nla/n12/steer_results.jsonl"))
    ap.add_argument("--baseline", default=str(PROJ / "data/nla/n12/baseline.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n12/steer_stats.json"))
    ap.add_argument("--primary-alpha", type=float, default=1.0,
                    help="The pre-registered alpha whose gate is THE verdict. Every other "
                         "alpha is secondary and BH-adjusted. Declare it before the run.")
    args = ap.parse_args()

    base = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(args.baseline) if l.strip()}
    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    ok = [r for r in rows if "error" not in r]

    # (condition, alpha) is the grouping key. Rows from before the sweep carry no `alpha`;
    # they are the banked alpha=1.0 run, so default accordingly rather than dropping them.
    by: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in ok:
        by[(r["condition"], float(r.get("alpha", 1.0)))].append(r)
    alphas = sorted({a for _, a in by})

    rep: dict[str, Any] = {
        "experiment": "n12_b4_steering_gate", "seed": SEED,
        "n_rows": len(rows), "n_errors": len(rows) - len(ok),
        "n_baseline_items": len(base),
        "baseline_l0_acc": round(st.mean(b["l0_correct"] for b in base.values()), 4),
        "baseline_l1b_acc": round(st.mean(b["l1b_correct"] for b in base.values()), 4),
        "alphas": alphas,
        "primary_alpha": args.primary_alpha,
        "conditions": {},
    }

    per_item: dict[tuple[str, float], dict[str, int]] = {}
    for (cond, alpha), rs in sorted(by.items()):
        deltas, recovered, damaged, b_only, c_only = [], 0, 0, 0, 0
        flags: dict[str, int] = {}
        for r in rs:
            sid = r["snippet_id"]
            if sid not in base:
                continue
            was = bool(base[sid]["l1b_correct"])
            now = bool(r["correct"])
            flags[sid] = int(now)
            deltas.append(int(now) - int(was))
            if not was and now:
                recovered += 1
                b_only += 1
            if was and not now:
                damaged += 1
                c_only += 1
        per_item[(cond, alpha)] = flags
        n = len(deltas)
        rep["conditions"].setdefault(cond, {})[f"alpha={alpha:g}"] = {
            "n": n,
            "acc": round(st.mean(flags.values()), 4) if flags else None,
            "delta_acc": round(st.mean(deltas), 4) if deltas else None,
            "delta_ci95": boot_ci(deltas),
            "recovered_wrong_to_right": recovered,
            "damaged_right_to_wrong": damaged,
            "mcnemar_vs_baseline_p": round(mcnemar_exact(b_only, c_only), 5),
            "parse_rate": round(st.mean(bool(r.get("parsed")) for r in rs), 4),
        }

    # ── the gate: V1 vs V3, paired on the same programs, AT EACH ALPHA ────
    by_alpha: dict[str, Any] = {}
    for alpha in alphas:
        v1 = per_item.get(("V1_gloss", alpha), {})
        v3 = per_item.get(("V3_taskvec", alpha), {})
        if v1 and v3:
            by_alpha[f"alpha={alpha:g}"] = gate(
                v1, v3, "secondary — BH-adjusted across the alpha family")

    # BH across the secondary alphas. The primary is excluded from its own correction: it is
    # one pre-registered test, not a discovery among many.
    secondary = [k for k in by_alpha if k != f"alpha={args.primary_alpha:g}"]
    if secondary:
        adj = bh_adjust([by_alpha[k]["mcnemar_p"] for k in secondary])
        for k, q in zip(secondary, adj):
            by_alpha[k]["mcnemar_q_bh"] = q
    rep["gate_V1_vs_V3_by_alpha"] = by_alpha

    # The verdict is the gate at the pre-registered alpha, and nowhere else.
    pkey = f"alpha={args.primary_alpha:g}"
    if pkey in by_alpha:
        rep["gate_V1_vs_V3"] = dict(by_alpha[pkey])
        rep["gate_V1_vs_V3"]["note"] = (
            f"PRIMARY (pre-registered alpha={args.primary_alpha:g}). V1 must beat V3 for the "
            f"NLA to be doing causal work. Other alphas are secondary and BH-adjusted; a "
            f"secondary alpha clearing the gate is a lead, not a verdict.")
    else:
        rep["gate_V1_vs_V3"] = {
            "error": f"no V1/V3 rows at the primary alpha {args.primary_alpha:g}",
            "alphas_present": alphas,
        }

    # ── dose-response: is there ANY monotone structure in alpha? ──────────
    # The damning number in the banked run was that the antipodal control matched random,
    # i.e. the direction has no consistent sign. A dose-response curve is where that would
    # show up if it were wrong: a real direction should scale, a sign-free one should not.
    rep["dose_response"] = {
        cond: [
            {"alpha": a,
             "delta_acc": rep["conditions"][cond].get(f"alpha={a:g}", {}).get("delta_acc"),
             "ci95": rep["conditions"][cond].get(f"alpha={a:g}", {}).get("delta_ci95")}
            for a in alphas if f"alpha={a:g}" in rep["conditions"][cond]
        ]
        for cond in sorted(rep["conditions"])
    }

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
