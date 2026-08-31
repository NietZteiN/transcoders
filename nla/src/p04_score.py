"""P0.4 — the frozen decision rules, evaluated arithmetically.

Written BEFORE any depth arm produced a row, for the same reason `p0_verdict.py` was: a rule
applied by eye to a table you are already looking at is not a pre-registered rule. Every
constant below is quoted from log/nla-harness/2026-08-28_p04-depth-prereg.md and none of them
is a free parameter of this script.

THE STATISTIC. Balanced Delta-accuracy over ALL items, never flip rate; `recovered` and
`damaged` reported separately and never netted. Every cross-layer contrast is PAIRED on the
items both arms scored — exact McNemar on the discordant pairs plus a percentile bootstrap CI
on the per-item difference. Primary alpha = 1.0; the other four are secondary and carry a
BH-FDR adjustment across the alpha family.

WHY IT REFUSES. Six results in this programme have reversed between a partial run and full
sample size. A cell short of the full 60 items returns NEEDS_DATA and the verdict is withheld;
it does not quietly report a mean over whatever arrived.

Env `nla-mi` or `transcoders-mi` (scipy). CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent

from steer_stats import bh_adjust, boot_ci, mcnemar_exact  # noqa: E402  (frozen, shared)

# ── frozen constants — all four are quoted from the pre-registration ──────────
PRIMARY_ALPHA = 1.0
THRESH = 0.10          # the P0.2 threshold, reused for comparability. Arbitrary, hence frozen.
PARSE_MIN = 0.80       # rule (d). P0.2's negative was destruction, not a null: parse 0.100.
N_EXPECTED = 60        # the full stimulus set. Anything less is NEEDS_DATA.
REPRO_AGREE = 0.90     # cross-cluster gate: per-item agreement juno L20 vs banked csr L20
REPRO_DELTA = 0.05     # cross-cluster gate: |Delta-acc difference|

INSTRUMENT_LAYER = 20
TEST_LAYERS = (13, 6)  # 13 = P0.1 coherence argmax (primary); 6 = band control (secondary)


def load_arm(arm_dir: Path) -> tuple[dict[str, dict], dict[tuple[str, float], list[dict]]]:
    """(baseline by snippet_id, rows grouped by (condition, alpha)) for one layer arm."""
    base_p, res_p = arm_dir / "baseline.jsonl", arm_dir / "steer_results.jsonl"
    if not base_p.exists() or not res_p.exists():
        return {}, {}
    base = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(base_p) if l.strip()}
    by: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for l in open(res_p):
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" in r:
            continue
        by[(r["condition"], float(r.get("alpha", 1.0)))].append(r)
    return base, by


def cell(base: dict, rows: list[dict]) -> dict[str, Any]:
    """One (condition, alpha) cell: per-item flags plus the within-arm numbers."""
    flags, deltas, recovered, damaged = {}, [], 0, 0
    for r in rows:
        sid = r["snippet_id"]
        if sid not in base:
            continue
        was, now = bool(base[sid]["l1b_correct"]), bool(r["correct"])
        flags[sid] = int(now)
        deltas.append(int(now) - int(was))
        recovered += int(not was and now)
        damaged += int(was and not now)
    return {
        "n": len(deltas),
        "flags": flags,
        "acc": round(st.mean(flags.values()), 4) if flags else None,
        "delta_acc": round(st.mean(deltas), 4) if deltas else None,
        "delta_ci95": boot_ci(deltas),
        "recovered_wrong_to_right": recovered,
        "damaged_right_to_wrong": damaged,
        "parse_rate": round(st.mean(bool(r.get("parsed")) for r in rows), 4) if rows else None,
    }


def paired(a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    """a - b, paired on the items both scored. `a` is the arm under test."""
    shared = sorted(set(a) & set(b))
    a_only = sum(1 for s in shared if a[s] and not b[s])
    b_only = sum(1 for s in shared if b[s] and not a[s])
    diffs = [a[s] - b[s] for s in shared]
    return {
        "n_paired": len(shared),
        "delta": round(st.mean(diffs), 4) if diffs else None,
        "delta_ci95": boot_ci(diffs),
        "discordant_a_only": a_only, "discordant_b_only": b_only,
        "mcnemar_p": round(mcnemar_exact(a_only, b_only), 5),
    }


def ci_excludes_zero(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def apply_rule(cells: dict, layer: int, cond: str, alpha: float) -> dict[str, Any]:
    """The four-part rule, verbatim from the pre-registration. Order matters: (a) is primary,
    (d) is what separates a real gain from destruction, and a short cell decides nothing."""
    here = cells.get((layer, cond, alpha))
    ref = cells.get((INSTRUMENT_LAYER, cond, alpha))
    rand = cells.get((layer, "R_random", alpha))
    missing = [n for n, c in (("test", here), ("L20", ref), ("random", rand)) if not c]
    if missing:
        return {"verdict": "NEEDS_DATA", "reason": f"absent cells: {', '.join(missing)}"}
    short = {n: c["n"] for n, c in (("test", here), ("L20", ref), ("random", rand))
             if c["n"] < N_EXPECTED}
    if short:
        return {"verdict": "NEEDS_DATA",
                "reason": f"cells short of n={N_EXPECTED}: {short}"}

    vs_l20 = paired(here["flags"], ref["flags"])
    vs_rand = paired(here["flags"], rand["flags"])
    a = vs_l20["delta"] >= THRESH and ci_excludes_zero(vs_l20["delta_ci95"])
    b = here["delta_acc"] >= THRESH and ci_excludes_zero(here["delta_ci95"])
    c = vs_rand["delta"] >= THRESH
    d = here["parse_rate"] >= PARSE_MIN

    if a and b and c and d:
        verdict = "DEPTH-LIMITED"
    elif a and b and c and not d:
        verdict = "DESTRUCTIVE"
    else:
        verdict = "NOT DEPTH-LIMITED"
    return {
        "verdict": verdict,
        "criteria": {
            "a_beats_L20": {"passed": a, "delta": vs_l20["delta"],
                            "ci95": vs_l20["delta_ci95"], "threshold": THRESH,
                            "mcnemar_p": vs_l20["mcnemar_p"]},
            "b_beats_baseline": {"passed": b, "delta_acc": here["delta_acc"],
                                 "ci95": here["delta_ci95"], "threshold": THRESH},
            "c_beats_random": {"passed": c, "delta": vs_rand["delta"],
                               "ci95": vs_rand["delta_ci95"], "threshold": THRESH},
            "d_parse_intact": {"passed": d, "parse_rate": here["parse_rate"],
                               "minimum": PARSE_MIN},
        },
        "recovered_wrong_to_right": here["recovered_wrong_to_right"],
        "damaged_right_to_wrong": here["damaged_right_to_wrong"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p04"))
    ap.add_argument("--banked", default=str(PROJ / "data/nla/n12"),
                    help="the csr-94608 B4 run, used ONLY for the cross-cluster reproduction "
                         "check. It is never the comparison arm — see the prereg.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_p = Path(args.out) if args.out else root / "p04_verdict.json"

    arms = {L: load_arm(root / f"L{L:02d}") for L in (6, 13, INSTRUMENT_LAYER)}
    cells: dict[tuple[int, str, float], dict] = {}
    alphas: set[float] = set()
    for L, (base, by) in arms.items():
        for (cond, a), rows in by.items():
            cells[(L, cond, a)] = cell(base, rows)
            alphas.add(a)

    rep: dict[str, Any] = {
        "experiment": "p0.4_depth",
        "prereg": "log/nla-harness/2026-08-28_p04-depth-prereg.md",
        "primary_alpha": PRIMARY_ALPHA, "threshold": THRESH, "parse_minimum": PARSE_MIN,
        "n_expected": N_EXPECTED,
        "arms_present": {f"L{L:02d}": {"baseline_items": len(b), "result_cells": len(by)}
                         for L, (b, by) in arms.items()},
        "alphas": sorted(alphas),
    }

    # The three arms' baselines are unsteered generations and are layer-independent, so they
    # must agree item for item. A disagreement means the arms are not comparable and nothing
    # below is interpretable — it is checked rather than assumed.
    bases = {L: b for L, (b, _) in arms.items() if b}
    if len(bases) > 1:
        ref_L = sorted(bases)[0]
        shared = set.intersection(*(set(b) for b in bases.values()))
        rep["baseline_consistency"] = {
            "shared_items": len(shared),
            "agreement": {f"L{L:02d}": round(
                st.mean(bases[L][s]["l1b_correct"] == bases[ref_L][s]["l1b_correct"]
                        for s in shared), 4) for L in sorted(bases)} if shared else {},
            "reference": f"L{ref_L:02d}",
        }

    # ── gate 2: does the juno L20 arm reproduce the banked csr-94608 L20 arm? ──
    bbase, bby = load_arm(Path(args.banked))
    repro: dict[str, Any] = {"checked": bool(bbase and bby)}
    for cond in ("V3_taskvec", "V4_oracle"):
        here = cells.get((INSTRUMENT_LAYER, cond, PRIMARY_ALPHA))
        there = cell(bbase, bby.get((cond, PRIMARY_ALPHA), [])) if bby else None
        if not here or not there or not there["n"]:
            repro[cond] = {"status": "NEEDS_DATA"}
            continue
        shared = sorted(set(here["flags"]) & set(there["flags"]))
        agree = round(st.mean(here["flags"][s] == there["flags"][s] for s in shared), 4) \
            if shared else None
        dgap = (round(abs(here["delta_acc"] - there["delta_acc"]), 4)
                if here["delta_acc"] is not None and there["delta_acc"] is not None else None)
        repro[cond] = {
            "n_shared": len(shared), "per_item_agreement": agree,
            "juno_delta_acc": here["delta_acc"], "banked_delta_acc": there["delta_acc"],
            "delta_acc_gap": dgap,
            "passed": bool(agree is not None and agree >= REPRO_AGREE
                           and dgap is not None and dgap <= REPRO_DELTA),
        }
    checked = [v for v in (repro.get("V3_taskvec"), repro.get("V4_oracle"))
               if isinstance(v, dict) and "passed" in v]
    repro["verdict"] = ("PASS" if checked and all(v["passed"] for v in checked)
                        else "NEEDS_DATA" if not checked else "FAIL")
    repro["note"] = ("A FAIL does not invalidate P0.4 — both contrasted arms ran on juno — but "
                     "the banked B4 numbers may not then be quoted in the same table.")
    rep["cross_cluster_reproduction"] = repro

    # ── the verdicts ──────────────────────────────────────────────────────────
    rep["primary"] = {
        "hypothesis": "H-P04a — is depth the constraint? V4_oracle, layer 13 vs 20, alpha=1.0",
        **apply_rule(cells, 13, "V4_oracle", PRIMARY_ALPHA),
    }

    # H-P04b is evaluated only if the primary fired; stating it otherwise invites reading a
    # band/point distinction into two arms that both did nothing.
    if rep["primary"]["verdict"] == "DEPTH-LIMITED":
        l6 = apply_rule(cells, 6, "V4_oracle", PRIMARY_ALPHA)
        rep["secondary_band_or_point"] = {
            "hypothesis": "H-P04b — does layer 6 do it too?",
            "layer6": l6,
            "verdict": ("BAND" if l6["verdict"] == "DEPTH-LIMITED"
                        else "NEEDS_DATA" if l6["verdict"] == "NEEDS_DATA" else "POINT"),
            "note": ("POINT is an unexplained layer-specific effect awaiting replication. It is "
                     "NOT evidence that coherence predicts steerability: P0.1's coherence gap "
                     "between L6 and L13 is 0.012 against a per-item SD of ~0.11."),
        }
    else:
        rep["secondary_band_or_point"] = {
            "verdict": "NOT EVALUATED",
            "reason": "H-P04b is defined only when the primary returns DEPTH-LIMITED.",
        }

    rep["secondary_V3"] = {
        "hypothesis": "H-P04c — the no-NLA bar at every layer",
        **{f"L{L:02d}": apply_rule(cells, L, "V3_taskvec", PRIMARY_ALPHA) for L in TEST_LAYERS},
    }
    # V3 strictly less informed than V4: V3 clearing while V4 fails is a bug, not a finding.
    v3_13 = rep["secondary_V3"]["L13"]["verdict"]
    if v3_13 == "DEPTH-LIMITED" and rep["primary"]["verdict"] == "NOT DEPTH-LIMITED":
        rep["secondary_V3"]["incoherence_flag"] = (
            "V3 cleared the rule at layer 13 where V4 did not. The oracle strictly dominates "
            "the leave-one-out task vector in information, so this is to be diagnosed as a bug "
            "before it is reported as a result.")

    # ── secondary alphas, BH-adjusted across the family ───────────────────────
    sec = {}
    for a in sorted(alphas - {PRIMARY_ALPHA}):
        sec[f"alpha={a:g}"] = apply_rule(cells, 13, "V4_oracle", a)
    ps = [v["criteria"]["a_beats_L20"]["mcnemar_p"] for v in sec.values() if "criteria" in v]
    if ps:
        for (k, v), q in zip([(k, v) for k, v in sec.items() if "criteria" in v], bh_adjust(ps)):
            v["mcnemar_q_bh"] = q
    rep["secondary_alphas"] = sec
    rep["dose_response"] = {
        f"L{L:02d}/{cond}": [{"alpha": a, "delta_acc": cells[(L, cond, a)]["delta_acc"],
                              "ci95": cells[(L, cond, a)]["delta_ci95"],
                              "parse_rate": cells[(L, cond, a)]["parse_rate"]}
                             for a in sorted(alphas) if (L, cond, a) in cells]
        for L in (6, 13, INSTRUMENT_LAYER)
        for cond in ("V3_taskvec", "V4_oracle", "R_random")
        if any((L, cond, a) in cells for a in alphas)
    }

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "dose_response"}, indent=2))
    print(f"\n[p04] PRIMARY: {rep['primary']['verdict']}")
    print(f"[p04] band/point: {rep['secondary_band_or_point']['verdict']}")
    print(f"[p04] reproduction gate: {repro['verdict']}")
    print(f"[p04] wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
