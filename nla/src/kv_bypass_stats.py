"""Evaluate the KV-bypass pre-registration arithmetically. No verdict is typed by hand.

The rules are frozen in log/nla-harness/2026-09-03_kv-bypass-prereg.md §6 and this script is
their only executor, so the verdict cannot drift from the text between the run and the write-up:

  H-C1  SUPPORTED  iff  multi-V3(matched) - single-V3  >= +0.10, bootstrap CI excludes zero,
                        AND multi-random(matched) does not match the gain (its own delta is
                        below the multi-V3 delta by more than the CI half-width).
  H-C0  SUPPORTED  iff  |multi-V3(matched) - single-V3| is inside the reproducibility floor.
  H-C2  FLAGGED    iff  the alpha=1.0 "loud" arm moves accuracy while the matched arm does not,
                        or its parse rate collapses toward P0.2's 0.100.
  UNINFORMATIVE    iff  both arms sit inside the unsteered baseline's own spread. This is checked
                        FIRST and short-circuits, because a flat comparison is not evidence for
                        H-C0 — the same guard P0.3 needed and the reason its curve clause exists.

Every comparison is PAIRED on the snippets both arms scored: the arms are the same programs, so
the item is the unit and the between-item variance is not the informative variance.

THE REPRODUCIBILITY FLOOR IS BINDING. Greedy bf16 does not reproduce per item across runs
(0.85-0.90 same-card agreement, measured 2026-08-29, cause unidentified after three interventions).
An accuracy difference smaller than that floor is noise regardless of its CI, so the floor enters
as an explicit threshold rather than a caveat in prose.
"""
from __future__ import annotations

import argparse, json, random, statistics as st, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
SEED = 20260724

# Measured floor on per-item greedy agreement (2026-08-29_reliability-floor-and-length.md).
# 1 - 0.85 = 0.15 disagreement is the worst measured case; half of it is the symmetric band
# within which a paired accuracy delta is indistinguishable from a re-run of the same condition.
FLOOR_AGREEMENT = 0.85
FLOOR_BAND = (1.0 - FLOOR_AGREEMENT) / 2.0     # 0.075


def boot_ci(vals: list[float], n_boot: int = 5000, seed: int = SEED) -> list[float]:
    if not vals:
        return [None, None]
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(st.mean(rng.choices(vals, k=n)) for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


def paired(a: dict[str, int], b: dict[str, int]) -> tuple[float, list[float], int]:
    """mean(a - b) over the snippets BOTH arms scored, plus the per-item deltas."""
    shared = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in shared]
    return (st.mean(d) if d else 0.0), d, len(shared)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="gemma12b")
    ap.add_argument("--results", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--energy-match", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = _PROJ / "data/nla/p0/steerv2" / args.host
    # Paths are resolved explicitly and never defaulted to another experiment's directory:
    # steer_stats.py's defaults point at the banked B4 run, and calling it without --out would
    # overwrite that banked result. Same failure mode, avoided by construction.
    res = Path(args.results or root / "run/steer_results.jsonl")
    bas = Path(args.baseline or root / "run/baseline.jsonl")
    em = Path(args.energy_match or root / "energy_match.json")
    out = Path(args.out or root / "kv_bypass_stats.json")
    for f in (res, bas):
        if not f.exists():
            sys.exit(f"missing {f}")

    base = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(bas) if l.strip()}
    rows = [json.loads(l) for l in open(res) if l.strip()]
    ok = [r for r in rows if "error" not in r]
    matched_alpha = (json.load(open(em))["matched_alphas"]["1.0"]["multi_alpha"]
                     if em.exists() else None)

    arms: dict[tuple[str, bool, float], dict[str, int]] = defaultdict(dict)
    parse: dict[tuple[str, bool, float], list[int]] = defaultdict(list)
    for r in ok:
        k = (r["condition"], bool(r.get("multilayer", False)), float(r.get("alpha", 1.0)))
        arms[k][r["snippet_id"]] = int(bool(r["correct"]))
        parse[k].append(int(bool(r.get("parsed"))))

    def arm(cond: str, ml: bool, a: float) -> dict[str, int] | None:
        return arms.get((cond, ml, a))

    unsteered = {s: int(bool(v["l1b_correct"])) for s, v in base.items()}
    rep: dict = {"experiment": "kv_bypass", "host": args.host, "seed": SEED,
                 "matched_alpha": matched_alpha, "floor_band": FLOOR_BAND,
                 "n_rows": len(rows), "n_errors": len(rows) - len(ok),
                 "unsteered_l1b_acc": round(st.mean(unsteered.values()), 4),
                 "arms": {}, "comparisons": {}}
    for k, flags in sorted(arms.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        cond, ml, a = k
        name = f"{'multi' if ml else 'single'}/{cond}/a={a:g}"
        rep["arms"][name] = {"n": len(flags), "acc": round(st.mean(flags.values()), 4),
                             "parse_rate": round(st.mean(parse[k]), 4),
                             "vs_unsteered": round(paired(flags, unsteered)[0], 4)}

    s_v3 = arm("V3_taskvec", False, 1.0)
    m_v3 = arm("V3_taskvec", True, matched_alpha) if matched_alpha else None
    m_v3_loud = arm("V3_taskvec", True, 1.0)
    m_rand = arm("R_random", True, matched_alpha) if matched_alpha else None

    missing = [n for n, v in [("single/V3@1.0", s_v3), ("multi/V3@matched", m_v3),
                              ("multi/V3@1.0", m_v3_loud), ("multi/R_random@matched", m_rand)]
               if v is None]
    if missing:
        # A missing arm is reported as MISSING, never omitted — an absent denominator looks
        # identical to a null unless it is named.
        rep["verdict"] = "INCOMPLETE"
        rep["missing_arms"] = missing
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print(f"[kv] INCOMPLETE — missing arms: {missing}")
        return 1

    for label, a, b in [("multi_v3_matched_vs_single_v3", m_v3, s_v3),
                        ("multi_v3_loud_vs_single_v3", m_v3_loud, s_v3),
                        ("multi_random_matched_vs_single_v3", m_rand, s_v3),
                        ("single_v3_vs_unsteered", s_v3, unsteered),
                        ("multi_v3_matched_vs_unsteered", m_v3, unsteered)]:
        mean, d, n = paired(a, b)
        ci = boot_ci(d)
        rep["comparisons"][label] = {"delta": round(mean, 4), "ci95": ci, "n_paired": n,
                                     "inside_floor": abs(mean) < FLOOR_BAND}

    C = rep["comparisons"]
    primary = C["multi_v3_matched_vs_single_v3"]
    rand = C["multi_random_matched_vs_single_v3"]
    lo, hi = primary["ci95"]
    half = (hi - lo) / 2 if lo is not None else 0.0

    # UNINFORMATIVE is checked first and short-circuits: if neither arm moved off the unsteered
    # baseline, the two arms agreeing tells us nothing about the bypass.
    if (C["single_v3_vs_unsteered"]["inside_floor"]
            and C["multi_v3_matched_vs_unsteered"]["inside_floor"]):
        rep["verdict"] = "UNINFORMATIVE"
        rep["reading"] = ("Neither arm moved off the unsteered baseline by more than the "
                          "reproducibility floor, so their agreement is not evidence for H-C0. "
                          "Same clause as P0.3's flat-curve rule.")
    elif (primary["delta"] >= 0.10 and lo is not None and lo > 0
          and rand["delta"] < primary["delta"] - half):
        rep["verdict"] = "H-C1 SUPPORTED"
        rep["reading"] = ("Closing the KV bypass recovers accuracy that single-layer steering "
                          "does not, and random directions at matched energy do not reproduce "
                          "it. Every write-side null in this thread — B4, B5, P0.2 — is "
                          "re-opened: none of them changed what the model subsequently read.")
    elif primary["inside_floor"]:
        rep["verdict"] = "H-C0 SUPPORTED"
        rep["reading"] = ("Closing the bypass changes nothing beyond the reproducibility floor. "
                          "The write-side negative survives its strongest mechanical challenge, "
                          "which is what makes it a bounded causal negative rather than an "
                          "unexplained one.")
    else:
        rep["verdict"] = "H-C1 NOT SUPPORTED"
        rep["reading"] = ("The primary moved but did not clear the pre-registered bar, or random "
                          "matched it. Reported as a negative for the bypass.")

    loud = C["multi_v3_loud_vs_single_v3"]
    loud_parse = rep["arms"].get(f"multi/V3_taskvec/a=1", {}).get("parse_rate")
    rep["h_c2_energy_flag"] = {
        "loud_delta": loud["delta"], "matched_delta": primary["delta"],
        "loud_parse_rate": loud_parse,
        "flagged": bool((not loud["inside_floor"] and primary["inside_floor"])
                        or (loud_parse is not None and loud_parse < 0.5)),
        "reading": ("If the loud arm moves while the matched arm does not, the effect is "
                    "magnitude and not delivery — and a parse-rate collapse reproduces P0.2's "
                    "0.100, which destroyed generation rather than under-delivering."),
    }
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))

    print(f"\n{'arm':<34}{'n':>4}{'acc':>8}{'parse':>8}{'vs base':>9}")
    for k, v in rep["arms"].items():
        print(f"{k:<34}{v['n']:>4}{v['acc']:>8.3f}{v['parse_rate']:>8.3f}{v['vs_unsteered']:>9.3f}")
    print(f"\n{'comparison':<38}{'delta':>8}{'ci95':>20}{'floor?':>8}")
    for k, v in C.items():
        print(f"{k:<38}{v['delta']:>8.3f}{str(v['ci95']):>20}{str(v['inside_floor']):>8}")
    print(f"\n[kv] floor band +-{FLOOR_BAND:.3f} · matched alpha {matched_alpha}")
    print(f"[kv] VERDICT: {rep['verdict']}")
    print(f"[kv] {rep['reading']}")
    if rep["h_c2_energy_flag"]["flagged"]:
        print(f"[kv] H-C2 FLAGGED: {rep['h_c2_energy_flag']}")
    print(f"[kv] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
