"""P0.2 / P0.3 verdicts — the frozen decision rules, evaluated arithmetically.

Pre-registration: log/nla-harness/2026-08-27_p0-triage-prereg.md

WHY THIS EXISTS, AND WHY IT IS NOT A RENEGOTIATION. `p0_summary.py` assembles the tables and
*prints the rule text*, but a human still had to do the comparison by eye. That is exactly where a
frozen rule quietly softens. This module does the arithmetic instead: it reads the same two rules,
verbatim, and reports what they fire. It contains no threshold that is not in the pre-registration,
and it has no branch that depends on which way the answer comes out.

RULES, QUOTED FROM THE PRE-REGISTRATION
  P0.2  CHANNEL-LIMITED  iff V4 improves by >= +0.10 balanced Delta-accuracy AND its CI excludes
        zero. R_random is read alongside: a V4 gain that R_random matches does NOT count.
        Primary contrast: V4 at `all_reply` minus V4 at `last_prompt`, at the primary alpha (1.0).
  P0.3  SITE LIVE  iff the [20,20] cell differs from the unsteered baseline by more than the
        seed-to-seed spread of the baseline itself, in the same direction at both seeds.
        SITE DEAD iff [20,20] is within seed noise while [20,27] is not.
        UNINFORMATIVE iff [20,27] is also within seed noise.

ON PARTIAL CELLS. A cell whose case count differs from the modal full-scale count is reported as
PARTIAL and is refused as an input to any branch that would use it. The SITE LIVE branch never
reads [20,27], so a partial [20,27] does not block it; the SITE DEAD / UNINFORMATIVE branches do,
and return NEEDS_DATA rather than guessing. This asymmetry is in the rule as written, not added
here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
_P0 = _PROJ / "data" / "nla" / "p0"

PRIMARY_ALPHA = 1.0
P02_MIN_GAIN = 0.10          # frozen
CONDITIONS = ["V4_oracle", "V3_taskvec", "V1_gloss", "R_random"]


def _alpha_key(a: float) -> str:
    return f"alpha={int(a) if float(a).is_integer() else a}"


def p02(new_path: Path, banked_path: Path) -> dict:
    new = json.loads(new_path.read_text())
    old = json.loads(banked_path.read_text())
    k = _alpha_key(PRIMARY_ALPHA)
    rows = {}
    for c in CONDITIONS:
        o = old["conditions"].get(c, {}).get(k)
        n = new["conditions"].get(c, {}).get(k)
        if o is None or n is None:
            continue
        rows[c] = {
            "last_prompt_delta": o["delta_acc"], "all_reply_delta": n["delta_acc"],
            "contrast": round(n["delta_acc"] - o["delta_acc"], 4),
            "all_reply_ci95": n["delta_ci95"],
            "parse_rate_last_prompt": o.get("parse_rate"),
            "parse_rate_all_reply": n.get("parse_rate"),
        }
    v4 = rows["V4_oracle"]
    lo, hi = v4["all_reply_ci95"]
    improves = v4["contrast"] >= P02_MIN_GAIN
    ci_excludes_zero = not (lo <= 0 <= hi)
    # R_random is a disqualifier for CHANNEL-LIMITED, never a qualifier for its negation.
    random_matches = rows["R_random"]["contrast"] >= v4["contrast"]
    verdict = ("CHANNEL-LIMITED" if (improves and ci_excludes_zero and not random_matches)
               else "NOT CHANNEL-LIMITED")
    return {"verdict": verdict, "primary_alpha": PRIMARY_ALPHA, "rows": rows,
            "v4_contrast": v4["contrast"], "v4_improves_by_at_least_0.10": improves,
            "v4_ci_excludes_zero": ci_excludes_zero, "r_random_matches_gain": random_matches}


def p03(cells_path: Path) -> dict:
    cells = json.loads(cells_path.read_text())
    counts = [v["cases"] for v in cells.values()]
    full = max(set(counts), key=counts.count)      # modal case count == full scale
    partial = {k for k, v in cells.items() if v["cases"] != full}

    b1, b2 = cells["base_s1000"]["p_at_1"], cells["base_s2000"]["p_at_1"]
    spread = abs(b1 - b2)

    def cell(name):
        c1, c2 = cells[f"{name}_s1000"], cells[f"{name}_s2000"]
        d1, d2 = c1["p_at_1"] - b1, c2["p_at_1"] - b2
        return {
            "s1000": c1["p_at_1"], "s2000": c2["p_at_1"],
            "delta_s1000": round(d1, 3), "delta_s2000": round(d2, 3),
            "exceeds_spread_both_seeds": abs(d1) > spread and abs(d2) > spread,
            "same_direction": (d1 > 0) == (d2 > 0),
            "cases": [c1["cases"], c2["cases"]],
            "partial": f"{name}_s1000" in partial or f"{name}_s2000" in partial,
        }

    table = {n: cell(n) for n in ["L20", "L2021", "L2023", "L2027"] if f"{n}_s1000" in cells}
    l20 = table["L20"]
    if l20["partial"]:
        verdict, why = "NEEDS_DATA", "[20,20] is partial; the rule cannot be applied to it"
    elif l20["exceeds_spread_both_seeds"] and l20["same_direction"]:
        verdict, why = "SITE LIVE", "[20,20] exceeds the baseline seed spread in the same direction at both seeds"
    else:
        wide = table.get("L2027")
        if wide is None or wide["partial"]:
            verdict, why = "NEEDS_DATA", "[20,20] is within seed noise; separating SITE DEAD from UNINFORMATIVE requires a complete [20,27]"
        elif wide["exceeds_spread_both_seeds"] and wide["same_direction"]:
            verdict, why = "SITE DEAD", "[20,20] within seed noise while [20,27] is not"
        else:
            verdict, why = "UNINFORMATIVE", "the whole curve is flat — explicitly NOT to be read as SITE DEAD"
    return {"verdict": verdict, "rationale": why, "baseline_seed_spread": round(spread, 3),
            "baseline": {"s1000": b1, "s2000": b2}, "full_scale_cases": full,
            "partial_cells": sorted(partial), "cells": table}


LICENSING = {
    ("CHANNEL-LIMITED", "SITE LIVE"):
        "the channel was the bottleneck -> Phase 1a (position-allocation; re-run the B4 gate at the better allocation)",
    ("NOT CHANNEL-LIMITED", "SITE DEAD"):
        "single-layer late intervention cannot move this task -> Phase 1b (readout paper with a bounded causal negative)",
    ("CHANNEL-LIMITED", "SITE DEAD"):
        "CONTRADICTORY — report neither; diagnose",
    ("NOT CHANNEL-LIMITED", "SITE LIVE"):
        "the site works but belief-shaped writes do not -> strongest support for 'no item-level belief' -> Phase 1b",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p02-new", default=str(_P0 / "p02_allreply" / "steer_stats.json"))
    ap.add_argument("--p02-banked", default=str(_PROJ / "data/nla/n12/steer_stats_sweep.json"))
    ap.add_argument("--p03-cells", default=str(_P0 / "p03" / "cells_scored.json"))
    ap.add_argument("--out", default=str(_P0 / "P0_VERDICT.json"))
    a = ap.parse_args()

    r2 = p02(Path(a.p02_new), Path(a.p02_banked))
    r3 = p03(Path(a.p03_cells))
    lic = LICENSING.get((r2["verdict"], r3["verdict"]),
                        "no licensing cell — a stage is undecided")

    print(f"P0.2  {r2['verdict']}   (V4 contrast {r2['v4_contrast']:+.4f} vs frozen +{P02_MIN_GAIN})")
    for c, v in r2["rows"].items():
        print(f"        {c:<12} last_prompt {v['last_prompt_delta']:+.4f} -> all_reply "
              f"{v['all_reply_delta']:+.4f}  ({v['contrast']:+.4f})   parse "
              f"{v['parse_rate_last_prompt']:.3f} -> {v['parse_rate_all_reply']:.3f}")
    print(f"\nP0.3  {r3['verdict']}   ({r3['rationale']})")
    print(f"        baseline seed spread {r3['baseline_seed_spread']:.3f} pts")
    for n, v in r3["cells"].items():
        flag = "  PARTIAL — not used" if v["partial"] else ""
        print(f"        {n:<7} {v['s1000']:.3f}/{v['s2000']:.3f}  d {v['delta_s1000']:+.3f}/"
              f"{v['delta_s2000']:+.3f}{flag}")
    print(f"\nLICENSES: {lic}")

    out = {"p02": r2, "p03": r3, "licenses": lic}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
