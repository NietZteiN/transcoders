"""N6 addendum — WHY does HT12's faithfulness->correctness effect vanish in the dense corpus?

EXPLORATORY. Not part of the frozen HT12 rule; no FDR, no confirmatory claim. Its only job is
to attribute the collapse, because the dense corpus differs from the banked one in TWO ways at
once and the headline is different depending on which one matters:

  (a) case selection / length — only 91 of 117 wrong cases found a control inside the hard
      |log(len ratio)| <= log(1.5) bound (match_controls.py), so the matched set is
      length-balanced where the banked set is not (wrong traces run 1.6x longer at the median,
      making length itself a correctness proxy);
  (b) position composition — the dense run deliberately over-weights the 0.70-0.90 strip, the
      known low-faithfulness trough, which the banked sweep does not.

The arms separate them:
  banked/matched-cases-only  same banked READS, restricted to the 182 matched cases -> isolates (a)
  dense/sweep-only           dense reads minus the strip weighting                  -> isolates (b)
  dense/strip-only           the trough on its own                                  -> descriptive

Reuses `fit()` from the confirmatory module verbatim so the covariates and the case-level
permutation/bootstrap are identical; only the input rows change.

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from n6_faithfulness_mixed import PROJ, fit, load_banked, load_dense  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    pairs = json.load(open(PROJ / "data/nla/n5/pairs.json"))["pairs"]
    paired = {p["wrong"] for p in pairs} | {p["control"] for p in pairs}

    B, D = load_banked(), load_dense()
    arms = {
        "banked/all-cases": B,
        "banked/matched-cases-only": B[B.case.isin(paired)],
        "dense/all": D,
        "dense/sweep-only": D[D.read_kind == "sweep"],
        "dense/strip-only": D[D.read_kind == "strip"],
    }

    out = {}
    print(f"{'arm':32s} {'n_case':>6s} {'beta_case':>10s} {'p':>8s} {'raw_diff':>9s}  boot CI95")
    for name, df in arms.items():
        if df.case.nunique() < 20:
            print(f"{name:32s}  too few cases — skipped")
            continue
        r = fit(df, name)
        out[name] = r
        c = r["case_level"]
        print(f"{name:32s} {r['n_cases']:6d} {c['beta']:10.5f} {c['p']:8.3f} "
              f"{c['raw_diff']:9.5f}  {c['boot_ci95']}")

    outp = PROJ / "data/nla/n6/2026-08-07/ht12_decomposition.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
