"""Resolve the alpha for a coverage arm, by one of TWO pre-registered conventions.

Reading the value out of the calibration artifact rather than hardcoding it means a run cannot
drift from the registered rule, and the convention actually used is printed into the job log.

  --convention delivered  (primary)
      Equalise DELIVERED perturbation at the last prompt token — the position every coverage
      setting shares and where the answer is produced. Answers: "does moving the edit to the
      identifier sites, at equal delivered effect on the answer position, change the outcome?"
      Can be UNREACHABLE: 45 identifier tokens reach the answer position only through attention,
      so they deliver far less there than a direct edit. If the grid does not bracket the target,
      this exits 2 and the arm is reported as unreachable — which is a finding, not a failure.

  --convention energy  (secondary, pre-registered alongside)
      Equalise TOTAL injected energy across edited positions: alpha_cov = alpha_ref / sqrt(n).
      Answers the different question: "same intervention budget, spent at the identifiers instead
      of at the answer position?" Needs no calibration data beyond the token count — it is
      arithmetic — so it is always available even when `delivered` is unreachable.

Neither convention is privileged by nature. They answer different questions and BOTH are run, so
the choice is not a degree of freedom exercised after seeing outcomes.

EXIT CODES are distinct on purpose: 0 resolved · 2 unreachable-by-rule · 1 genuine error. A
principled refusal and a crash must not look the same to a scheduler.
"""
import argparse, json, math, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="data/nla/p0/steerv2/gemma12b/coverage_match.json")
ap.add_argument("--coverage", required=True)
ap.add_argument("--convention", choices=["delivered", "energy"], default="delivered")
ap.add_argument("--band", type=float, nargs=2, default=[0.5, 2.0])
a = ap.parse_args()

f = Path(a.file)
if not f.exists():
    print(f"missing {f}", file=sys.stderr); sys.exit(1)
rep = json.load(open(f))

if a.convention == "energy":
    n = rep.get("tokens_per_item_median")
    ref = rep["reference"]["alpha"]
    if a.coverage == rep["reference"]["coverage"]:
        print(ref); sys.exit(0)
    if not n:
        print("no token count in calibration artifact", file=sys.stderr); sys.exit(1)
    # sum of squared coefficients held constant against the 1-token reference
    print(round(ref / math.sqrt(n), 6)); sys.exit(0)

m = rep.get("matched", {}).get(a.coverage)
if not m:
    print(f"no matched entry for coverage {a.coverage!r}", file=sys.stderr); sys.exit(1)
lo, hi = a.band
if not (lo <= (m.get("ratio") or 0) <= hi):
    print(f"UNREACHABLE: best ratio {m.get('ratio')} outside [{lo}, {hi}] at alpha "
          f"{m.get('alpha')} (r={m.get('r')}) vs reference r={rep['reference']['r']}. "
          f"The identifier sites cannot deliver the reference perturbation to the answer "
          f"position at any alpha on the grid.", file=sys.stderr)
    sys.exit(2)
print(m["alpha"]); sys.exit(0)
