"""Print the multi-layer alpha selected by the FROZEN rule in the KV-bypass pre-registration.

The rule (log/nla-harness/2026-09-03_kv-bypass-prereg.md §4): the grid point minimising
|log r_multi(a) - log r_single(1.0)|, where r is median relative displacement of the final-layer
residual at the steered position. Reading it from the artifact rather than hardcoding a number
means the run cannot silently drift from the registered rule.
"""
import json, sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1
           else "data/nla/p0/steerv2/energy_match.json")
rep = json.load(open(src))
m = rep["matched_alphas"].get("1.0")
if not m:
    sys.exit(f"no matched alpha for single alpha=1.0 in {src}")
ratio = m["ratio"]
if not (0.5 <= ratio <= 2.0):
    # The grid is coarse; a match worse than 2x is not an energy match and would reintroduce
    # exactly the confound the rule exists to remove. Fail loudly rather than run it.
    sys.exit(f"matched displacement ratio {ratio:.2f} is outside [0.5, 2.0] — the alpha grid "
             f"does not bracket the single-layer displacement. Widen MULTI_ALPHAS and re-run.")
print(m["multi_alpha"])
