# 2026-08-16 — B4 v2 pre-registration: the alpha grid, the position sweep, and V2

**Status: PRE-REGISTRATION. Written before any sweep row is generated.** The only B4 numbers in
existence are the banked α = 1.0 / `last_prompt` run (`2026-08-14_n12-steering-gate.md`), which
is reused verbatim by the resume key rather than re-run.

## Why re-open a gate that already failed

The banked run tested **one** alpha at **one** injection position and failed:
V1 0.550 vs V3 0.550, an exact tie, discordant 7-vs-7, McNemar p = 1.00. `steer_run.py` always
supported `--alphas`; the run simply used `--frozen-alpha 1.0`. So "V1 does not beat V3" was
established at a single point in a two-dimensional space.

**Stated plainly, and before the numbers: this sweep is expected to strengthen a negative, not
overturn it.** The damning result in the banked run is not the tie — it is that the **antipodal
control matched random exactly** (+0.033 vs +0.033). Reversing a direction that carries signal
should hurt; it did not, so the direction has no consistent *sign*. That is not a magnitude
problem, and scaling a sign-free vector does not make it informative. What the sweep buys is the
difference between "V1 does not beat V3 at α = 1" and "V1 does not beat V3 anywhere across a 16×
range of α, at two injection positions, against seven controls" — a claim we own rather than a
limitation a reviewer supplies.

If the dose-response curve is flat, B4 is closed and the paper is a ceiling paper.

## Frozen decision rules

| | |
|---|---|
| **Primary** | gate **V1 > V3** at **α = 1.0**, positions **`last_prompt`** |
| **Success statistic** | balanced Δaccuracy over **all** items, not flip rate |
| **Secondary** | every other (α, position) cell |
| **Multiplicity** | BH-FDR across the secondary alpha family; the primary is not in its own correction |
| **Grid** | α ∈ {0.25, 0.5, 1.0, 2.0, 4.0}; positions ∈ {`last_prompt`, `all_reply`} at the primary α |

The primary is set to the **banked run's own setting**, deliberately. Choosing the
best-looking α after seeing the grid is the garden of forking paths, and this programme has been
fooled by partial results twice in one month (N11's coupling 3/3 → 3/10; B4's own n = 34 interim
tracking the oracle → an exact tie at n = 60). N7 recorded the BH family as *vacuous* because no
p-value ever entered it; this is the first real family in the programme.

Flip rate is rejected as the statistic because a vector that makes the model shout "sorting" at
everything scores brilliantly on flips while destroying answers it already had. `recovered` and
`damaged` are reported separately so the trade stays visible rather than netted away.

## What changed in the code, and why it had to change first

1. **`steer_stats.py` could not score a multi-alpha run.** It grouped by condition alone, so
   `flags[sid]` was assigned once per row and each item's accuracy became **whichever alpha
   appeared last in the file**, while `deltas` pooled every alpha into one mean. Not a pooled
   number — a scrambled one, and a plausible-looking one. Now keyed by `(condition, alpha)`,
   with a per-alpha gate, the BH family, and a dose-response table. **Regression-checked: the
   banked single-alpha result reproduces exactly** — all 7 conditions, bootstrap CIs included.
2. **The injection position was hardcoded** (`positions="last_prompt"`), though `SteerSpec`
   always accepted `all_reply` / `all` / explicit indices. Now `--positions`.
3. **The resume key now carries the position.** Without that, a position sweep would silently
   inherit the previous position's rows as "done" and report `last_prompt` numbers under a
   different label. Suffixed rather than inserted, so the 420 banked rows still resume.
4. **V2 wired.** `--with-v2` was parsed and never read; `substitute_terms` was imported and
   never called. V1 swaps the *whole* gloss, so a V1 effect could come from any of up to eight
   terms; V2 changes exactly one, via `word_edit_direction(require_single=True)`, which raises
   rather than silently making two substitutions. Items with no well-defined single-word edit
   are skipped and counted — **V2's denominator therefore differs from the battery's and its
   row must be read against its own n.**

## Hard limitation, declared now

**n is capped at 70.** `dataset_a` + `dataset_b` contain 70 snippets with usable L0/L1b pairs
and ground truth. No amount of GPU time raises it; only a corpus change does. The banked run
used 60. This is the ceiling on every B4 claim, and it is the reason the composed Java runner
(B5) matters — it moves the causal experiment onto the 862-snippet corpus that B0 and B3 share.

**No layer sweep.** The AR is trained at layer 20 only, so V1/V2 are undefined elsewhere and a
layer sweep would sweep only the no-NLA baselines — structurally favouring V3. Recorded as a
limitation, not left to be discovered.

## What would make this null

The dose-response is flat, or non-monotone in a way that tracks fluency damage rather than
recovery. Or every alpha large enough to move accuracy also wrecks the answer format, in which
case `parse_rate` falls with α and the accuracy change is a parsing artifact — `steer_stats`
reports `parse_rate` per cell for exactly this reason.

## Setup

```
env      nla-mi (run) / transcoders-mi (score)
config   nla/configs/b4_steering.yaml
runner   nla/src/steer_run.py
driver   NLA_GPU=<id> bash nla/scripts/b4_sweep.sh
seed     20260724
layer    20
score    steer_stats.py --primary-alpha 1.0
```
