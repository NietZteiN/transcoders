# 2026-08-26 — E5b: B0's "steering hurts" side does NOT replicate at a second seed

**This is the result the outline's ">= 2 seed bases" requirement existed to find, and it
undermines the B0 headline.** Recorded immediately, before any attempt to reconcile it.

## Setup

```
config  configs/rq1_b0_seed2.yaml (byte-copy of rq1_b0.yaml, seed_base 2000)
grid    108 units / 15,018 generations / 0 failures, 4 GPUs
scored  results/tables/grid_seed2.csv
```

Forking `seed_base` forks every `unit_id`, so nothing was reused: this is a genuinely
independent replicate. It is also the first grid in the repo where the seed actually reaches
the sampler (verified in unit logs: `[artifact-fix] seeded torch/numpy/random with 2000`).

## Result: steer − obf on HumanEval, per transform

| transform | seed 1000 | seed 2000 | swing |
|---|---|---|---|
| arith_rewriting | **−8.22** | **−5.65** | 2.6 |
| branch_inversion | −2.21 | +0.66 | 2.9 |
| opaque_predicates | −3.54 | **+4.22** | **7.8** |
| loop_transformation | −0.87 | +2.92 | 3.8 |
| adversarial_rename | −0.57 | +3.16 | 3.7 |
| **family of 5** | **−1.24** [−4.41, +1.41] | **+4.46** [+0.85, +7.09] | **sign flip** |

**The sign flips.** Four of five transforms move from negative to positive. Only
`arith_rewriting` stays clearly negative in both, and even it moves 2.6 points. The two
family CIs overlap only in a narrow band around +0.9 to +1.4.

## What this does to B0

The 2026-08-16 headline was a family contrast: CodeSteer's own four transforms +3.89, the four
untested −3.14, difference **+7.02 [+2.00, +10.09]**. That rested on the untested four being
*negative*. At seed 2000 they are *positive*. The contrast as published is therefore **not
established**; it was a one-draw result presented with a CI that described sampling within the
draw rather than variability across draws.

Note the individual per-transform CIs at seed 1000 all spanned zero already, which was reported
at the time. The family contrast was the only powered statistic, and it is the one that does
not survive.

## A gap in this replicate, and it is mine

`rq1_b0_seed2.yaml` was forked from `rq1_b0.yaml`, which deliberately omits CodeSteer's original
four transforms because their seed-1000 anchors already existed under a different experiment id.
At seed 2000 those anchors do **not** exist, so the *full* old-vs-new contrast cannot be computed
at the second seed at all. What is shown above is the absolute delta on the new family, which is
comparable across seeds; the contrast is not. Closing that needs the original four at seed 2000,
about 20 GPU-h.

## Honest reading of the seed-1000 arm

Seed 1000 predates the `--seed` artifact-fix, so those runs are *uncontrolled draws* at
temperature 0.9 rather than a controlled seed. The comparison above therefore absorbs both
seed variance and ordinary run-to-run sampling noise. That does not rescue the finding — it
means the run-to-run variability of this measurement is large enough to flip the sign of a
5-transform family average, which is the substantive problem either way.

## Consequence

- **B0 must not be reported as a boundary result on current evidence.** The "attention steering
  helps where CodeSteer tested and hurts where it did not" claim depends on a sign that does not
  hold across draws.
- The FSE recommendation built on B0 (see the 2026-08-17 programme report) needs revisiting.
  Its central positive is the thing that just failed to replicate.
- What survives: `arith_rewriting` is negative at both seeds (−8.22, −5.65), which is the one
  cell CodeSteer itself flagged. A single-transform claim with two seeds is defensible where a
  five-transform family average is not.

## Next

1. Run CodeSteer's original four at seed 2000 (~20 GPU-h) to get the full contrast at a second
   seed rather than the partial picture above.
2. Treat 2 seeds as the floor, not the target. A measurement this variable needs 3+ before any
   sign-dependent claim.
3. Re-run the seed-1000 grid *seeded* so both arms are controlled draws and the comparison
   isolates seed rather than seed-plus-noise.
