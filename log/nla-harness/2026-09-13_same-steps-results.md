### Target Date: 2026-09-13 (H-C10 same-steps control — `DATA-NOT-STEPS`)

- **Hypotheses / what we're testing:** the single confound left in H-C4's `DATA-LIMITED` verdict
  ([`2026-09-12_dose-response-results.md`](2026-09-12_dose-response-results.md)): `--train-frac` at a fixed
  1 epoch varies **unique rows** and **optimiser steps** together, so the +8.96-nat top step could have been
  bought with either. Rules frozen in [`2026-09-12_same-steps-prereg.md`](2026-09-12_same-steps-prereg.md) with
  `A = f050e2` (half the rows, **2 epochs**), `B = f100` (all rows, 1 epoch — **equal steps to A**),
  `C = f050` (half the rows, 1 epoch, half the steps), all paired per item at L7:
  **`STEPS-NOT-DATA`** if `B − A` CI contains 0 *and* `A − C` ≥ +1.46 with CI excluding 0 ·
  **`DATA-NOT-STEPS`** if `B − A` ≥ +1.46 with CI excluding 0 · **`MIXED`** otherwise.
  **Prediction recorded before the run: `DATA-NOT-STEPS`** (stated with low confidence and flagged as an
  extrapolation from generic SFT behaviour, after `DATA-SATURATED` was refuted the day before).

- **Setup:** job **391966**, `sbatch nla/scripts/nla_samesteps_sbatch.sh`, node **g-08-05** (H200 NVL, 143 771 MiB),
  `2026-09-13T02:49:03Z → 04:37:31Z`, **1:48:29 elapsed, rc=0, 1.81 GPU-h** (prereg estimated ~1.8 — exact).
  Host Gemma-3-4B-it, layer **7**, `nla/configs/nla_ml.yaml` + `nla/configs/nla_ml_gate_dose.yaml`, seed 20260724,
  `--deterministic` OFF, N_BOOT 10 000, cluster bootstrap over the same **60 items / 471 repaired spans** as every
  run in this family. Own root `/scratch/juno/jvl210002/nla_ml_dose/f050e2` with `corpus`/`explain`/`acts`
  symlinked from the shared dose base, so no banked or dose pair was touched. Scored with **`--ignore-liveness`**
  per the standing deviation ([`2026-09-12_dose-liveness-deviation.md`](2026-09-12_dose-liveness-deviation.md)),
  which is why the gate prints `reportable=False` — by design, all dose arms apples-to-apples.
  sha256 (12): `nla/src/nla_train.py` `2fe146f217ea` · `nla/src/nla_ml_gate.py` `1f7a9d3191a9` ·
  `nla/src/dose_score.py` `4c1b93368046` · `nla/configs/nla_ml_gate_dose.yaml` `bd34400976a2` ·
  `nla/configs/nla_ml.yaml` `b31f5e27cb97` · `nla/scripts/nla_samesteps_sbatch.sh` `f81baa16e79d`.
  Pre-flight gates in-job: `pytest nla/tests/test_dose_score.py` **9 passed**,
  `pytest nla/tests/test_beta_sweep.py -k subsample` **1 passed / 13 deselected**.
  **The equal-steps contract is confirmed in the run's own first training line** — `[sft_av] L7: 43411 train rows,
  858 holdout, injection_scale 5100.0, micro 8 x accum 16 = 128, 678 steps` — i.e. half the rows × 2 epochs
  lands on exactly the **678** steps `f100` took at full data × 1 epoch.
  Scoring (no GPU): frozen rule applied by an inline script reusing `dose_score.boot`/`paired`/`STEP_NATS`;
  output `data/nla/ml/gemma4b/gate/c10_samesteps_stats.json`.

- **Results:** per-arm means at L7 over 60 items (nats of `G_sum`):

  | arm | rows | epochs | steps | `S_c3` | `S_swap` | `S_edit` | `S_foreign` | **`SPEC`** | AR holdout `fve` | AV gap |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `C = f050` | 43 411 | 1 | 339 | +57.58 | +73.15 | +30.41 | +24.93 | **+5.49** | 0.3264 | +0.1454 |
  | **`A = f050e2`** | 43 411 | **2** | **678** | **+61.19** | +73.15 | +33.09 | +27.11 | **+5.98** | **0.3856** | +0.1790 |
  | `B = f100` | 86 822 | 1 | **678** | +66.54 | +73.15 | +33.67 | +24.95 | **+8.72** | 0.3984 | +0.1681 |

  Paired per-item contrasts (bar **+1.46**):

  | contrast | `S_c3` | `SPEC` |
  |---|---|---|
  | **`B − A`** — unique rows, at **equal steps** | **+5.35 [+1.81, +9.13]** | **+2.74 [+0.77, +4.89]** |
  | `A − C` — the second epoch over the same rows | +3.61 **[−0.11, +7.49]** | +0.50 **[−0.94, +2.05]** |
  | `B − C` — H-C4's original top step (reproduced) | +8.96 [+5.10, +13.17] | +3.23 [+1.03, +5.70] |

  Gate telemetry: `S_swap` **+73.15 identical** across all three arms (it never touches the trained pair, as
  designed); `SELF` **+0.000**; identity passes; `n_editable_L7` 119; `M-NO-GAIN` again (vacuous at one layer);
  `cos_cycle` 0.9932–0.9935 and `n_no_tags` 0 in all three.

- **What worked / hypothesis verdict:** **H-C10 → `DATA-NOT-STEPS` (frozen rule, primary `S_c3`).** `B − A` =
  **+5.35 [+1.81, +9.13]** clears the +1.46 bar with the CI excluding 0, so the first clause of the rule fires
  outright. The `STEPS-NOT-DATA` branch is doubly blocked: `B − A` does not contain 0, *and* `A − C` = +3.61
  **[−0.11, +7.49]** does not clear. **`SPEC` — the quantity H-C8 established as the one the programme wants —
  gives the same verdict more sharply:** `B − A` = **+2.74 [+0.77, +4.89]** while the second epoch contributes
  **+0.50 [−0.94, +2.05]**, i.e. of the +3.23 total step, unique data carries ~85 % and repetition essentially
  nothing. **My pre-registered prediction `DATA-NOT-STEPS` is SUPPORTED** — recorded plainly because the previous
  prediction in this family was refuted; the low confidence I attached to it was not vindicated or needed.
  **Consequence: H-C4's `DATA-LIMITED` survives its last confound, and H-C7 (~30 GPU-h corpus regeneration) is
  justified by the frozen rule rather than by preference.**

- **Observations:**
  - **The decomposition is clean.** H-C4's +8.96 splits into **+5.35 unique data (CI excludes 0)** and
    **+3.61 extra steps (CI includes 0)**. The steps half is not zero and should not be reported as zero — it is
    *unresolved* at n=60, the same honesty H-S5's `SPEC-NON-ADDITIVE` required. On `SPEC` the steps term is much
    smaller (+0.50) and the data term is the whole effect.
  - **The proxies say "steps" while the causal readout says "data" — a direct caution against using `fve` as a
    stand-in for fidelity.** AR holdout `fve` runs 0.3264 → **0.3856** → 0.3984: the second epoch recovers ~82 %
    of the reconstruction gap. But on `S_c3` it recovers only 40 %, and on `SPEC` only 15 %. H-C6 said proxies
    track *within a host*; this refines it — they track **optimisation**, and the extra unique rows buy causal
    effect the reconstruction proxy cannot see. Anything in this programme that has ever been gated on `fve`
    (including the liveness floor that forced the standing deviation) inherits that caveat.
  - **`foreign` moved in `f050e2` (+27.11) where it had been flat across H-C4's arms (+25.8/+24.9/+25.0).** So the
    second epoch raised the *non-specific* component while leaving the specific one alone — which is exactly why
    `A`'s `SPEC` (+5.98) barely beats `C`'s (+5.49) despite its `S_edit` rising +2.68. Repetition buys decoy
    destruction, not content.
  - No silent-failure signature: `SELF` exactly 0.000, `S_swap` bit-stable at +73.15 across arms, zero no-tag
    generations, `cos_cycle` ≈ 0.993 everywhere.

- **New questions / new hypotheses:**
  - **H-C12:** is the +3.61 steps term real but underpowered, or zero? A third arm at **half the rows × 4 epochs**
    (1 356 steps, still half the data) separates "repetition helps a little" from "repetition does nothing" and
    costs the same ~1.8 GPU-h. Worth it only if H-C7 is *not* bought, since H-C7 answers the practical question
    directly.
  - **H-C13:** the `fve`-vs-causal divergence above is testable as a claim rather than an observation — across all
    four arms now on disk (`f025`/`f050`/`f050e2`/`f100`), does `fve` predict `S_c3` at all once steps are
    controlled? n=4 is too few to fit, but the *ordering* is already violated (`f050e2` > `f100` on AV gap while
    `f100` > `f050e2` on every causal readout). No GPU.
  - Unchanged and now unblocked: **H-C7** at ~30 GPU-h with the forecasts frozen in the open ledger (transport
    ratio ≈ 0.966, `SPEC` ≈ +12); **H-C11** (the released pair's flat edit at ~10× volume *with* RL still argues
    against a pure-volume account at the top end) should be settled **before** H-C7 is bought, since it is free.

- **Next Steps:** report job **391968** (accuracy, started 04:3xZ on the same queue) and **392094** (tier ladder)
  as each lands; submit the held erasure-vector run ([`2026-09-12_erasure-vector-prereg.md`](2026-09-12_erasure-vector-prereg.md))
  once 391968 reports. Do **H-C11** (no GPU) before deciding H-C7.
