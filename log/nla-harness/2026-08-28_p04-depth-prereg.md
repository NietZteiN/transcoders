### Target Date: 2026-08-28 (P0.4 pre-registration — is the instrument at the wrong depth?)

**Status: PRE-REGISTRATION. Written before a single P0.4 generation exists.** P0.4 was left
"proposed, NOT pre-registered, wired but disabled" in
[`../../nla/continuation/01_RESUME_PHASE0.md`](../../nla/continuation/01_RESUME_PHASE0.md) with the
explicit instruction *"write its decision rule first"*. This is that rule. The P0.1/P0.2/P0.3
verdicts quoted below are from [`2026-08-28_p0-triage-results.md`](2026-08-28_p0-triage-results.md)
and are **not** renegotiated here.

- **Hypotheses / what we're testing:**

  Phase 0 closed on licensing row 4 — *the site works but belief-shaped writes do not*. Every
  intervention in the programme has been written at **layer 20 of 28**, because that is where the
  released `kitft/nla-qwen2.5-7b-L20-{ar,av}` pair was trained, not because anything measured said
  20 was the right depth. P0.1 then measured it, and the answer was uncomfortable:

  | | layer 13 | layer 20 |
  |---|---|---|
  | cross-item coherence (LOO) | **0.543** ± 0.108 | 0.459 ± 0.172 |
  | relative magnitude ‖Δ‖/‖h‖ | 0.0443 | **0.0675** |
  | rotation vs Δ₂₀ | 0.210 | 1.000 |

  Coherence peaks at 13; magnitude peaks at 20; the two directions are nearly orthogonal
  (cos = 0.21). **V3 and V4 are pure activation differences and are defined at every layer**, so
  this is testable with no autoencoder, no training, and no `allocation_replication`.

  **Read the whole P0.1 curve before believing "L13 is special."** Coherence over ℓ ∈ [0, 17] is
  flat in the band 0.49–0.54 — L13's 0.543 is the argmax of eighteen near-ties. The curve's only
  real structure is a **regime change at layer 18**, where coherence starts falling
  (0.502 → 0.486 → 0.469 → 0.459) exactly as relative magnitude jumps
  (0.0394 → 0.0600 → 0.0657 → 0.0675). So the honest hypothesis is not *"layer 13"* but
  *"the coherent low-magnitude band below 18, versus the incoherent high-magnitude band at and
  above 18."* The design below is built to tell those two apart rather than to confirm either.

  - **H-P04a (primary — is depth the constraint?).** If the belief-injection failure is a *depth*
    failure, then the **oracle** — which has the item's own true−decoy difference and therefore
    cannot fail for lack of content — should recover accuracy when written at 13 and not at 20.
    V4 is the ceiling; if the ceiling does not move, no NLA-derived direction at that depth will.
  - **H-P04b (band or point?).** If H-P04a fires at 13, does it also fire at **6** — same coherence
    plateau (0.531 vs 0.543, a gap of 0.012 against a per-item SD of ~0.11), seven layers away,
    rotation 0.068 vs Δ₂₀? A band effect and a point effect license completely different claims.
  - **H-P04c (does the no-NLA bar move too?).** V3, the contrastive task vector, is the bar B4 was
    measured against. Reported at every layer, secondary to V4.

- **Setup / how it will be run:**

  ```
  runner     nla/src/steer_run.py --layer {6,13,20} --positions last_prompt
             --alphas 0.25,0.5,1.0,2.0,4.0 --only-conditions V3_taskvec,V4_oracle,R_random
  corpus     the same 60 L0/L1b pairs steer_run.load_pairs yields (dataset_a + dataset_b),
             ground truth on the L1b side — identical to B4, P0.1 and P0.2
  model      Qwen/Qwen2.5-7B-Instruct, bf16, greedy (do_sample=False)
  seed       20260724
  scale      3 layers x 3 conditions x 5 alphas x 60 items = 2,700 steered generations
  hardware   juno h200, one GPU per layer arm, three arms in parallel
  out        data/nla/p0/p04/L{06,13,20}/
  ```

  **Layer 20 is re-run on juno rather than taken from the bank.** The banked B4 `last_prompt` rows
  were generated on csr-94608 (A6000, older torch); the L13/L06 arms will run on juno (H200, torch
  2.11.0). Comparing a new arm against a bank collected on different hardware confounds depth with
  stack. Both arms therefore run here, and the standing "resume, don't recompute" constraint is
  honoured differently: the bank is used as a **reproduction check**, not as the comparison arm.

  **Only three conditions run.** V1/V2/F/A all route through the AR reconstructor, which was
  trained at layer 20 — reconstructing into layer 13's basis is not a defined operation and would
  be a category error, not a control. `--only-conditions` restricts the battery accordingly, and
  `steer_run.py` will skip loading the AR entirely when no AR-dependent condition survives.

  **Why no second seed.** Decoding is greedy and the item set is the full 60 with no subsampling,
  so the run is deterministic given the corpus; the only seed-dependent condition is `R_random`,
  which already draws a *different* direction per item (`seed = SEED + hash(sid) % 9973`) and is
  therefore an average over 60 independent draws rather than a single one.

  **Two validation gates. Both must pass before any number below is believed.**

  1. **Layer-indexing gate.** `ActivationExtractor` hooks `layers[K]`; `ActivationSteerer` hooks
     `layers[K]`. They are documented as "keep in sync" and never checked. For each
     ℓ ∈ {6, 13, 20}: read h at ℓ, steer at ℓ with a known unit direction d̂ and α, re-read, and
     require the observed change to equal `α‖h‖d̂` to within **1e-3 relative**. A miss means the
     write and the read are at different depths and every P0.4 curve is mislabelled — in which case
     **P0.4 is not reported at all**, exactly as P0.1's gate was specified.
  2. **Cross-cluster reproduction gate.** The juno L20 arm must reproduce the banked csr-94608 L20
     arm: per-item correctness agreement **≥ 0.90** for V3 and V4 at α = 1.0, and
     **|Δacc_juno − Δacc_banked| ≤ 0.05**. Failing this does *not* invalidate P0.4 — the L13-vs-L20
     contrast is within-cluster either way — but it must be reported as a stack effect, and the
     banked B4 numbers may then not be quoted alongside juno numbers in the same table.

- **Decision rules — frozen, and not to be renegotiated once the numbers are visible:**

  Statistic throughout: **balanced Δaccuracy** (accuracy change across all 60 items, not flip
  rate), `recovered` and `damaged` reported separately and never netted. All contrasts are
  **paired on the same items** (exact McNemar on discordant pairs + percentile bootstrap CI on the
  per-item difference). **Primary alpha = 1.0**, pre-registered; the other four alphas are
  secondary and carry BH-FDR adjustment across the alpha family. The +0.10 threshold is the same
  one P0.2 used, reused for comparability, and is arbitrary — which is why it is frozen here.

  **PRIMARY (H-P04a). The verdict is the V4 contrast at α = 1.0 and nowhere else.**

  **DEPTH-LIMITED** iff all four hold at layer 13:
  - (a) **V4@13 − V4@20 ≥ +0.10**, paired, bootstrap CI excluding zero;
  - (b) **V4@13's own Δacc vs the unsteered L1b baseline ≥ +0.10**, CI excluding zero — beating a
    layer that does nothing is not the same as doing something;
  - (c) **V4@13 − R_random@13 ≥ +0.10** — a gain the norm-matched random direction also produces
    is generic perturbation, not delivery. This is P0.2's guard, restated;
  - (d) **parse rate at layer 13, α = 1.0 ≥ 0.80.** P0.2's negative was not a null but a
    *destruction* — widening the write drove V1's parse rate to 0.100. A Δaccuracy computed on
    replies that mostly no longer emit an `Output:` line is an artifact of the denominator.

  Verdict text: *the instrument is at the wrong depth.* This licenses re-running the B4 belief
  gate with an NLA trained at the better depth, and it is the one result that would reopen
  Phase 0's licensing row 4.

  **NOT DEPTH-LIMITED** iff (a) fails — the contrast is under +0.10 or its CI spans zero. Depth is
  not the constraint. Phase 0's row-4 reading then has a **third independent leg**: the channel is
  not the bottleneck (P0.2), the site is not inert (P0.3), and the depth is not wrong (P0.4).
  This is the outcome that most strengthens the Phase-1b readout paper, and it is a publishable
  pre-registered negative.

  **DESTRUCTIVE** iff (a)–(c) hold but (d) fails. Reported as the same over-delivery pathology
  P0.2 found at `all_reply`, **not** as a depth result. A write that destroys generation is not a
  write that carries content.

  **SECONDARY (H-P04b), applied only if layer 13 returns DEPTH-LIMITED.** The identical four-part
  rule is evaluated at layer 6.
  - **BAND** — layer 6 also satisfies (a)–(d). The effect belongs to the coherent low-magnitude
    band below 18, not to any one layer, and the claim is about the band.
  - **POINT** — layer 13 satisfies it and layer 6 does not. **This is to be reported as an
    unexplained layer-specific effect awaiting replication, and explicitly NOT as "coherence
    predicts steerability"** — the coherence gap between 6 and 13 is 0.012 against a per-item SD of
    ~0.11, so the P0.1 curve cannot support that inference no matter how the steering comes out.
    An argmax over eighteen near-ties is a hypothesis, not an explanation.

  **SECONDARY (H-P04c).** V3 is scored under the same four-part rule and reported at all three
  layers. V3 clearing the bar while V4 does not is **incoherent** — the oracle strictly dominates
  the leave-one-out task vector in information — and is to be diagnosed as a bug, not reported.

  **Standing constraints, restated so they are not renegotiated:**
  - The gate at the pre-registered α = 1.0 is the verdict; a secondary alpha clearing the rule is a
    lead, not a result.
  - `recovered` / `damaged` separate, never netted into Δaccuracy.
  - P0.4 **cannot overturn P0.2 or P0.3**, which it does not touch. A DEPTH-LIMITED verdict does
    not mean the channel was wide enough or the site was live for belief; it means the depth was
    wrong *as well*, and Phase 0 would be re-opened rather than reversed.
  - Full-sample only. This programme has had six interim-to-full reversals; **no partial P0.4 table
    is to be reported**, and the scorer returns `NEEDS_DATA` rather than a verdict on a short cell.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** `nla/src/p04_gate.py` (indexing gate) → `nla/scripts/p04_sbatch.sh` (three arms)
  → `nla/src/p04_score.py` (frozen rules) → results in a new dated entry.
