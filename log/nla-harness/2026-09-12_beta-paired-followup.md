# 2026-09-12 · FOLLOW-UP ANALYSIS — the β benefit is real and replicates; "beats one layer" is chance; H-S10 closed without GPU

**Thread:** nla-harness · **Experiment:** E-S post-hoc analysis of job **391301** · **Status:** analysis only, **no new GPU**
Extends [`2026-09-12_beta-layerset-results.md`](2026-09-12_beta-layerset-results.md) (append-only: that entry stands
as filed; this one adds the paired contrasts it did not compute and **withdraws one lead it raised**).
Source: `data/nla/ml/gemma4b/gate/beta_sweep/beta_rows.jsonl` (60 items), same seed 20260724 / N_BOOT 10 000 /
cluster bootstrap over items. Nothing here is a new pre-registered test; rules below are labelled post-hoc where
they are, and no verdict word from the prereg is changed.

### Target Date: 2026-09-12 (paired β contrasts; multiplicity discipline)

- **Hypotheses / what we're testing:** the results entry reported β cells as *means*, which cannot say whether β
  helps *within* a layer set. Three questions, all answerable from the banked rows: (1) at fixed k, does the best
  β beat β=1 **paired per item**? (2) is the single-layer β lead (H-S10, raised as "a lead, not a result") real?
  (3) does any multi-layer cell beat the best single-layer cell once multiplicity is accounted for?

- **Results:**

  **(1) Paired `SPEC` gain of best-β over β=1.0, at fixed layer set:**

  | cell | best β | paired ΔSPEC | CI excludes 0 |
  |---|---|---|---|
  | `band` (L2–13) | 0.35 | **+4.41** [+2.47, +6.62] | ✓ |
  | `topk16` | 0.35 | **+3.81** [+1.68, +6.31] | ✓ |
  | `coverage` | 0.35 | **+3.50** [+0.98, +6.42] | ✓ |
  | `faithful` | 0.35 | **+3.48** [+1.29, +5.91] | ✓ |
  | `topk29` | 0.35 | +2.34 [+0.01, +4.63] | ✓ (marginal) |
  | `spec_top12` | 0.35 | +2.31 [+0.09, +4.59] | ✓ (marginal) |
  | `live_no_tail` | 0.35 | +2.34 [−0.04, +4.67] | ✗ |
  | `all_live` | 0.35 | +2.25 [−0.14, +4.71] | ✗ |
  | `topk2` | 0.75 | +1.97 [+0.42, +3.79] | ✓ |
  | `spaced` | 0.5 | +1.81 [−0.31, +3.99] | ✗ |
  | `topk1` | 0.75 | +1.15 [−0.18, +2.56] | ✗ |
  | `topk8` | 0.35 | +1.01 [−0.27, +2.38] | ✗ |
  | `topk4` | 0.5 | +0.24 [−1.80, +2.50] | ✗ |

  **All 13 of 13 cells are positive** (sign test, two-sided **p = 2.4 × 10⁻⁴**), 6 individually clear, and the
  effect is largest exactly where the mechanism predicts (k ≥ 12). β = **0.35** is the argmax in 8 of 13 cells.

  **(2) H-S10, properly paired at k=1 (β=0.75 vs β=1.0):**

  | arm | paired Δ | |
  |---|---|---|
  | `SPEC` | +1.15 [−0.19, +2.57] | CI contains 0 |
  | `edit` | −0.03 [−2.67, +2.67] | CI contains 0 |
  | `foreign` | −1.18 [−3.68, +1.28] | CI contains 0 |
  | `swap` | +0.93 [−1.01, +2.75] | CI contains 0 |
  | **`c3`** | **−4.02 [−6.60, −1.27]** | **CI excludes 0 — β=0.75 is WORSE** |

  **(3) "Does any multi-layer cell beat the best single-layer cell?"** Best single-layer cell = `topk1`/β=0.75.
  Across **54** post-hoc cells, exactly **one** clears: `band`/β=0.35 at **+2.65 [+0.51, +4.87]**. At a nominal
  95 % interval, 54 comparisons produce **≈ 2.7 expected false positives**, so one clearing cell is precisely
  what chance delivers. 53 of 54 do not clear.

- **What worked / hypothesis verdict:**
  - **The β benefit at multi-layer is established, and it is not a lucky cell.** It replicates across
    independently pre-specified layer sets — 13/13 positive, p = 2.4 × 10⁻⁴, six clearing individually, with the
    largest gains at k ≥ 12. This is the paired statement the results entry was missing, and it strengthens
    H-S6 (`CEILING-REPAIRED`) from "the ceiling recovers" to "the *content-specific* effect also recovers".
  - **H-S10 → CLOSED, NEGATIVE, and no GPU was spent.** The single-layer β lead does not survive pairing:
    ΔSPEC +1.15 [−0.19, +2.57]. **The 0.1 GPU-h confirmatory sweep proposed in the results entry is cancelled**
    — the banked rows already answer it. Stronger: at one layer the transport arm is significantly **worse** at
    β=0.75 (`c3` −4.02 [−6.60, −1.27]), so **full replacement is correct for a single-layer write** and β earns
    its place only once several layers share the write. That is a cleaner mechanistic statement than the lead
    was: β is not a better write, it is a fix for *stacking* writes.
  - **"Multi-layer beats one layer" is WITHDRAWN as unsupported.** The results entry did not claim it, and this
    analysis confirms it must not be: the one clearing cell out of 54 is chance. H-S5's pre-registered
    `SPEC-NON-ADDITIVE` remains the verdict of record, and the post-hoc scan does nothing to soften it.

- **Observations:**
  1. **Two distinct effects were entangled in the surface.** β does two things: it stops the host damage (H-S6,
     on `swap`) *and* it recovers specific effect at multi-layer (this entry, on `SPEC`). But it does **not**
     improve a single-layer write — it makes the transport arm worse there. So β is properly described as *the
     cost of stacking, refunded*, not as a better steering operator.
  2. **β = 0.35 is the practical default** for any multi-layer write on this host (argmax in 8/13 cells), with
     β = 0.2 preferred when the ceiling rather than specificity is the target (argmax of `swap` at k = 16, 29).
  3. **The ceiling is not the constraint any more; the edit is.** At `band`/β=0.35 the arms are `swap` +69.79,
     `c3` +66.81, `edit` +36.22, `foreign` +26.37 → SPEC +9.85, i.e. **14 % of the ceiling**. Multi-layer
     geometry has been pushed about as far as it goes; what is left on the table is the edit step itself.
  4. Multiplicity was the live risk in this dataset and is now handled explicitly: the 54-cell scan is reported
     with its expected-false-positive count rather than as a finding, and the 13-cell sign test is reported
     because its cells were specified in the config **before** the run.

- **New questions / new hypotheses:**
  - **H-C4 is re-scoped and its cost is CORRECTED** (see the next entry): the earlier "~5 GPU-h" estimate in
    [`2026-09-12_12b-budget-results.md`](2026-09-12_12b-budget-results.md) counted only training. Doubling the
    SFT data means regenerating the explanation corpus, which cost **10 shards × ~3 h ≈ 30 GPU-h**
    (`log/slurm/384635_*_nla_ml_explain.out`), so 2× data is **~35 GPU-h**, not ~5. The cheap discriminating
    version is a **downward** dose-response at fixed corpus.
  - **H-S12 (band position) is now the most interesting steering question left** and is unaffected by this
    analysis: `band` L2–L13 wins with a *smaller* editable union than `coverage`, so width/position, not span
    coverage, is doing the work.

- **Next Steps:** index this entry; pre-register the re-scoped H-C4 (downward dose-response at 4B L7) and run it.
  Do not run the cancelled H-S10 sweep.
