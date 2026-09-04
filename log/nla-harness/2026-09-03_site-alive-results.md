### Target Date: 2026-09-03 (R/S/T results — the metric is confounded by damage)

Results for the family pre-registered this morning in [`2026-09-03_site-alive-prereg.md`](2026-09-03_site-alive-prereg.md).
Read that entry first; the rules quoted below are frozen there and were **not** renegotiated after seeing data.
This entry does not amend the prereg — it records what the frozen rules returned and why the
headline verdicts should not be believed.

- **Hypotheses / what we're testing:** As pre-registered. **H-R:** the continuous clean-trace LLR
  `M_i(θ) = [log P(y_clean|x_l1b,θ) − log P(y_corr|x_l1b,θ)] / |y_clean|` is a sensitive readout that
  detects belief movement the 6-item flippable denominator could not (confirm: some belief arm clears
  ΔM ≥ +0.05 with CI excluding 0 and random at matched α does not match it). **H-S:** a direction
  optimised on M at the banked site (L32 / last prompt token / α = 1) generalises to held-out items
  (confirm: held-out ΔM ≥ +0.05, CI excl 0, on ≥ 4/5 splits, label-shuffle control flat).
  **H-T:** patching clean activations into the L1b run localises the failure to a token class × layer
  cell (gated on the sanity cell `all @ L0` recovering ≥ 0.80 of the clean–corrupt gap).

- **Setup:** juno SLURM, 1 GPU each, `google/gemma-3-12b-it`, layer 32, seed 20260724, bf16, `--deterministic` OFF.
  Code committed `0ec197f` (R) and `fea07ea` (S, T). sha256 (12): `trace_llr.py` 3d93d967ebb7 ·
  `site_optimise.py` b39ecc5d2eb0 · `patch_map.py` 112ad7c96e1c.
  Jobs **375789** R (`log/slurm/375789_trace_llr.out`, node g-08-*, finished 2026-09-04T01:38:13Z),
  **375791** S (`375791_site_opt.out`, finished 03:03:00Z, 1.361 h), **375798** T (`375798_patch_map.out`,
  node g-08-13, refused at smoke 02:0*Z). S and T were chained `afterok:375789` and each re-read R's
  verdict at start (both logged `# R verdict: R-1`). Env `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  PYTHONHASHSEED=0`. Outputs `data/nla/p0/{trace_llr,site_opt,patch_map}/gemma12b/`. Corpus = the
  banked 60 L0/L1b pairs; traces are the model's own greedy L0/L1b replies stored as token ids.

- **Results:**
  - **Sanity (R), n = 60:** the clean prompt prefers its own clean trace on **60/60** items,
    mean gap **+0.905** nats/token (flippable subset **+0.669**, all positive). Noop#1−noop#2 floor
    **0.010** < the 0.05 support threshold. The readout is well-formed.
  - **R verdict returned `R-1`**, supporting arm `V1_gloss|8.0|id_spans|SL` (ΔM **+1.486**
    [+1.215, +1.764]), `random_moves = True`. But the matched control
    `R_random|8.0|id_spans|SL` gives ΔM **+1.189** [+0.956, +1.430] — **CIs overlap heavily**; random
    delivers 80 % of the "belief" effect. At the matched-energy α = 0.149071, random (**+0.079**)
    *exceeds* both belief arms (V1 **+0.016**, V3 **+0.008**). The prompting baseline `P_prompt`
    gives **+0.073**, beating every vector arm at matched energy.
  - **Post-hoc decomposition of ΔM into its two halves** (mean per-token change, 60 items, arm-wise
    table in the run dir): the clean half **moves the wrong way in 21 of 28 arms**. For the
    "supporting" arm `V1_gloss@8.0`: Δ log P(clean) = **−0.485**, Δ log P(corrupt) = **−1.758**.
    ΔM is positive only because the corrupt trace collapses faster than the clean one.
    The three arms whose gain is genuinely carried by the clean half (≤ 6 % from the corrupt term)
    are `P_prompt` (+0.069 clean), `V1_gloss@0.149|id_spans` (+0.015) and `V3@0.149|id_spans` (+0.007);
    `R_random@0.149|id_spans` raises the clean half by **+0.056**, more than either belief vector.
  - **S verdict returned `S-LIVE`** (4/5 splits live, shuffle live on 1/5, train rises 5/5). The
    per-split numbers refute the reading:

    | split | test ΔM | test CI95 | L0 specificity (Δlogp/tok) | greedy acc | base acc | rescued flippable |
    |---|---|---|---|---|---|---|
    | 0 | +5.86 | [+3.46, +8.42] | **−14.06** | **0.000** | 0.767 | 0 / 2 |
    | 1 | +20.84 | [+14.70, +27.22] | **−48.81** | **0.000** | 0.600 | 0 / 3 |
    | 2 | +11.02 | [+7.03, +15.09] | **−30.79** | **0.033** | 0.600 | 0 / 2 |
    | 3 | +0.01 | [+0.00, +0.02] | −0.05 | 0.633 | 0.633 | 1 / 2 |
    | 4 | +9.68 | [+4.98, +14.67] | **−36.17** | **0.000** | 0.700 | 0 / 4 |

    Shuffle-control deltas stay in [−0.28, +0.26] with specificity ≥ −1.63. Cosine to the banked
    directions is ~0 throughout (|cos V1| ≤ 0.021, |cos V3| ≤ 0.017, |cos V4| ≤ 0.001).
  - **T refused at the smoke gate.** Sanity cell `all @ L0` recovered **0.784** of the clean–corrupt
    gap on 3 items, below the frozen 0.80 threshold → `T-VOID`; the sbatch assertion fired and the
    full 3,900-pass run never launched. Smoke grid: `L0|id` 0.784, `L0|code` −0.003, `L0|instr` 0.000,
    `L0|last` −0.001, `L0|all` 0.784; `L32|id` 1.072, `L32|code` 0.013, `L32|instr` 0.001,
    `L32|last` −0.002, `L32|all` **1.177** (over-recovery > 1).

- **What worked / hypothesis verdict:**
  - **H-R: REFUTED, and the metric with it.** M is well-formed (60/60 sanity, floor 0.010) but it is
    **monotone in damage**: any intervention that degrades the model raises M, because the corrupt
    trace is the model's own greedy L1b reply and collapses faster than the clean one. The frozen
    R-1 clause "random at matched α does not match" was operationalised as *random lower by more than
    the floor* (0.01) — a 0.30-nat gap between two CIs that overlap across most of their range passes
    that test. **The operationalisation was too permissive; the honest reading of the numbers is that
    V1 does not separate from random.** Recorded here rather than by editing the prereg.
  - **H-S: REFUTED by its own pre-registered specificity check.** The optimiser did not find a belief
    direction; it found "destroy the model" — held-out ΔM up to +20.8 while log P(clean | clean prompt)
    falls by up to **−48.8** nats/token and greedy accuracy goes **0.767 → 0.000**. Rescued flippable
    items: **1 across all five splits**, on the one split (3) that converged to a near-null vector.
    The frozen S rule computed specificity and greedy accuracy but did **not** gate the verdict on
    them; `S-LIVE` is a rule artifact.
  - **H-T: not tested.** The gate did its job — 0.784 < 0.80 on n = 3 — and no GPU was spent on a map
    built over an alignment that cannot round-trip. Note the near-miss and the `L32|all` = 1.177
    over-recovery are both diagnostic (see below), not noise to be threshold-shopped away.
  - **Baselines:** the mandated prompting baseline **beats every steering vector** on the only part of
    M that is not damage (clean half: `P_prompt` +0.069 vs best vector +0.015 at matched energy).
    Consistent with AxBench (Wu et al., ICML 2025).

- **Observations:**
  - The confound is structural, not a coding error: **a difference of two teacher-forced log-likelihoods
    is not a rescue proxy when the subtrahend is the model's own greedy output.** Any LLR-style metric
    built on the model's own corrupt continuation inherits it. This retro-explains why the banked α = 8
    arms looked dramatic on continuous readouts while rescuing **0** items on accuracy.
  - The three stages agree, which is the useful part: at matched energy nothing belief-shaped moves the
    clean trace more than random; when you let an optimiser off the leash it goes straight for damage;
    and prompting beats all of it. That is a coherent negative, not three unrelated nulls.
  - `L32|all` = 1.177 > 1 means patching *every* position at layer 32 overshoots the clean run — the
    L0-prompt reference and the patched run are not on the same footing (likely the position-alignment
    broadcast on `replace` blocks, whose median is 54 tokens against 44.5 annotated span tokens).
    The 0.784 near-miss at L0 is the same defect seen from the other side. Fix the alignment, do not
    move the threshold.
  - Silent-failure checks: no dead-feature/absorption analogue applies here (no dictionary in this
    family); the relevant one is **off-target steering**, and it is exactly what S caught.

- **New questions / new hypotheses:**
  - **H-R2:** a damage-controlled metric — clean-half only (Δ log P(y_clean | x_l1b)) with a
    specificity veto (Δ log P(y_clean | x_l0) ≥ −floor) — is the readout M should have been. Under it,
    R's ranking is already computable from the banked rows with **no GPU** (`--score-only`).
  - **H-S2:** re-run S optimising the clean half under a hard specificity constraint. Prediction, stated
    before running: held-out ΔM collapses to within the floor, i.e. S-DEAD — the honest version of the
    result the frozen rule mislabelled S-LIVE.
  - **H-T2:** with 1:1 alignment on `replace` blocks (or items restricted to those with equal token
    length — currently 0/60, so this means a different alignment, not a filter), `all @ L0` reaches
    ≥ 0.80 and `L32|all` lands at ~1.00.
  - Open: is split 3's near-null vector (1/2 flippable rescued, accuracy unchanged) signal or the one
    draw in five that didn't run away? n = 1; not worth a claim.

- **Next Steps:**
  1. **No new GPU work until the metric is fixed.** The R rows on disk are sufficient to compute the
     damage-controlled ranking on CPU — do that first.
  2. Write the H-R2 rule *before* looking at its output, as an amendment entry (append-only), then run
     `--score-only`.
  3. Fix `patch_map.py`'s alignment before re-submitting T; re-run the 3-item smoke against the
     unchanged 0.80 gate.
  4. Correct the standing power gate note: the flippable denominator (6/60) is still the binding
     constraint on any accuracy-level claim, independent of everything above.
