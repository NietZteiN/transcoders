### Target Date: 2026-08-06 (PRE-REGISTRATION — HT12 / HT13 / HT14, frozen before any of them runs)
- **Purpose:** This entry exists to be *dated before* N6, N7 and N8 are executed. Everything below — hypotheses, models, decision rules, thresholds, seeds, exclusions — is fixed now. Anything discovered after this entry that is not specified here is **exploratory** and will be labelled as such, with no p-values claimed and no FDR membership.
- **Family:** HT12, HT13, HT14 form ONE confirmatory family. **Benjamini–Hochberg FDR at q = 0.05 across the three primary tests.** Seed **20260724** everywhere. Analyses run in `transcoders-mi` (statsmodels 0.14.6 + patsy); no GPU.

---

## Standing honesty clauses (apply to all three)
1. **The 2026-08-04 tier gaps are permanently post-hoc.** The finding that motivated HT12 (correct runs +0.010–0.019 more describable, every tier) was discovered *on this same data*. N6 **sharpens** it under proper clustering and length control; it **cannot independently confirm** it. An independent confirmation needs a fresh generation run and is out of scope.
2. **Wrong traces are 1.6× longer than correct ones** (median reply_tokens 686 vs 418). Length is therefore itself a correctness signal and enters every model as a covariate; N5's matched controls additionally reject any pair with |log(len ratio)| > log(1.5).
3. **Precise first-error localization failed validation** (2026-08-06: D2-state vs D3 agree 39%, r = +0.10; D3 self-agrees on 46%). Only **3 D2-call** cases (conf 0.90, known-answer verified) and **20 D0 slice** set-diffs are trusted. HT13's `after_error` term is therefore defined on the **fixed 0.70–0.90 relative-position strip**, not on a per-case localized point, except for those 3 cases which are reported separately as a sensitivity arm.
4. **Decode regime is fixed:** default sglang config, sequential AV, temp 0 — the regime that reproduces the banked corpus exactly (verified 60/60). Deterministic-inference mode is NOT used, as it yields a different (0/60 matching) read distribution.

---

## HT12 — reconstruction faithfulness predicts correctness (N6)
**Claim:** a read's round-trip faithfulness is higher in runs the model got right, over and above trace length, read length, tier, read kind and position-in-trace.

**Primary model** (`statsmodels.MixedLM`, groups = `task_key`):
```
rt_cos ~ correct + tier + read_kind + log(read_len) + log(reply_tokens) + u_rel + u_rel²  + (1 | case)
```
**Co-primaries, all three required:**
1. Mixed-model Wald test on `β_correct`, p < 0.05 after BH-FDR within the family.
2. **Case-level permutation test:** shuffle `correct` at the *case* level within tier, 10,000 permutations, p < 0.05.
3. **Case-level cluster bootstrap** (4,000 resamples over cases) 95% CI for the mean per-case difference excludes 0.

**CONFIRM** = all three hold. **REFUTE** = the mixed-model 95% CI for `β_correct` includes 0, **or** |β_correct| < 0.005 (half the smallest observed per-tier gap — an effect too small to be a usable signal even if statistically present). **Anything else = equivocal**, reported as such, with no promotion to an N1 secondary hypothesis.

**Pre-specified sensitivity arms** (reported, never substituted for the primary): drop `answer`-kind reads (truncated cases have none, and truncation is perfectly confounded with wrongness); `groups = snippet_id` with case as a variance component; N5's dense reads analysed as a **separate replication arm**, never pooled with the banked corpus (different sampling design).

---

## HT13 — NLA↔CoT alignment drops after the error region (N7)
**Claim:** judge-scored alignment between a read and the surrounding reasoning falls specifically *after* the error region in wrong runs — not globally.

**Instrument:** Llama-3.1-8B-Instruct (cross-family vs the Qwen target *and* the Qwen NLA), temp 0, plain generation with lenient JSON parsing (**not** constrained decoding — it segfaulted the scheduler on 2026-08-06). Per read: `{score 0-3, category ∈ agrees|generic|unrelated|contradicts, evidence}`; `align = score/3`.

**Blinding is enforced by dataflow, not by instruction:** `sanitize_windows.py` holds ground truth and emits a column-allowlisted pack (`read_id, task_key_hash, window_text, read_text, u_rel, tier`); `judge_align.py` opens only that file. Unit tests assert the allowlist and that no case's answer appears in any prompt.

**Model** (groups = `task_key`), with `after_error = 1[u_read > 0.70]` (the fixed strip, per clause 3):
```
align ~ correct × after_error + tier + log(read_len) + u_rel + u_rel² + (1 | case)
```
**CONFIRM** = the interaction coefficient is negative with p < 0.05 (BH-FDR) **AND** the *before-error* simple effect's 95% CI includes 0 — i.e. the drop localizes rather than tracking correctness globally.
**REFUTE** = the interaction CI includes 0, **or** the before-error effect matches the after-error effect in magnitude (that is a different, weaker claim and will be reported as such, not as support).

**Validity gates, all pre-specified — failing any means N7 is reported as instrument-null and the artifact's alignment sort stays disabled:**
- **shuffled-pair null** (read vs a window from a different case, same tier, same relative position): judge must separate true from shuffled at **AUC ≥ 0.65**;
- **distant-pair null** (same case, |Δu| > 0.4): guards against the judge merely detecting shared topic;
- **AR-space baseline the judge must beat:** `cos(ar.reconstruct(read), ar.reconstruct(window))` — free, no new dependency, but Qwen-family so a baseline only;
- **κ(Llama-3.1-8B, Phi-3.5-mini) ≥ 0.40** on n = 400 randomly sampled reads (unweighted κ on category, quadratic-weighted on score).

---

## HT14 — the answer appears earlier, internally, in correct runs (N8)
**Claim:** reads reference the model's eventual answer earlier (in relative trace position) in correct runs than in wrong ones.

**Measurement — NLA-based, per the user's explicit choice over logit-lens.** `mentions_answer(read, answer)` = literal match (word-boundary) ∨ numeric-value match (tol 1e-9) ∨ container match (≥⅔ of elements, in order).

**Exclusions, fixed now:** cases with no parsed answer (all 23 truncated); answers shorter than 3 characters; the cheap-token set `{0,1,2,-1,true,false,none,null,[],{}}` → all moved to a labelled sensitivity arm, never the primary. Reads anchored on sub-word digit pieces are dropped (established noise source).

**Primary statistic is the EXCESS over a foreign-answer null:** for every read, also test 20 foreign answers drawn from the same tier with matched type and digit count; report `f_own(bin) − f_foreign(bin)`. This is what prevents "the read is full of small integers" from counting as a hit.

**Onset & censoring:** `onset_NLA` = the first of 10 equal-width `u` bins where the case's own hit rate ≥ 0.5 **and stays ≥ 0.5 in the next bin**. Never reached → **right-censored, not dropped** (this is how the ~58% of wrong traces that never state their answer stay in the analysis instead of biasing the group).
**Test:** tier-stratified **log-rank** on censored onset; case-weighted rates with a 4,000-resample cluster bootstrap.
**CONFIRM** = log-rank p < 0.05 (BH-FDR) **AND** the correct group's median onset is ≥ 0.10 of relative position earlier.
**REFUTE** = p > 0.05 or the direction reverses.

**Calibration against the text measure — reported, never substituted:** on the subset where both are defined, Spearman ρ, a Bland–Altman plot, and κ on the binary early(<0.25)/late split. **Pre-specified interpretation: if ρ < 0.3, the divergence is itself the finding** — the NLA either sees the answer forming before it is written or does not see it at all — not a failure of either measure.

---
- **Next Steps:** build and launch N5 (default mode, sequential, stride 20, ~8.4 h). N6/N7/N8 run only after N5 lands, against the rules above.
