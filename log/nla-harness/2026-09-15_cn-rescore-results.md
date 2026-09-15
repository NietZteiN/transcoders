# 2026-09-15 · H-R9 RESULTS — the noise floor was sampling noise, and on the honest scoring nothing beats doing nothing

**Thread:** nla-harness · **Experiment:** H-R9 (re-score, no GPU) · **Status:** RESOLVED
**Rule frozen in:** [`2026-09-15_pass1-decoding-correction.md`](2026-09-15_pass1-decoding-correction.md), pushed as
commit `75e6bda` **before** this scoring was run. **Re-scores:** [`2026-09-15_bakeoff-results.md`](2026-09-15_bakeoff-results.md)
(H-R2) and [`2026-09-15_better-vector-results.md`](2026-09-15_better-vector-results.md) (H-R6) — both stand as written;
this entry is the better-powered reading of the same banked rows.

## Hypotheses (as falsifiable predictions, stated in the correction entry)

- **H-R9.** Re-scoring all 16 arms with the Chen et al. `c/n` estimator (accuracy over cases × **all 3** banked
  runs) instead of `pass@1` (run 1 only) will **shrink the sampling noise by ≈ √3** and **leave every verdict word
  standing**.
- **Frozen re-read gate.** The H-R2a/H-R6 verdicts are re-read on `c/n` **only if** the β = 0 identity arms land
  within **±0.05** of `unsteered` — i.e. only if the +0.067 `pass@1` floor really was sampling noise. Otherwise
  the `c/n` pass is descriptive and the `pass@1` verdicts stand.

## Setup (reproducible; no GPU, nothing re-generated)

- Same 16 banked jsonl files, 50 snippets × **434 cases** × 3 runs each, `/scratch/…/ase2026/bakeoff_codellama7b/`.
  Truth labels come from the case packs (`gate/packs_ren_subset.jsonl`, `gate/packs_orig_subset.jsonl`), which the
  per-arm jsonl does not carry; the two packs agree on all 50 snippets (50/50 identical label maps, checked).
- `nla/src/ase_bakeoff_stats.py` gains `--scoring {pass1,cn}` + `--packs`; **only `load()` changes**, so pairing,
  seeds, cluster bootstrap (N_BOOT 10 000, seed 20260724) and every contrast definition are shared between modes.
  An unparsed case stays wrong in both (`pred.get(c)` is None, which equals neither label).
- **Identity gate (passed).** `--scoring pass1` on the current file reproduces the banked
  `2026-09-15_bakeoff_stats.json` **byte-identically** (modulo the two new keys `scoring`/`H_R9`), so the patch
  cannot have moved the registered statistic. sha `69f8e29f…` → `bcf4387b…`.
- Command: `python nla/src/ase_bakeoff_stats.py --dir …/bakeoff_codellama7b --original
  …/bakeoff_codellama7b/original_unsteered.jsonl --scoring cn --packs …/gate/packs_ren_subset.jsonl
  …/gate/packs_orig_subset.jsonl --out …/bakeoff_stats_cn.json`. Kept:
  [`../../nla/results/2026-09-15_ase_bakeoff/2026-09-15_bakeoff_stats_cn.json`](../../nla/results/2026-09-15_ase_bakeoff/2026-09-15_bakeoff_stats_cn.json).

## Results — accuracy (c/n, case-weighted), Δ vs `unsteered` 0.5960, 95 % CI

| arm | accuracy | Δ | 95 % CI | parse | acc/parse | was (Pass@1) |
|---|--:|--:|---|--:|--:|--:|
| `swap_oracle` | 0.6743 | +0.0783 | [−0.0084, +0.1644] | 0.883 | 0.764 | 0.735 (+0.203, CI cleared 0) |
| `role_proto` | 0.6006 | +0.0046 | [−0.0770, +0.0851] | 0.998 | 0.602 | 0.650 (+0.118) |
| **`unsteered`** | **0.5960** | — | — | 0.902 | 0.661 | 0.532 |
| `codesteer_auto` | 0.5806 | −0.0154 | [−0.0944, +0.0645] | 0.863 | 0.673 | 0.576 (+0.044) |
| `codesteer_beta0` | 0.5691 | −0.0269 | [−0.0932, +0.0386] | 0.817 | 0.697 | 0.599 (+0.067) |
| `rand_prior_beta0` | 0.5691 | −0.0269 | [−0.0931, +0.0385] | 0.817 | 0.697 | 0.599 (+0.067) |
| `ridge_map` | 0.5691 | −0.0269 | [−0.1300, +0.0788] | 0.939 | 0.606 | 0.601 (+0.069) |
| `erasure` | 0.5668 | −0.0292 | [−0.1162, +0.0568] | 0.829 | 0.684 | 0.576 (+0.044) |
| `uniform_prior` | 0.5614 | −0.0346 | [−0.1091, +0.0371] | 0.815 | 0.689 | 0.597 (+0.065) |
| `rand_prior` | 0.5568 | −0.0392 | [−0.1109, +0.0296] | 0.861 | 0.647 | 0.585 (+0.053) |
| `codesteer` | 0.5492 | −0.0469 | [−0.1338, +0.0387] | 0.838 | 0.655 | 0.608 (+0.076) |
| `combined` | 0.5492 | −0.0469 | [−0.1411, +0.0470] | 0.868 | 0.632 | 0.544 (+0.012) |
| `original_unsteered` | 0.5453 | −0.0507 | [−0.1337, +0.0324] | 0.760 | 0.717 | 0.558 (+0.025) |
| `prompt` | 0.5415 | −0.0545 | [−0.1372, +0.0275] | 0.836 | 0.648 | 0.634 (+0.101) |
| `foreign` | 0.5230 | −0.0730 | [−0.1481, +0.0010] | 0.962 | 0.544 | 0.574 (+0.041) |
| `prompt_types` | 0.3786 | −0.2174 | [−0.2964, **−0.1397**] | 0.633 | 0.598 | 0.426 (−0.106) |

**Contrasts.** H-R2a `erasure − codesteer_auto` **−0.0138** 95 % [−0.0998, +0.0694], α/3 [−0.1177, +0.0858] →
`MATCH-CODESTEER`. H-R2b `codesteer − rand_prior` −0.0077 [−0.0864, +0.0722] → `SLICE-IRRELEVANT`.
H-R6a `ridge_map − erasure` +0.0023 [−0.0948, +0.1061] → `MAP-NOT-BETTER`. H-R6b `role_proto − foreign`
**+0.0776 [−0.0016, +0.1520]** → `CATEGORY-MEANING-INERT` (misses by 0.0016 on the CI lower bound).
H-R6c registered `role_proto − prompt_types` +0.2220 [+0.1364, +0.3050] → `LATENT-BEATS-PROMPT`; conservative
`role_proto − prompt` **+0.0591 [−0.0183, +0.1368]** → `PROMPT-SUFFICES`, the reading carried.
`codesteer_auto − codesteer` **+0.0315** [−0.0471, +0.1098]. `original − unsteered` **−0.0507** [−0.1337, +0.0324].

## Hypothesis verdict

- **H-R9 ✓ SUPPORTED, both clauses, and the gate PASSES.** Typical CI half-width **±0.145 → ±0.083** (ratio 1.75
  against the predicted √3 = 1.73). The β = 0 identity floor goes **+0.067 → −0.0269**, inside the frozen ±0.05, so
  the `pass@1` floor *was* run-1 sampling luck and the verdicts are re-read here. **Every verdict word survives**
  (`MATCH-CODESTEER`, `SLICE-IRRELEVANT`, `MAP-NOT-BETTER`, `CATEGORY-MEANING-INERT`, `PROMPT-SUFFICES`).

## Observations — what the re-score changes, which is the ranking, not the verdicts

1. **Nothing beats `unsteered` any more.** On `pass@1`, 13 of 14 steered arms were above the no-intervention
   baseline and `swap_oracle` cleared zero (+0.203 [+0.061, +0.344]). On accuracy, **`unsteered` 0.5960 is third of
   sixteen**, only `swap_oracle` (+0.078, CI touching 0 at −0.008) and `role_proto` (+0.005) sit above it, and
   **no arm's CI excludes zero on the up-side**. The single interval that excludes zero in the whole table is a
   *harm* (`prompt_types` −0.217).
2. **`prompt` was the biggest illusion.** Best of ours at +0.101 on run 1; **−0.055** over three runs, 13th of 16.
   The H-R2a best-of-ours therefore moves `prompt` → `erasure`, and the comparator moves `codesteer` →
   `codesteer_auto` (whose head calibration now *beats* README-exact by +0.032, the exact reverse of the run-1
   reading, −0.032). Both substitutions are what the frozen "better of" rules require; the verdict is unchanged.
3. **The damage is now the wrong sign, not merely small.** `original − unsteered` = **−0.051**: in their raw-prompt
   runtime, on these 50 programs, the *renamed* code scores five points **higher** than the original. H-R2c was
   already unreadable with a +0.025 denominator; it is now unreadable because the denominator is negative. H-R4
   stays refuted, for a stronger reason.
4. **`role_proto`'s win is entirely compliance.** Parse 0.998 at `acc/parse` **0.602** — second-lowest in the table,
   below `unsteered`'s 0.661. It converts non-answers into answers at *worse-than-baseline* accuracy, which nets
   +0.005. `foreign` is the same mechanism with the arithmetic going the other way (parse 0.962, acc/parse 0.544,
   Δ −0.073). This is the 2026-09-13 Gemma split reproduced on a different host: obfuscation makes the model not
   answer; shallow-layer writes make it answer.
5. **H-R6b came within 0.0016 of firing.** `role_proto − foreign` +0.0776 [−0.0016, +0.152] against a 0.05 bar with
   a CI that must exclude 0. Not fired, and it must not be reported as a positive — but it is the only
   oracle-free contrast in the programme that has ever come this close, and it is a *within-write* contrast
   (both arms write, so parse-rate inflation is partly differenced out: 0.998 vs 0.962).

## New questions / hypotheses

- **H-R10 (raised).** `role_proto − foreign` at +0.0776 [−0.0016, +0.152] is a coin-flip away from the bar, and both
  arms write at near-total parse rates, so the contrast is the cleanest meaning-vs-nothing test we have. It is
  *not* re-readable on this corpus (that would be retuning after seeing data). Pre-register it as the **primary**
  contrast on H-R7's damage-bearing corpus, with the same 0.05 bar, before that corpus is scored.
- **H-R7 is now the only forward path** and its screen must change: screening for *renaming damage* on this host
  is screening for a deficit that does not exist (−0.051 here). Screen on **per-item accuracy under the renamed
  condition** instead (items the model gets wrong when renamed and right when original), and pre-register the
  yield before running.
- **H-R5 (heads) is parked.** There is no positive effect left whose head support is worth localising.
- **Method rule for the family, now general:** never rank arms on a single sampled run. Every future accuracy
  readout in this thread reports `c/n` over all banked runs, and `pass@k>1` is never reported as accuracy.
