# NLA Steering vs CodeSteer — the accuracy bake-off

**Can post-hoc steering of a code LLM recover accuracy lost to adversarial identifier renaming — and does our residual-stream (NLA-style) steering match or beat CodeSteer's attention steering?**

*Report · 2026-09-15 · one line per method · written to be read cold.*

**Answer, in one line:** on the 50 programs and the model we are permitted to run, **nothing beats leaving the code alone** — not CodeSteer, not prompting, not any of our latent writes, not even an oracle that writes the program's own clean state — because in this runtime the renamed code is not the damaged condition in the first place.

| | |
|---|---|
| Subject model | `codellama/CodeLlama-7b-Instruct-hf` (the only permitted model on which the ASE-2026 protocol showed renaming damage; see H-R1) |
| Stimuli | 50 HumanEval-X Java programs, adversarially renamed with the ASE-2026 renamer; **434 boolean cases**, 8.7 per program (the first 50 `PACKS-PAIRED` snippets of the paper's Zenodo release) |
| Runtime | the paper's own `SteeredCausalLM` (raw prompt, its own sampling defaults `do_sample=True`, T 0.7, top_p 1.0, `torch.manual_seed(20260724)`, 3 runs per program), so every arm — theirs and ours — goes through one decoder and one parser |
| Metric | **accuracy of a sampled reply**: share of the 434 cases answered correctly over all 3 runs (the Chen et al. `c/n` estimator), case-weighted across programs as Σ accᵢ·nᵢ / Σ nᵢ; an unparsed reply counts as wrong. The paper's own `pass@1` — run 1 only, one sampled reply — is kept in the stats JSON and compared in §1; it is noisier by ≈ √3 and it ranked the arms differently |
| Statistics | paired per program, cluster bootstrap over programs, N_BOOT 10 000, seed 20260724, 95 % CIs; the H-R2a verdict reads on a Bonferroni α/3 interval |
| Our writes | `PositionReplacer` at CodeLlama layer 7, `h[p] ← ‖h[p]‖·unit(v)`, on the renamed-identifier token positions |
| CodeSteer | README-exact level-2 post-hoc steering, β_post 0.8, layers 24–31, effect gate live |
| Pre-registration | [`log/nla-harness/2026-09-14_steering-bakeoff-prereg.md`](../../log/nla-harness/2026-09-14_steering-bakeoff-prereg.md) (+ three amendments) and [`2026-09-14_better-vector-prereg.md`](../../log/nla-harness/2026-09-14_better-vector-prereg.md) |
| Results entries | [`2026-09-15_bakeoff-results.md`](../../log/nla-harness/2026-09-15_bakeoff-results.md) · [`2026-09-15_better-vector-results.md`](../../log/nla-harness/2026-09-15_better-vector-results.md) · [`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md) · [`2026-09-15_pass1-decoding-correction.md`](../../log/nla-harness/2026-09-15_pass1-decoding-correction.md) · [`2026-09-15_cn-rescore-results.md`](../../log/nla-harness/2026-09-15_cn-rescore-results.md) (H-R9, the scoring this report uses) |
| Kept numbers | [`nla/results/2026-09-15_ase_bakeoff/`](../../nla/results/2026-09-15_ase_bakeoff/) (stats JSON, ridge-fit report, dated config) |

---

## 1. The table — every method, one line

**`NLA` marks our arms.** **`NLA·write`** = a latent write through the NLA harness (`PositionReplacer`, CodeLlama layer 7, renamed-identifier token positions, β = 1); `NLA·prompt` = our text-only baselines, same harness, no write. Everything else is theirs or a reference. *No arm here uses a trained verbalizer/reconstructor pair — that arm (`ar_role`) is deferred, see §4.*

**accuracy** = share of the 434 boolean cases answered correctly, over **all 3 sampled runs** (the Chen et al. `c/n` estimator), case-weighted across programs; an unparsed reply is wrong. `Δ` is paired vs `unsteered`; `parse` = share of replies the paper's parser could read; `acc|p` = accuracy ÷ parse, separating *answering more often* from *answering better*.

| # | method | type | what it does | accuracy | Δ [95 % CI] | parse | acc\|p | note |
|---|---|---|---|--:|---|--:|--:|---|
| 1 | `original_unsteered` | ref | un-renamed code, no intervention | 0.545 | −0.051 [−0.134, +0.032] | 0.76 | 0.72 | renaming does not hurt here — it *helps* |
| 2 | `unsteered` | ref | renamed code, no intervention | **0.596** | — | 0.90 | 0.66 | nothing beats it |
| 3 | `codesteer_beta0` | ctrl | CodeSteer, β_post = 0 (identity) | 0.569 | −0.027 [−0.092, +0.039] | 0.82 | 0.70 | the re-run floor |
| 4 | `rand_prior_beta0` | ctrl | random prior, β_post = 0 (identity) | 0.569 | −0.027 [−0.095, +0.039] | 0.82 | 0.70 | identical to row 3 |
| 5 | `codesteer` | CodeSteer | slice prior, level-2 post-hoc, β 0.8, L24–31 | 0.549 | −0.047 [−0.134, +0.037] | 0.84 | 0.66 | README-exact |
| 6 | `codesteer_auto` | CodeSteer | + Eq. 10 head calibration, top-4/layer | 0.581 | −0.015 [−0.094, +0.066] | 0.86 | 0.67 | **H-R2a comparator** (+0.032 over row 5) |
| 7 | `rand_prior` | ctrl | their machinery, random prior, β 0.8 | 0.557 | −0.039 [−0.113, +0.032] | 0.86 | 0.65 | H-R2b |
| 8 | `uniform_prior` | ctrl | their machinery, uniform prior, β 0.8 | 0.561 | −0.035 [−0.110, +0.035] | 0.82 | 0.69 | prior is inert |
| 9 | `prompt` | **NLA·prompt** | one-line "identifiers may mislead" warning | 0.542 | −0.055 [−0.136, +0.028] | 0.84 | 0.65 | led on Pass@1, does not here |
| 10 | `prompt_types` | **NLA·prompt** | warning + every identifier's type and role, as text | 0.379 | −0.217 [−0.299, **−0.141**] | 0.63 | 0.60 | only arm that clearly *harms* |
| 11 | `erasure` | **NLA·write** | leave-one-out mean-difference vector | 0.567 | −0.029 [−0.118, +0.060] | 0.83 | 0.68 | best-of-ours in H-R2a |
| 12 | `foreign` | **NLA·write** | another snippet's span state | 0.523 | −0.073 [−0.149, +0.001] | 0.96 | 0.54 | H-R6b comparator |
| 13 | `combined` | **NLA·write** | `erasure` + CodeSteer β 0.8 | 0.549 | −0.047 [−0.141, +0.046] | 0.87 | 0.63 | stacking buys nothing |
| 14 | `ridge_map` | **NLA·write** | learned decoy→clean map, 98 held-out snippets (λ 100, r 256) | 0.569 | −0.027 [−0.127, +0.079] | 0.94 | 0.61 | H-R6a |
| 15 | `role_proto` | **NLA·write** | clean prototype, same declared type + AST role | 0.601 | +0.005 [−0.076, +0.083] | 1.00 | 0.60 | best non-oracle, and it is a wash |
| 16 | `swap_oracle` | **NLA·write** · ceiling | the item's *own* clean layer-7 state | **0.674** | +0.078 [−0.008, +0.165] | 0.88 | 0.76 | the ceiling, and even it does not clear 0 |

<sub>Job ids, in row order: 399863 · 399522 · 399523 · 399524 · 400250 · 400251 · 400252 · 400253 · 399686 · 401536 · 399688 · 399687 · 400254 · 401537 · 401535 · 399685.</sub>

**Why accuracy over all 3 runs, and not the registered Pass@1.** The runs were banked with the paper's own sampling defaults (`do_sample=True`, T 0.7), and their `pass@1` reads **run 1 only** — one sampled reply, two thirds of the evidence discarded. Re-scoring the same rows over all three runs (H-R9, no new GPU) shrinks every CI by ≈ √3 and moves the picture:

| | Pass@1 (run 1) | accuracy (3 runs) |
|---|--:|--:|
| `unsteered` | 0.532 | 0.596 |
| β = 0 identity floor (rows 3–4) | **+0.067** | **−0.027** |
| best non-oracle arm | `role_proto` +0.118 | `role_proto` +0.005 |
| `prompt` | +0.101 (rank 1 of ours) | −0.055 (rank 5 of 6) |
| `swap_oracle` | +0.203 [+0.061, +0.344] | +0.078 [−0.008, +0.165] |
| typical CI width | ±0.145 | ±0.083 |

The +0.067 "floor" was run-1 luck: it lands at −0.027 with three runs, inside the ±0.05 the correction entry froze as the re-read gate, so **the verdicts below read on accuracy.** The `pass@1` scoring is kept in `2026-09-15_bakeoff_stats.json`; the accuracy scoring is `…_stats_cn.json`. Verdict *words* are identical under both.

## 2. Verdicts (rules frozen before the runs)

| hypothesis | contrast | result | verdict |
|---|---|---|---|
| **H-R2a** match CodeSteer? | best of ours (`erasure`) − best of theirs (`codesteer_auto`) | **−0.014**, 95 % [−0.100, +0.069], α/3 [−0.118, +0.086] | **`MATCH-CODESTEER`** (predicted) |
| **H-R2b** does the AST slice prior matter? | `codesteer` − `rand_prior` | −0.008 [−0.086, +0.072] | **`SLICE-IRRELEVANT`** (predicted) |
| **H-R2c** restoration ratio | (steered − unsteered)/(original − unsteered) | denominator is **−0.051** — the *wrong sign*; ratios ±hundreds of % | **unreadable**, and now for a stronger reason than before |
| **H-R4** does the raw prompt enlarge the damage? | original − unsteered | −0.051 [−0.134, +0.032] | **refuted** — raw prompting hurts *clean* code, so there is no damage left to enlarge |
| **H-R6a** does a learned map beat the mean-difference? | `ridge_map` − `erasure` | +0.002 [−0.095, +0.106] | **`MAP-NOT-BETTER`** — held-out cosine 0.38 (vs 0.17) bought nothing |
| **H-R6b** does category meaning help? | `role_proto` − `foreign` | +0.078 [−0.002, +0.152] | **`CATEGORY-MEANING-INERT`** — misses the bar by 0.002 on the CI |
| **H-R6c** latent write vs the same facts as text | `role_proto` − `prompt_types` | +0.222 [+0.136, +0.305] | registered **`LATENT-BEATS-PROMPT`** — but the baseline is broken, see §3 |
| H-R6c, conservative | `role_proto` − `prompt` (the intact prompting arm) | +0.059 [−0.018, +0.137] | **`PROMPT-SUFFICES`** (predicted) — the reading carried into the ledger |
| **H-R9** does `c/n` scoring shrink the floor? | β = 0 identity arms vs `unsteered` | +0.067 → **−0.027**, CIs ±0.145 → ±0.083 | **floor gate PASSES** — the floor was sampling noise; verdicts re-read here |

## 3. How to read it

1. **Nothing beats doing nothing.** On the better-powered scoring, **no arm — theirs or ours — is above `unsteered` with a CI clearing zero.** The item's own clean state (`swap_oracle`, +0.078 [−0.008, +0.165]) is the closest and still misses; `role_proto` (+0.005) is a wash; every other steered arm, CodeSteer's two included, is *below* the un-steered baseline. The honest summary of the bake-off is: **match, in the sense that everything matches everything**.
2. **There is no damage to restore — the renamed code scores *higher*.** `original_unsteered` 0.545 vs `unsteered` 0.596, a paired −0.051 [−0.134, +0.032]. The paper's headline damage (−36 points on Qwen2.5-7B) does not exist on any model we are permitted to run (H-R1: Llama-3.1-8B +0.03, CodeGemma-7B +0.08, CodeLlama-7B +0.11 under a chat template, and now *negative* in their own raw-prompt runtime). Every method above is being ranked on a denominator that has the wrong sign, which is why H-R2c is unreadable rather than merely noisy.
3. **Parse rate is the confound on every row, and it fully explains our best arm.** `role_proto` answers **99.8 %** of the time but at `acc|p` **0.60** — the second-lowest conditional accuracy in the table, against `unsteered`'s 0.66. It buys +0.005 by answering more, not by understanding more. `foreign` shows the same mechanism failing (parse 0.96, `acc|p` 0.54, Δ −0.073), and the 2026-09-13 Gemma diagnostic found the same split independently: obfuscation makes the model *not answer*, shallow-layer writes make it *answer*.
4. **The Pass@1 ranking was run-1 luck.** `prompt` led every arm of ours at +0.101 on run 1 and sits at −0.055 over three runs; the β = 0 identity "floor" flipped from +0.067 to −0.027. Any future arm in this family must be scored `c/n` over all banked runs before it is ranked, and a single greedy or single-sample readout should not be trusted at this effect size.
5. **CodeSteer's own knobs are inert here, and its calibration now looks better than its README.** The Eq. 10 head subset beats README-exact all-heads by **+0.032** (the reverse of the Pass@1 reading, −0.032); a random attention prior is within 0.008 of the AST slice prior; a uniform prior that moves attention by 0.0002 scores like everything else; stacking our erasure onto CodeSteer buys nothing (−0.047).
6. **The one clear effect in the table is a harm.** `prompt_types` — the correct declared type and AST role of every renamed identifier, given as text — is −0.217 [−0.299, −0.141], the only interval excluding zero. It is a format collapse, not a comprehension one: parse rate falls to 0.63 because the 7B Instruct model stops emitting bare JSON once the prompt grows. That makes the registered H-R6c comparison degenerate, so the conservative reading against the intact `prompt` arm (`PROMPT-SUFFICES`) is the one carried.

## 4. What would change the answer

- **H-R9 — resolved, no GPU.** All 16 arms re-scored `c/n` over the 3 banked runs; the floor gate passed (β = 0 arms −0.027, inside ±0.05), CIs shrank ≈ √3 as predicted, and every verdict word survived while the *ranking* did not. Nothing further to run.
- **H-R7 — damage-bearing stimuli (now the only way forward).** Pre-register a damage screen on *held-out* programs (not these 50), keep the ones where renaming actually costs accuracy, re-run `unsteered` / `codesteer` / `prompt` / `role_proto` / `swap_oracle` with replies saved. Without this, every further arm ranks noise — and §3.2 now says the noise is not even centred on a real deficit.
- **H-R8 — compliance vs meaning.** Sharpened by the re-score: `role_proto` answers 99.8 % of the time at `acc|p` 0.60 vs `unsteered`'s 0.66, so the arm looks like a pure format steer. Parsed-only paired contrasts from saved replies would settle it.
- **H-R5 — the heads.** Unblocked, but weakly motivated now: there is no positive effect here whose head support is worth localising. Park it behind H-R7.
- **Deferred, not justified by these numbers:** `swap_guess` (states of a self-de-obfuscated prompt) and `ar_role` (an NLA AR trained on CodeLlama-7B at layer 7, ~1 day port + ~20 GPU-h) — H-R6a–c show no latent headroom over prompting on this host and stimulus set.

## 5. Provenance

All arms: SLURM partition `h200,h100`, one GPU each, `HF_HUB_OFFLINE=1`, `PYTHONHASHSEED=0`; job ids in the table. Runner `nla/src/ase_steer_run.py`; vectors `nla/src/ase_vectors.py` (config `nla/configs/ase_vectors.yaml`, sha `fdca8630…`, fit pool 98 programs / 3 506 identifier spans, λ 100, rank 256, seed 20260724); role tagger `nla/src/ase_roles.py` (javalang); scorer `nla/src/ase_bakeoff_stats.py` (sha `bcf4387b…`; sha `69f8e29f…` produced the Pass@1 stats and the `--scoring pass1` path of the current file reproduces that JSON exactly — the identity gate on the re-score). The accuracy scoring is `--scoring cn --packs <renamed> <original>`, which needs the packs because the per-arm jsonl carries predictions but no truth labels. Two harness faults are on record and pre-date every number here: the first CodeSteer arms ran at β = 0 ([`2026-09-14_bakeoff-beta-fault.md`](../../log/nla-harness/2026-09-14_bakeoff-beta-fault.md); those runs are the noise floor), and the ridge fit crashed once on a YAML float ([`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md); fixed, same grid). A third fault was in the *reporting*, not the runs: the first version of this report called the decoder greedy when the arms sampled at T 0.7 ([`2026-09-15_pass1-decoding-correction.md`](../../log/nla-harness/2026-09-15_pass1-decoding-correction.md)), which is what prompted the `c/n` re-score ([`2026-09-15_cn-rescore-results.md`](../../log/nla-harness/2026-09-15_cn-rescore-results.md)).
