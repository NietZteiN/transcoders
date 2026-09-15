# NLA Steering vs CodeSteer — the accuracy bake-off

**Can post-hoc steering of a code LLM recover accuracy lost to adversarial identifier renaming — and does our residual-stream (NLA-style) steering match or beat CodeSteer's attention steering?**

*Report · 2026-09-15 · one line per method · written to be read cold.*

| | |
|---|---|
| Subject model | `codellama/CodeLlama-7b-Instruct-hf` (the only permitted model on which the ASE-2026 protocol showed renaming damage; see H-R1) |
| Stimuli | 50 HumanEval-X Java programs, adversarially renamed with the ASE-2026 renamer; 11.8 boolean cases per program (the first 50 `PACKS-PAIRED` snippets of the paper's Zenodo release) |
| Runtime | the paper's own `SteeredCausalLM` (raw prompt, greedy, 3 sampled runs per program), so every arm — theirs and ours — goes through one decoder and one parser |
| Metric | case-weighted Pass@1: a case counts as correct if the first run predicts its T/F label; an unparsed reply counts as wrong |
| Statistics | paired per program, cluster bootstrap over programs, N_BOOT 10 000, seed 20260724, 95 % CIs; the H-R2a verdict reads on a Bonferroni α/3 interval |
| Our writes | `PositionReplacer` at CodeLlama layer 7, `h[p] ← ‖h[p]‖·unit(v)`, on the renamed-identifier token positions |
| CodeSteer | README-exact level-2 post-hoc steering, β_post 0.8, layers 24–31, effect gate live |
| Pre-registration | [`log/nla-harness/2026-09-14_steering-bakeoff-prereg.md`](../../log/nla-harness/2026-09-14_steering-bakeoff-prereg.md) (+ three amendments) and [`2026-09-14_better-vector-prereg.md`](../../log/nla-harness/2026-09-14_better-vector-prereg.md) |
| Results entries | [`2026-09-15_bakeoff-results.md`](../../log/nla-harness/2026-09-15_bakeoff-results.md) · [`2026-09-15_better-vector-results.md`](../../log/nla-harness/2026-09-15_better-vector-results.md) · [`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md) |
| Kept numbers | [`nla/results/2026-09-15_ase_bakeoff/`](../../nla/results/2026-09-15_ase_bakeoff/) (stats JSON, ridge-fit report, dated config) |

---

## 1. The table — every method, one line

Sorted by family: references, controls, CodeSteer, our prompt arms, our residual-stream arms, the oracle ceiling. **Pass@1** is the headline; **Δ vs unsteered** is the paired gain over the renamed code with no intervention; **parse rate** is the share of replies the paper's parser could read; **acc. among parsed** is Pass@1 ÷ parse rate — a crude conditional accuracy that separates *answering more often* from *answering better*.

| # | method | family | what it does | Pass@1 | Δ vs unsteered | 95 % CI | parse rate | acc. among parsed | job | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `original_unsteered` | reference | Original (un-renamed) code, no intervention | **0.558** | +0.025 | [-0.134, +0.188] | 0.760 | 0.733 | 399863 | renaming costs only +0.025 here |
| 2 | `unsteered` | reference | Renamed code, no intervention | **0.532** | — | — | 0.902 | 0.590 | 399522 | — |
| 3 | `codesteer_beta0` | control | CodeSteer with β_post = 0 (identity) | **0.599** | +0.067 | [-0.082, +0.212] | 0.817 | 0.733 | 399523 | sets the ±0.14 resolution |
| 4 | `rand_prior_beta0` | control | random prior with β_post = 0 (identity) | **0.599** | +0.067 | [-0.083, +0.213] | 0.817 | 0.733 | 399524 | same 0.5991 as above |
| 5 | `codesteer` | CodeSteer | README-exact: slice-hybrid prior, level-2 post-hoc, β_post 0.8, layers 24–31, all heads | **0.608** | +0.076 | [-0.081, +0.233] | 0.838 | 0.726 | 400250 | H-R2a comparator |
| 6 | `codesteer_auto` | CodeSteer | as above + Eq. 10 head calibration (top-4 heads/layer) | **0.576** | +0.044 | [-0.087, +0.175] | 0.863 | 0.668 | 400251 | −0.032 vs all-heads |
| 7 | `rand_prior` | control | CodeSteer machinery with a random attention prior, β 0.8 | **0.585** | +0.053 | [-0.086, +0.192] | 0.861 | 0.680 | 400252 | H-R2b |
| 8 | `uniform_prior` | control | CodeSteer machinery with a uniform prior, β 0.8 | **0.597** | +0.065 | [-0.092, +0.211] | 0.815 | 0.732 | 400253 | scores like everything else |
| 9 | `prompt` | ours · prompt | warning text prepended: "identifiers may be misleading…" | **0.634** | +0.101 | [-0.038, +0.233] | 0.836 | 0.758 | 399686 | best of ours in H-R2a |
| 10 | `prompt_types` | ours · prompt | warning + declared type and AST role of every renamed identifier, as text | **0.426** | -0.106 | [-0.246, +0.036] | 0.633 | 0.674 | 401536 | format collapse: parse 0.63 |
| 11 | `erasure` | ours · residual | leave-one-out mean-difference vector added at layer 7 span positions | **0.576** | +0.044 | [-0.124, +0.201] | 0.829 | 0.695 | 399688 | ≈ foreign |
| 12 | `foreign` | ours · residual | another snippet's span state written at layer 7 | **0.574** | +0.041 | [-0.110, +0.196] | 0.962 | 0.596 | 399687 | H-R6b comparator |
| 13 | `combined` | ours · residual | erasure at layer 7 + CodeSteer β 0.8 | **0.544** | +0.012 | [-0.123, +0.147] | 0.868 | 0.626 | 400254 | worst steered arm |
| 14 | `ridge_map` | ours · residual | reduced-rank ridge map decoy-state → clean delta, fit on 98 other snippets (λ 100, rank 256) | **0.601** | +0.069 | [-0.091, +0.223] | 0.939 | 0.641 | 401537 | H-R6a: MAP-NOT-BETTER |
| 15 | `role_proto` | ours · residual | clean-state prototype of other snippets with the same declared type + AST role, written at layer 7 | **0.650** | +0.118 | [-0.035, +0.271] | 0.998 | 0.651 | 401535 | best non-oracle arm; parse 0.998 |
| 16 | `swap_oracle` | ceiling | the item's own clean (original-code) layer-7 state written at the span positions | **0.735** | +0.203 | [+0.061, +0.344] | 0.883 | 0.833 | 399685 | only arm outside the noise floor |

**Noise floor.** Rows 3–4 are the paper's steering machinery with β = 0 — nothing is steered, only the decoder is re-run — and they score +0.067 over `unsteered`. Any effect smaller than about **±0.14** (the width of that CI) is indistinguishable from re-running the decoder. Every row except the oracle sits inside it.

## 2. Verdicts (rules frozen before the runs)

| hypothesis | contrast | result | verdict |
|---|---|---|---|
| **H-R2a** match CodeSteer? | best of ours (`prompt`) − `codesteer` | **+0.025**, 95 % [−0.108, +0.161], α/3 [−0.138, +0.194] | **`MATCH-CODESTEER`** (predicted) |
| **H-R2b** does the AST slice prior matter? | `codesteer` − `rand_prior` | +0.023 [−0.115, +0.164] | **`SLICE-IRRELEVANT`** (predicted) |
| **H-R2c** restoration ratio | (steered − unsteered)/(original − unsteered) | denominator is **+0.025**; ratios 300–800 % with ±1 000-point CIs | **unreadable** — their 104.99 % is against a −36-point damage; ours is +2.5 |
| **H-R4** does the raw prompt enlarge the damage? | original − unsteered, raw vs chat-templated | +0.025 raw vs +0.113 chat-templated | **refuted** — the raw prompt hurts *clean* code (0.558 vs 0.800), not renaming |
| **H-R6a** does a learned map beat the mean-difference? | `ridge_map` − `erasure` | +0.025 [−0.109, +0.163] | **`MAP-NOT-BETTER`** — held-out cosine 0.38 (vs 0.17) did not transfer to accuracy |
| **H-R6b** does category meaning help? | `role_proto` − `foreign` | +0.076 [−0.049, +0.196] | **`CATEGORY-MEANING-INERT`** — point estimate clears the 0.05 bar, CI does not |
| **H-R6c** latent write vs the same facts as text | `role_proto` − `prompt_types` | +0.224 [+0.078, +0.364] | registered **`LATENT-BEATS-PROMPT`** — but see §3 |
| H-R6c, conservative | `role_proto` − `prompt` (the intact prompting arm) | +0.016 [−0.118, +0.145] | **`PROMPT-SUFFICES`** (predicted) — the reading carried into the ledger |

## 3. How to read it

1. **Match, yes; surpass, no.** Our best latent arm (`role_proto`, 0.650) is +0.04 over CodeSteer and our one-line prompt (0.634) is +0.03 over it; neither is outside the noise floor, and nothing except the program's own clean state (`swap_oracle`, 0.735, **+0.203 [+0.06, +0.34]**) is.
2. **There is nothing to restore.** On these 50 programs, in the paper's runtime, renaming costs **2.5 points** (0.558 → 0.532). The paper's headline damage (−36 points on Qwen2.5-7B) does not exist on any model we are permitted to run (H-R1: Llama-3.1-8B +0.03, CodeGemma-7B +0.08, CodeLlama-7B +0.11 under the chat template, +0.025 raw). Every method above, theirs and ours, is being ranked on a denominator the size of the noise.
3. **Parse rate is a confound on every row.** `codesteer` and `prompt` *lower* the parse rate (0.84) and raise accuracy among parsed replies (0.73 / 0.76); the residual writes do the opposite — `role_proto` makes the model answer 99.8 % of the time at a conditional accuracy of 0.65. The 2026-09-13 Gemma diagnostic found the same split: obfuscation makes the model *not answer*, shallow-layer writes make it *answer*. `role_proto`'s lead is mostly compliance.
4. **The registered H-R6c baseline collapsed on format, not comprehension.** Prepending a correct fact list (`` `peers`: declared `int`, role: accumulator ``) drops the parse rate to 0.63 — the 7B Instruct model stops emitting the bare JSON. The registered verdict is therefore degenerate and the conservative reading (`PROMPT-SUFFICES`) is the one we carry.
5. **CodeSteer's own knobs are inert here.** The calibrated head subset is −0.032 vs all heads; a random prior is −0.023 vs the AST prior; a uniform prior that moves attention by 0.0002 scores the same as everything else; adding erasure to CodeSteer (`combined`) is the worst steered arm.

## 4. What would change the answer

- **H-R7 — damage-bearing stimuli (recommended next).** Pre-register a damage screen on *held-out* programs (not these 50), keep the ones where renaming actually costs accuracy, re-run `unsteered` / `codesteer` / `prompt` / `role_proto` / `swap_oracle` with replies saved. Without this, every further arm ranks noise.
- **H-R8 — compliance vs meaning.** Parsed-only paired contrasts from saved replies decide whether `role_proto` is a format steer.
- **H-R5 — the heads.** Unblocked: how many heads the NLA effect needs and whether they are CodeSteer's (comparator `codesteer` 0.608).
- **Deferred, not justified by these numbers:** `swap_guess` (states of a self-de-obfuscated prompt) and `ar_role` (an NLA AR trained on CodeLlama-7B at layer 7, ~1 day port + ~20 GPU-h) — H-R6a–c show no latent headroom over prompting on this host and stimulus set.

## 5. Provenance

All arms: SLURM partition `h200,h100`, one GPU each, `HF_HUB_OFFLINE=1`, `PYTHONHASHSEED=0`; job ids in the table. Runner `nla/src/ase_steer_run.py`; vectors `nla/src/ase_vectors.py` (config `nla/configs/ase_vectors.yaml`, sha `fdca8630…`, fit pool 98 programs / 3 506 identifier spans, λ 100, rank 256, seed 20260724); role tagger `nla/src/ase_roles.py` (javalang); scorer `nla/src/ase_bakeoff_stats.py` (sha `69f8e29f…`). Two harness faults are on record and pre-date every number here: the first CodeSteer arms ran at β = 0 ([`2026-09-14_bakeoff-beta-fault.md`](../../log/nla-harness/2026-09-14_bakeoff-beta-fault.md); those runs are the noise floor), and the ridge fit crashed once on a YAML float ([`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md); fixed, same grid).
