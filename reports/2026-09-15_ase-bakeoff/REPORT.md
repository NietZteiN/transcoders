# NLA Steering vs CodeSteer — the accuracy bake-off

**Can post-hoc steering of a code LLM recover accuracy lost to adversarial identifier renaming — and does our residual-stream (NLA-style) steering match or beat CodeSteer's attention steering?**

*Report · 2026-09-15 · one line per method · written to be read cold.*

| | |
|---|---|
| Subject model | `codellama/CodeLlama-7b-Instruct-hf` (the only permitted model on which the ASE-2026 protocol showed renaming damage; see H-R1) |
| Stimuli | 50 HumanEval-X Java programs, adversarially renamed with the ASE-2026 renamer; 11.8 boolean cases per program (the first 50 `PACKS-PAIRED` snippets of the paper's Zenodo release) |
| Runtime | the paper's own `SteeredCausalLM` (raw prompt, its own sampling defaults `do_sample=True`, T 0.7, top_p 1.0, `torch.manual_seed(20260724)`, 3 runs per program), so every arm — theirs and ours — goes through one decoder and one parser |
| Metric | **case-weighted Pass@1 — i.e. accuracy of one sampled reply.** Per program, `pass@k` = share of its boolean cases that *any* of the first k runs got right, so `pass@1` is just "run 1 predicted this case's T/F label"; an unparsed reply counts as wrong. Across programs it is case-weighted, Σ pass1ᵢ·nᵢ / Σ nᵢ. Not the Chen et al. c/n estimator: three runs exist per program but the headline uses only the first, so it is noisier than averaging all three. `pass@2`/`pass@3` are in the jsonl and are *not* accuracy — "any of k right" is a ceiling |
| Statistics | paired per program, cluster bootstrap over programs, N_BOOT 10 000, seed 20260724, 95 % CIs; the H-R2a verdict reads on a Bonferroni α/3 interval |
| Our writes | `PositionReplacer` at CodeLlama layer 7, `h[p] ← ‖h[p]‖·unit(v)`, on the renamed-identifier token positions |
| CodeSteer | README-exact level-2 post-hoc steering, β_post 0.8, layers 24–31, effect gate live |
| Pre-registration | [`log/nla-harness/2026-09-14_steering-bakeoff-prereg.md`](../../log/nla-harness/2026-09-14_steering-bakeoff-prereg.md) (+ three amendments) and [`2026-09-14_better-vector-prereg.md`](../../log/nla-harness/2026-09-14_better-vector-prereg.md) |
| Results entries | [`2026-09-15_bakeoff-results.md`](../../log/nla-harness/2026-09-15_bakeoff-results.md) · [`2026-09-15_better-vector-results.md`](../../log/nla-harness/2026-09-15_better-vector-results.md) · [`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md) |
| Kept numbers | [`nla/results/2026-09-15_ase_bakeoff/`](../../nla/results/2026-09-15_ase_bakeoff/) (stats JSON, ridge-fit report, dated config) |

---

## 1. The table — every method, one line

**`NLA` marks our arms.** **`NLA·write`** = a latent write through the NLA harness (`PositionReplacer`, CodeLlama layer 7, renamed-identifier token positions, β = 1); `NLA·prompt` = our text-only baselines, same harness, no write. Everything else is theirs or a reference. *No arm here uses a trained verbalizer/reconstructor pair — that arm (`ar_role`) is deferred, see §4.*

`Δ` is paired vs `unsteered`; `parse` = share of replies the paper's parser could read; `acc|p` = Pass@1 ÷ parse — crude conditional accuracy, separating *answering more often* from *answering better*.

| # | method | type | what it does | Pass@1 | Δ [95 % CI] | parse | acc\|p | note |
|---|---|---|---|--:|---|--:|--:|---|
| 1 | `original_unsteered` | ref | un-renamed code, no intervention | **0.558** | +0.025 [−0.134, +0.188] | 0.76 | 0.73 | renaming costs only +0.025 |
| 2 | `unsteered` | ref | renamed code, no intervention | **0.532** | — | 0.90 | 0.59 | the baseline |
| 3 | `codesteer_beta0` | ctrl | CodeSteer, β_post = 0 (identity) | **0.599** | +0.067 [−0.082, +0.212] | 0.82 | 0.73 | sets the ±0.14 floor |
| 4 | `rand_prior_beta0` | ctrl | random prior, β_post = 0 (identity) | **0.599** | +0.067 [−0.083, +0.213] | 0.82 | 0.73 | identical to row 3 |
| 5 | `codesteer` | CodeSteer | slice prior, level-2 post-hoc, β 0.8, L24–31 | **0.608** | +0.076 [−0.081, +0.233] | 0.84 | 0.73 | **H-R2a comparator** |
| 6 | `codesteer_auto` | CodeSteer | + Eq. 10 head calibration, top-4/layer | **0.576** | +0.044 [−0.087, +0.175] | 0.86 | 0.67 | −0.032 vs all heads |
| 7 | `rand_prior` | ctrl | their machinery, random prior, β 0.8 | **0.585** | +0.053 [−0.086, +0.192] | 0.86 | 0.68 | H-R2b |
| 8 | `uniform_prior` | ctrl | their machinery, uniform prior, β 0.8 | **0.597** | +0.065 [−0.092, +0.211] | 0.82 | 0.73 | prior is inert |
| 9 | `prompt` | **NLA·prompt** | one-line "identifiers may mislead" warning | **0.634** | +0.101 [−0.038, +0.233] | 0.84 | 0.76 | best of ours in H-R2a |
| 10 | `prompt_types` | **NLA·prompt** | warning + every identifier's type and role, as text | **0.426** | −0.106 [−0.246, +0.036] | 0.63 | 0.67 | format collapse |
| 11 | `erasure` | **NLA·write** | leave-one-out mean-difference vector | **0.576** | +0.044 [−0.124, +0.201] | 0.83 | 0.70 | ≈ `foreign` |
| 12 | `foreign` | **NLA·write** | another snippet's span state | **0.574** | +0.041 [−0.110, +0.196] | 0.96 | 0.60 | H-R6b comparator |
| 13 | `combined` | **NLA·write** | `erasure` + CodeSteer β 0.8 | **0.544** | +0.012 [−0.123, +0.147] | 0.87 | 0.63 | worst steered arm |
| 14 | `ridge_map` | **NLA·write** | learned decoy→clean map, 98 held-out snippets (λ 100, r 256) | **0.601** | +0.069 [−0.091, +0.223] | 0.94 | 0.64 | H-R6a |
| 15 | `role_proto` | **NLA·write** | clean prototype, same declared type + AST role | **0.650** | +0.118 [−0.035, +0.271] | 1.00 | 0.65 | best non-oracle |
| 16 | `swap_oracle` | **NLA·write** · ceiling | the item's *own* clean layer-7 state | **0.735** | +0.203 [**+0.061, +0.344**] | 0.88 | 0.83 | only arm clearing the floor |

<sub>Job ids, in row order: 399863 · 399522 · 399523 · 399524 · 400250 · 400251 · 400252 · 400253 · 399686 · 401536 · 399688 · 399687 · 400254 · 401537 · 401535 · 399685.</sub>

**Noise floor.** Rows 3–4 are the paper's machinery with β = 0 — nothing steered, only the decoder re-run — and they still score +0.067. Any effect below about **±0.14** is indistinguishable from re-running the decoder. Every row except the oracle sits inside it.

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
