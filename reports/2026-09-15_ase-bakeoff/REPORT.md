# NLA Steering vs CodeSteer — the accuracy bake-off

**Can post-hoc steering of a code LLM recover accuracy lost to adversarial identifier renaming — and does our residual-stream (NLA-style) steering match or beat CodeSteer's attention steering?**

*Report · 2026-09-15 · one line per method · written to be read cold. Headline table re-run 2026-09-15 with the chat template; the earlier raw-prompt table is kept in §4 as the runtime comparison.*

**Answer, in one line:** on the 50 programs and the model we are permitted to run, **no steering method beats leaving the code alone** — not CodeSteer, not prompting, not any of our latent writes, not even an oracle that writes the program's own clean state. That answer now stands on a *working* prompt: the first pass ran in the paper's raw-prompt runtime, which held CodeLlama at chance (0.545 against chance 0.500) and inverted the renaming damage; applying the model's own chat template inside their runtime lifts **every** arm by +0.11 to +0.23 and restores a renaming deficit of the right sign (+0.058), and CodeSteer is still inert (−0.017 / −0.002).

| | |
|---|---|
| Subject model | `codellama/CodeLlama-7b-Instruct-hf` (the only permitted model on which the ASE-2026 protocol showed renaming damage; see H-R1) |
| Stimuli | 50 HumanEval-X Java programs, adversarially renamed with the ASE-2026 renamer; **434 boolean cases**, 8.7 per program (the first 50 `PACKS-PAIRED` snippets of the paper's Zenodo release) |
| Runtime | the paper's own `SteeredCausalLM`, with **one** change — `_build_prompt` wrapped in CodeLlama's chat template, prompt ids gated identical to `apply_chat_template` — and everything else theirs: their sampling defaults (`do_sample=True`, T 0.7, top_p 1.0, top_k 7, `torch.manual_seed(20260724)`), 3 runs per program, their strict-JSON parser, so every arm goes through one decoder and one parser |
| Metric | **accuracy of a sampled reply**: share of the 434 cases answered correctly over all 3 runs (the Chen et al. `c/n` estimator), case-weighted across programs as Σ accᵢ·nᵢ / Σ nᵢ; an unparsed reply counts as wrong. Chance is exactly **0.500** (packs are balanced True/False by construction) |
| Statistics | paired per program, cluster bootstrap over programs, N_BOOT 10 000, seed 20260724, 95 % CIs; H-R7a reads on Bonferroni α/2 and H-R7b on α/3 intervals |
| Our writes | `PositionReplacer` at CodeLlama layer 7, `h[p] ← ‖h[p]‖·unit(v)`, on the renamed-identifier token positions; vectors refitted under the chat template |
| CodeSteer | README-exact level-2 post-hoc steering, β_post 0.8, layers 24–31, plus the Eq. 10 calibrated head subset; effect gate live (4 096/4 096 attention rows changed, mean L1 shift 0.269) |
| Pre-registration | [`2026-09-14_steering-bakeoff-prereg.md`](../../log/nla-harness/2026-09-14_steering-bakeoff-prereg.md) (+ three amendments) · [`2026-09-14_better-vector-prereg.md`](../../log/nla-harness/2026-09-14_better-vector-prereg.md) · [`2026-09-15_chat-template-rerun-prereg.md`](../../log/nla-harness/2026-09-15_chat-template-rerun-prereg.md) (this table's rules, frozen before the runs) |
| Results entries | [`2026-09-15_chat-template-rerun-results.md`](../../log/nla-harness/2026-09-15_chat-template-rerun-results.md) (**this table**) · [`2026-09-15_bakeoff-results.md`](../../log/nla-harness/2026-09-15_bakeoff-results.md) · [`2026-09-15_better-vector-results.md`](../../log/nla-harness/2026-09-15_better-vector-results.md) · [`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md) · [`2026-09-15_pass1-decoding-correction.md`](../../log/nla-harness/2026-09-15_pass1-decoding-correction.md) · [`2026-09-15_cn-rescore-results.md`](../../log/nla-harness/2026-09-15_cn-rescore-results.md) · [`2026-09-15_parse-rate-correction.md`](../../log/nla-harness/2026-09-15_parse-rate-correction.md) |
| Kept numbers | [`nla/results/2026-09-15_ase_bakeoff/`](../../nla/results/2026-09-15_ase_bakeoff/) (both stats JSONs, the chat−raw comparison, ridge-fit report, dated config) |

---

## 1. The table — every method, one line

**`NLA` marks our arms.** **`NLA·write`** = a latent write through the NLA harness (`PositionReplacer`, CodeLlama layer 7, renamed-identifier token positions, β = 1); `NLA·prompt` = our text-only baselines, same harness, no write. Everything else is theirs or a reference. *No arm here uses a trained verbalizer/reconstructor pair — that arm (`ar_role`) is deferred, see §5.*

**accuracy** = share of the 434 boolean cases answered correctly, over **all 3 sampled runs** (the Chen et al. `c/n` estimator), case-weighted across programs; an unparsed reply is wrong. `Δ` is paired vs `unsteered`; `parse` = share of the 434 × 3 **cases** that received a readable T/F (per-case, against the packs); `acc|p` = accuracy among parsed cases, separating *answering more often* from *answering better*.

| # | method | type | what it does | accuracy | Δ [95 % CI] | parse | acc\|p | note |
|---|---|---|---|--:|---|--:|--:|---|
| 1 | `original_unsteered` | ref | un-renamed code, no intervention | **0.773** | +0.058 [−0.019, +0.140] | 0.89 | 0.87 | the renaming deficit, right sign at last |
| 2 | `ridge_map` | **NLA·write** | learned decoy→clean map, 98 held-out programs (λ 100, r 256) | **0.760** | **+0.046** [−0.027, +0.120] | 0.92 | 0.82 | best steered arm, CI still ∋ 0 |
| 3 | `swap_oracle` | **NLA·write** · ceiling | the item's *own* clean layer-7 state | 0.737 | +0.022 [−0.067, +0.108] | 0.88 | 0.84 | the ceiling — and it is this low |
| 4 | `role_proto` | **NLA·write** | clean prototype, same declared type + AST role | 0.728 | +0.014 [−0.057, +0.091] | 0.96 | 0.76 | highest parse rate of any arm |
| 5 | `prompt` | **NLA·prompt** | one-line "identifiers may mislead" warning | 0.727 | +0.013 [−0.055, +0.084] | 0.91 | 0.80 | the baseline CodeSteer has to beat |
| 6 | **`unsteered`** | ref | renamed code, no intervention | **0.714** | — | 0.90 | 0.79 | nothing clears it |
| 7 | `codesteer_auto` | CodeSteer | + Eq. 10 head calibration, top-4/layer | 0.712 | −0.002 [−0.088, +0.084] | 0.89 | 0.80 | **H-R7a comparator** |
| 8 | `erasure` | **NLA·write** | leave-one-out mean-difference vector | 0.709 | −0.005 [−0.092, +0.082] | 0.90 | 0.79 | training-free |
| 9 | `combined` | **NLA·write** | `erasure` + CodeSteer β 0.8 | 0.698 | −0.016 [−0.096, +0.068] | 0.88 | 0.80 | stacking buys nothing |
| 10 | `codesteer` | CodeSteer | slice prior, level-2 post-hoc, β 0.8, L24–31 | 0.697 | −0.017 [−0.106, +0.075] | 0.88 | 0.79 | README-exact |
| 11 | `uniform_prior` | ctrl | their machinery, uniform prior, β 0.8 | 0.696 | −0.018 [−0.079, +0.043] | 0.89 | 0.78 | prior is inert |
| 12 | `foreign` | **NLA·write** | another program's span state | 0.685 | −0.029 [−0.096, +0.042] | 0.88 | 0.78 | H-R7c comparator |
| 13 | `codesteer_beta0` | ctrl | CodeSteer, β_post = 0 (identity) | 0.677 | −0.038 [−0.109, +0.040] | 0.85 | 0.80 | the noise floor, gate passes (±0.05) |
| 14 | `rand_prior` | ctrl | their machinery, random prior, β 0.8 | 0.670 | −0.045 [−0.119, +0.027] | 0.83 | 0.81 | H-R7d |
| 15 | `prompt_types` | **NLA·prompt** | warning + every identifier's type and role, as text | 0.571 | −0.144 [−0.227, **−0.052**] | 0.77 | 0.74 | the only interval excluding zero — a harm |

<sub>Job ids, in row order: 403730 · 403731 · 403739 · 403732 · 403734 · 403729 · 403733 · 403741 · 403738 · 403735 · 403737 · 403740 · 405637 · 403736 · 403743. `rand_prior_beta0` was dropped (byte-identical to `codesteer_beta0`: one identity floor suffices).</sub>

## 2. Verdicts (rules frozen before the runs)

| hypothesis | contrast | result | verdict |
|---|---|---|---|
| **H-R7·damage** is there anything to restore? | `original` − `unsteered` | **+0.058**, 95 % [−0.019, +0.140] | **`DAMAGE-WEAK`** (predicted) — clears the +0.03 bar, CI includes 0, **so every verdict below is provisional** |
| **H-R7a** does CodeSteer restore it? | `codesteer` − `unsteered` · `codesteer_auto` − `unsteered`, α/2 | −0.017 [−0.115, +0.086] · −0.002 [−0.100, +0.095] | **`CODESTEER-INERT`** (predicted) |
| **H-R7b** match CodeSteer? | best of ours (`prompt`) − best of theirs (`codesteer_auto`), α/3 | +0.015, 95 % [−0.063, +0.097], α/3 [−0.079, +0.117] | **`MATCH-CODESTEER`** |
| **H-R7d** does the AST slice prior matter? | `codesteer` − `rand_prior` | +0.028 [−0.057, +0.112] | **`SLICE-IRRELEVANT`** |
| **H-R7c**/H-R6a learned map vs mean-difference | `ridge_map` − `erasure` | +0.052 [−0.021, +0.129] | **`MAP-NOT-BETTER`** |
| **H-R7c**/H-R6b does category meaning help? | `role_proto` − `foreign` (**H-R10's pre-registered primary**) | +0.043 [−0.022, +0.117] | **`CATEGORY-MEANING-INERT`** · **H-R10 ✗ not supported** (it was +0.078 on the raw corpus; it shrank) |
| **H-R7c**/H-R6c latent write vs the same facts as text | `ridge_map` − `prompt_types` | +0.190 [+0.105, +0.272] | registered **`LATENT-BEATS-PROMPT`** — but the baseline is degraded, see §3.6 |
| H-R6c, conservative | `ridge_map` − `prompt` (the intact prompting arm) | +0.033 [−0.036, +0.098] | **`PROMPT-SUFFICES`** — the reading carried into the ledger |
| H-R2c restoration ratio | (steered − unsteered)/(original − unsteered) | denominator +0.058 → ratios ±hundreds of % (`ridge_map` 79 % [−351, +506]) | **unreadable** — reported, not read |
| identity floor (no verdict) | `codesteer_beta0` − `unsteered` | −0.038, inside the frozen ±0.05 | **floor gate PASSES** — the intervals above are sampling noise around a real zero |

## 3. How to read it

1. **Nothing beats doing nothing, and now that means something.** No arm — theirs or ours — clears `unsteered` with a CI excluding zero. The difference from the first pass is that the comparison is now fair: CodeLlama sits at 0.714 renamed / 0.773 clean instead of 0.545/0.596 at chance, and the deficit CodeSteer exists to repair is present with the right sign (+0.058). Their method returns −0.017 (README-exact) and −0.002 (Eq. 10 calibrated), with the effect gate confirming 4 096 of 4 096 attention rows were actually modified. **This is real steering with no effect, not a mis-wired run.**
2. **The whole first-pass ranking was an artefact of their prompt.** Every one of the 15 arms gains between **+0.108 and +0.227** accuracy under the chat template, and per-case parse rises from 0.54–0.86 to 0.77–0.96 (§4). Two of the three headline readings from the raw-prompt table changed: `ridge_map` went from inert (−0.027) to the best steered arm (+0.046), and `prompt_types`' apparent collapse was mostly format damage. The methodological rule this buys: **anchor the no-op arm against an independent harness before scoring any contrast.** `unsteered` under the chat template inside their runtime is 0.7143 against 0.714 from our own H-R1 runner on the same 50 programs — a 0.000 difference, which also retires "maybe it is their top_k 7 sampler".
3. **Our best arm is a learned map, and it is the only one that never points the wrong way.** `ridge_map` (a ridge regression from decoy-identifier span states to clean span states, fitted on 98 *held-out* programs, held-out cosine 0.379 vs 0.175 for mean-only) is +0.046 [−0.027, +0.120] and tops the steered table on both prompt forms. It is not certified — the CI includes zero — and the honest next step is a replication at a fresh sampling seed (H-R13), not a claim.
4. **The oracle bounds the whole exercise.** `swap_oracle` writes each program's *own* clean layer-7 state at the renamed positions — the best any span-level write could do — and gets +0.022. When the ceiling is inside the noise, no non-oracle steering method on this corpus can be separated from zero. That is a statement about the corpus (50 programs, ±0.07–0.09 CI half-width, +0.058 of available damage), not about steering.
5. **CodeSteer's own knobs stay inert.** A random attention prior is within 0.028 of their AST/slice prior (`SLICE-IRRELEVANT`), a uniform prior scores like everything else, the Eq. 10 head calibration and the README's all-heads setting are within 0.015 of each other, and stacking our erasure vector onto CodeSteer buys nothing (−0.016).
6. **The one clear effect in the table is still a harm, but a smaller one than it looked.** `prompt_types` — every renamed identifier's correct declared type and AST role, as text — is −0.144 [−0.227, −0.052]. In the raw runtime it read −0.217 at parse 0.55; under the template it is −0.144 at parse 0.77, so roughly a third of the "collapse" was the prompt form. Its accuracy among answered cases (0.74) remains ordinary, so this is a formatting/verbosity effect, which is why the registered `LATENT-BEATS-PROMPT` margin against it is not the reading carried.

## 4. The runtime comparison — what the chat template changed (H-R7e)

Same 50 programs, same cases, same model, same seed, same scoring; the only difference is whether their `_build_prompt` output is wrapped in CodeLlama's chat template. The first pass (14 arms, jobs 399522–401537) is the "raw" column.

| arm | chat acc | raw acc | chat − raw [95 % CI] | chat parse | raw parse |
|---|--:|--:|---|--:|--:|
| `original_unsteered` | 0.773 | 0.545 | **+0.227** [+0.154, +0.300] | 0.89 | 0.72 |
| `prompt_types` | 0.571 | 0.379 | +0.192 [+0.102, +0.286] | 0.77 | 0.54 |
| `ridge_map` | 0.760 | 0.569 | +0.191 [+0.099, +0.288] | 0.92 | 0.78 |
| `prompt` | 0.727 | 0.541 | +0.186 [+0.096, +0.278] | 0.91 | 0.80 |
| `foreign` | 0.685 | 0.523 | +0.162 [+0.078, +0.250] | 0.88 | 0.78 |
| `combined` | 0.698 | 0.549 | +0.149 [+0.046, +0.257] | 0.88 | 0.82 |
| `codesteer` | 0.697 | 0.549 | +0.148 [+0.067, +0.230] | 0.88 | 0.79 |
| `erasure` | 0.709 | 0.567 | +0.142 [+0.050, +0.236] | 0.90 | 0.81 |
| `uniform_prior` | 0.696 | 0.561 | +0.134 [+0.053, +0.220] | 0.89 | 0.79 |
| `codesteer_auto` | 0.712 | 0.581 | +0.131 [+0.061, +0.203] | 0.89 | 0.80 |
| `role_proto` | 0.728 | 0.601 | +0.127 [+0.062, +0.198] | 0.96 | 0.84 |
| `unsteered` | 0.714 | 0.596 | +0.118 [+0.023, +0.212] | 0.90 | 0.83 |
| `rand_prior` | 0.670 | 0.557 | +0.113 [+0.021, +0.203] | 0.83 | 0.79 |
| `codesteer_beta0` | 0.677 | 0.569 | +0.108 [+0.029, +0.189] | 0.85 | 0.77 |
| `swap_oracle` | 0.737 | 0.674 | +0.062 [−0.029, +0.154] | 0.88 | 0.86 |

**Why it matters for the paper's claim, not just for us.** The paper's `run_llama` feeds CodeLlama-*Instruct* the raw prompt with no chat template. That held clean-code accuracy at 0.545 where chance is 0.500, and at that floor renaming had nothing left to break — the *renamed* condition scored **higher** (−0.051 damage, the wrong sign), which made the restoration denominator meaningless. Under the template the deficit is +0.058 and every arm is 11–23 points higher. `swap_oracle` gains least (+0.062, CI ∋ 0), exactly as expected for the one arm that was already writing the clean state. The paper's headline damage (76.5 → 40.2) is on a model family we may not run; on the permitted models it is small under a chat template (H-R1: Llama-3.1-8B +0.03, CodeGemma-7B +0.08, CodeLlama-7B +0.11) and absent in their raw-prompt runtime. So the first pass was a **failed precondition** of the CodeSteer claim, and this pass is the fair test — in which the claim does not reproduce on this model.

## 5. Methods — what each arm actually does

**Task and stimuli.** Each item is a HumanEval-X Java program, adversarially renamed by the ASE-2026 renamer so that identifiers suggest *plausible but wrong* semantics (the L1b tier of this programme's obfuscation ladder), paired with boolean claims about what the code does. 50 programs, 434 cases, 8.7 per program, balanced True/False by construction — **chance is exactly 0.500**. Functional equivalence between renamed and original is execution-validated upstream, which is what licenses scoring both conditions against one label set.

**Model.** `CodeLlama-7b-Instruct`. Of the permitted models it is the only one where the paper's own protocol showed a renaming penalty worth recovering (H-R1: +0.113 under a chat template, vs +0.030 Llama-3.1-8B, +0.076 CodeGemma-7B), and it is a Llama architecture so `install_llama_steering` applies unmodified.

**Runtime.** Every arm — theirs and ours — runs through their `SteeredCausalLM`, so prompt assembly, the sampling loop, decoding and the strict-JSON parser are bit-for-bit shared and the only thing that varies is the intervention. The unsteered control is the same object with `steering_config = None`. The single change in this pass is the chat template (§4), installed as the *instance's* `_build_prompt` so their generation, their prior's token alignment, their Eq. 10 calibration and our residual alignment all see one templated string; their code is untouched.

**Theirs, and its controls.**
- `codesteer` — README-exact level-2 post-hoc attention steering: hybrid AST-slice prior (`SlicingHybridPrior(SlicingPrior(ASTPrior))`, javalang only — the Joern prior is never touched), 12 equal-count bins, β_post 0.8, all heads of the last 8 layers (24–31), decode-time only.
- `codesteer_auto` — the same with the paper's Eq. 10 calibration selecting a sparse head subset (top-4 heads/layer). The two readings of the paper disagree, so both are run and **the better of the two is the comparator** (conservative for us).
- `rand_prior` / `uniform_prior` — their machinery with the prior replaced by a random prior of matched mass, and by a flat one: does the *content* of the prior do any work, or only its magnitude?
- `codesteer_beta0` — their machinery at β_post = 0, where `level2_post` returns the attention unchanged. It measures the split-prefill code path plus sampling variance, enters no verdict, and is the floor every interval is read against.

**Ours — latent writes.** All written with `PositionReplacer` at the output of layer 7, at the renamed-identifier token positions (`ase_align_java.align` on character spans, BOS offset measured per prompt, never assumed), as a norm-preserving direction substitution `h[p] ← ‖h[p]‖·unit(v)`.
- `swap_oracle` — the program's *own* clean state (per token where the original and decoy spans tokenise to equal length, else the original span mean at every decoy token). Not deployable; it bounds what any span-position write can achieve.
- `erasure` — training-free: the leave-one-out mean clean-minus-decoy difference over every span of every *other* program in the run, with the LOO magnitude. Nothing from the written item's original enters its vector.
- `ridge_map` — reduced-rank ridge regression of the delta on the centred decoy state (`δ̂ = (h1b − μ)A + c`, `v = h1b + δ̂`; `A = 0` recovers `erasure`), so the write is decoy-state-dependent rather than one global direction. λ and rank chosen by **grouped 5-fold CV on the fit pool only** (groups = programs), criterion held-out cosine against the true delta; selected λ 100, rank 256, held-out cos **0.3791** vs 0.1751 mean-only.
- `role_proto` — the mean clean state of pool spans sharing the identifier's declared type **and** AST role, falling back to type then to method-vs-variable; the resolved level is recorded per span (this fit: 1 214 / 64 / 15). Installs a *category* meaning, and never sees the true name.
- `foreign` — a different program's span state at the same positions: the specificity control. A write that helps must help more than a wrong write of the same shape.
- `combined` — `erasure` at prefill plus `codesteer` at decode, two different channels, read for super-/sub-additivity.

**Ours — text baselines (mandatory, not optional).** CLAUDE.md §4 requires a prompting/dense baseline with every steering claim, because prompting has repeatedly beaten latent steering in the 2025–26 literature. Both run through the identical harness with no write: `prompt` prepends one sentence ("the identifiers in this code may be misleading; reason about what the code actually does"), and `prompt_types` adds one line per renamed identifier with its declared type and inferred role — the same information `role_proto` installs latently, as text, with no true names.

**Fit/test separation.** The fit pool is every aligned program in the release *except* the 50 under test: 98 programs, 3 506 identifier spans, captured under the chat template (1 293 test spans). Nothing from a test program enters any vector written into it, `swap_oracle` excepted and labelled accordingly. The fit is configured by a version-controlled YAML frozen at pre-registration; a post-capture change becomes a new dated config, not an edit.

**Scoring.** `c/n`: the share of the 434 cases correct across **all three** sampled runs, case-weighted across programs as Σ accᵢ·nᵢ / Σ nᵢ, unparsed = wrong. The paper's `pass@1` reads run 1 only and discards two thirds of the evidence; it is ≈ √3 noisier and it ranked the arms differently (§4 of the superseded table). Parse rate is reported per **case** against the packs — the runner's own `parsed_frac` counts prediction keys absent from the truth set — and `acc|p` separates answering more often from answering better.

**Statistics.** Every contrast is paired per program and bootstrapped over programs (cluster bootstrap, N_BOOT 10 000, seed 20260724), recomputing the case-weighted difference each draw; no analysis pools cases as if independent. Where a pre-registration fixed several comparisons the verdict reads on a Bonferroni-adjusted interval (α/2 for H-R7a, α/3 for H-R7b). Thresholds — including the 0.05 equivalence band behind every `MATCH`/`INERT` word — were written down before the runs and not retuned after.

**Gates (a failure exits 3 and nothing is reportable).** Prompt-ids gate: the templated string must tokenize to exactly `apply_chat_template(..., tokenize=True)` on all 50 prompts. Write self-gate: writing a program's own exact per-token state must leave the last-position next-token logits within 0.05 of the unhooked forward and must report the expected `n_positions_written` (all arms: worst 0.0000). Attention effect gate, **two-sided**: real steering arms abort on zero effective level-2 operations (logged 4 096/4 096, mean L1 shift 0.269), the identity floor aborts on *any* (logged zero). Ridge margin gate: the learned map must beat mean-only on held-out cosine by ≥ 0.05 or its GPU arm is not run. Floor gate: the identity arm must land within ±0.05 of `unsteered` or the intervals are not read as sampling noise.

## 6. What would change the answer

- **H-R12 — power, before any more GPU (no GPU to compute).** Every verdict above is provisional because the damage CI includes 0. At the observed per-program variance, separating a +0.05 effect needs roughly **150–200 programs** (3–4× this corpus). Compute that from these 50 programs' bootstrap distribution and pre-register the smallest detectable effect *before* extending the corpus. Power, not method, is now the binding constraint.
- **H-R13 — replicate `ridge_map` at a fresh sampling seed (~1.5 GPU-h).** It is the only arm whose point estimate never goes the wrong way and whose held-out fit is prompt-form-invariant (cos 0.379 vs 0.38). `RIDGE-REAL` if Δ vs `unsteered` stays ≥ +0.03; `RIDGE-NOISE` otherwise. Cheap, and it decides whether there is anything here to build on.
- **H-R11 — does attention fixation fire on CodeLlama at all?** CodeSteer's prior encodes a fixation mechanism. An inert steering method on a model whose attention never fixates is *expected*, not a failed replication. This is Instrument-1 territory and it is now the more diagnostic question than any further steering arm.
- **H-R5 — the heads.** Unblocked but weakly motivated: there is no certified positive effect whose head support is worth localising. Park behind H-R12/H-R13.
- **Deferred, still not justified:** `swap_guess` (states of a self-de-obfuscated prompt) and `ar_role` (an NLA AR trained on CodeLlama-7B at layer 7, ~1 day port + ~20 GPU-h) — no latent headroom over prompting has been demonstrated on this host and stimulus set.

## 7. Provenance

All arms: SLURM partition `h200,h100`, one GPU each, `HF_HUB_OFFLINE=1`, `PYTHONHASHSEED=0`, three `--dependency=singleton` lanes (≤ 3 GPUs); job ids in §1, nodes and UTC timestamps in the results entry. Runner `nla/src/ase_steer_run.py` (arms 1–14 sha `f99961e3…`; the identity-floor arm `cad05d86…`, a diff touching only arm dispatch and the effect gate). Chat template: `install_chat_template` replaces `_build_prompt` on the instance with `tok.add_bos_token = False` and the literal `<s>` kept, gated by `tok(text)["input_ids"] == apply_chat_template(..., tokenize=True, return_dict=False)` on all 50 prompts (1 293 identifier spans aligned, BOS offset 0). Vectors refitted under the template: `nla/src/ase_vectors.py`, config `nla/configs/ase_vectors_chat.yaml`, pool 98 programs / 3 506 spans, λ 100 rank 256, held-out cos 0.3791 vs mean-only 0.1751 (margin gate PASS), prototypes `{type_role: 1214, type: 64, kind: 15}`, seed 20260724. Role tagger `nla/src/ase_roles.py` (javalang). Scorer `nla/src/ase_bakeoff_stats.py` sha `383fe716…` (`--scoring cn --packs <renamed> <original>`; it needs the packs because the per-arm jsonl carries predictions but no truth labels), identity-gated against the banked raw-prompt JSON before use; comparison table `nla/src/ase_chat_vs_raw.py` sha `249f9990…`, self-tested to exact zeros on raw-vs-raw. Gates live in every run: residual self-gate (worst max|Δlogit| 0.0000, tol 0.05), attention effect gate (two-sided — real arms fail on zero effective level-2 calls, the identity floor fails on *any*), ridge margin gate, chat-template ids gate.

**Faults on record** (all pre-date or are orthogonal to the numbers above; each has a dated entry). In the runs: the first CodeSteer arms ran at β_post = 0 ([`2026-09-14_bakeoff-beta-fault.md`](../../log/nla-harness/2026-09-14_bakeoff-beta-fault.md); those runs became the noise floor), the ridge fit crashed once on a YAML float ([`2026-09-15_pool-fit-gate.md`](../../log/nla-harness/2026-09-15_pool-fit-gate.md)), and in this pass the identity-floor arm died twice before running — once on an arm name that had only ever existed as a consequence of the β fault, once because the effect gate treated "changed nothing" as fatal on the one arm whose job is to change nothing (gate now two-sided; it passed with 0/4 096 effective calls on all 50 programs, which independently confirms the raw-prompt floor's label). In the *reporting*: the decoder was described as greedy when the arms sampled at T 0.7 ([`2026-09-15_pass1-decoding-correction.md`](../../log/nla-harness/2026-09-15_pass1-decoding-correction.md)), which prompted the `c/n` re-score ([`2026-09-15_cn-rescore-results.md`](../../log/nla-harness/2026-09-15_cn-rescore-results.md)); and the `parse` column was inflated by the runner's `parsed_frac = len(pred)/n_cases`, which counts prediction keys absent from the truth set ([`2026-09-15_parse-rate-correction.md`](../../log/nla-harness/2026-09-15_parse-rate-correction.md)) — it now reports per-case rates computed against the packs, in both tables.
