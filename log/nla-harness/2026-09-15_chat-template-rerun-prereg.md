# 2026-09-15 · PRE-REGISTRATION H-R7 — the bake-off re-run in a runtime where the damage exists (chat template)

**Thread:** nla-harness · **Experiment:** H-R7 (ASE) · **Status:** pre-registered, rules frozen before submission
**User request (verbatim):** *"Ok well need to rerun to get accurate results then change the setup to do that
something comparable"* — after [`2026-09-15_parse-rate-correction.md`](2026-09-15_parse-rate-correction.md) showed
the bake-off floor is their raw-prompt runtime, not the steering.

## What changes, and what does not

**One change:** the prompt that reaches their `SteeredCausalLM` is wrapped in CodeLlama-Instruct's chat template
(`[INST] … [/INST]`), exactly the ids our H-R1 harness fed the same model (`ase_tf_run.py`:
`apply_chat_template([{"role":"user","content": prompt}], add_generation_prompt=True)`). Implemented as a
`--chat-template` flag on `ase_steer_run.py` that replaces the instance's `_build_prompt` with the same string
run through `apply_chat_template(tokenize=False)`, so **every** consumer of the prompt inside their runtime —
generation, the AST/slice prior's token alignment, Eq. 10 head calibration, and our residual-write alignment —
sees the one templated string. Their code is not edited.

**Nothing else changes:** same 50 programs, same 434 cases, same packs, same `SEED 20260724`, same sampler
(their `do_sample=True`, T 0.7, top_p 1.0, top_k 7, 3 runs per program), same parser (`parse_predicted_labels`,
strict JSON), same steering configs (`PAPER_CFG`, β_post 0.8, last 8 layers; `codesteer_auto` top-4/layer),
same residual writes (layer 7, β 1, same fitted vectors `vectors_codellama7b_L7.pt`), same scorer
(`ase_bakeoff_stats.py --scoring cn`, every frozen contrast and threshold of H-R2/H-R6/H-R9 verbatim).

**Vectors refitted, not reused.** `ridge_map` / `role_proto` write layer-7 states captured from the *raw*
prompt (`pool_codellama7b_L7.pt`); an `[INST]`-wrapped prompt has different layer-7 states, so those vectors
would be off-distribution in the re-run. `nla/configs/ase_vectors_chat.yaml` (a dated copy of the H-R6 config:
`chat_template: true` and three output paths, nothing else) re-captures the pool under the template (0.4 min
GPU) and re-fits with the same λ/rank grid, CV and 0.05 cosine gate; `ridge_map` runs only if that gate passes
again, exactly as in H-R6.

**Arms (15):** all of the bake-off except `rand_prior_beta0`, which was byte-identical to `codesteer_beta0`
(both are β_post = 0 identity through the same code path; one identity floor suffices).

## Identity gates (any failure → exit 3, `R7-HARNESS-FAULT`, nothing reportable)

1. **Template-ids gate, every prompt:** `tokenizer(templated_text)["input_ids"]` must equal
   `apply_chat_template(messages, tokenize=True, add_generation_prompt=True)` exactly — the ids their runtime
   generates from are the ids H-R1 generated from. (The template text starts with `<s>`, which the tokenizer
   would add again; the wrapper strips it and the gate proves the result is right rather than assuming it.)
2. **Residual self-gate** (existing): writing the renamed prompt's own layer-7 state leaves last-position logits
   within `--self-tol`; `n_positions_written` = aligned decoy tokens.
3. **Effect gate** (existing): a steered arm with zero effective level-2 calls on any snippet aborts.
4. **Scorer identity:** the patched `ase_bakeoff_stats.py` (sha `383fe716…`) reproduces `2026-09-15_bakeoff_stats_cn.json`
   on the old directory modulo the new `H_R7` key — **checked before submission: PASS** (on the raw-prompt
   directory the new block reads `DAMAGE-ABSENT` −0.0507 [−0.1333, +0.0313] and `CODESTEER-INERT`, as §3 of the report says in words).

## Hypotheses and frozen rules (cluster bootstrap over programs, N_BOOT 10 000, seed 20260724, case-weighted `c/n`)

- **H-R7·damage (readability gate, evaluated first).** `original_unsteered − unsteered`, paired.
  **`DAMAGE-PRESENT`** if > 0 with the 95 % CI excluding 0 · **`DAMAGE-WEAK`** if the point estimate ≥ +0.03
  but the CI includes 0 · **`DAMAGE-ABSENT`** otherwise. Restoration verdicts (below) are read only under
  `DAMAGE-PRESENT`; under `DAMAGE-WEAK` they are reported as provisional; under `DAMAGE-ABSENT` every arm is
  reported descriptively and the bake-off is declared unrunnable on this model in *any* runtime.
  *Prediction:* `DAMAGE-WEAK` (H-R1 on these 50 gave −0.066 `c/n`; the full 1 930-case set gave +0.113
  [+0.007, +0.220], so a 434-case CI of ≈ ±0.08 is marginal).
- **H-R7a (their claim, on a deficit that exists).** `codesteer − unsteered` and `codesteer_auto − unsteered`,
  Bonferroni α/2 (97.5 % CIs). **`CODESTEER-RESTORES`** if either ≥ +0.03 with its adjusted CI excluding 0 ·
  **`CODESTEER-INERT`** if both |Δ| < 0.05 with CIs containing 0 · **`CODESTEER-HARMS`** if either < 0 with
  the adjusted CI excluding 0. Restoration ratio `(steered − unsteered)/(original − unsteered)` reported with CI
  for every arm, as registered in H-R2c. *Prediction:* `CODESTEER-INERT` — I expect the attention prior to move
  little on a 7B Llama whose fixation was never shown; the paper's effect is on Qwen.
- **H-R7b (ours vs theirs) = H-R2a verbatim:** best-of-ours (`erasure`, `prompt`; α/3) − better-of-theirs;
  `SURPASS` / `MATCH` (|Δ| < 0.05) / `BELOW`. **H-R7c = H-R6a/b/c verbatim** (`ridge_map − erasure`,
  `role_proto − foreign`, best latent − `prompt_types`, R6_MIN 0.05). **H-R7d = H-R2b verbatim**
  (`codesteer − rand_prior`, `SLICE-MATTERS` / `SLICE-IRRELEVANT`). The identity floor (`codesteer_beta0 −
  unsteered`) must be within ±0.05 (H-R9's gate) for any of these to be read.
- **H-R7e (the runtime itself, descriptive but the reason for the run).** For each arm, `chat − raw` paired on
  the same programs; and `unsteered_chat` vs H-R1's `renamed_unsteered` (same ids and 512 new tokens; HF's
  default top_k 50 in ours vs their top_k 7) — if these differ by > 0.05 the two runtimes are still not the
  same instrument and the difference is reported as a fault to chase, not a finding.

## Setup

- `sbatch nla/scripts/nla_r7_pool_sbatch.sh` (capture + fit), then `sbatch nla/scripts/nla_r7_arm_sbatch.sh <arm>` → `/scratch/…/ase2026/bakeoff_codellama7b_chat/<arm>.jsonl`;
  `--partition=h200,h100`, one GPU, `HF_HUB_OFFLINE=1`, `PYTHONHASHSEED=0`. Three lanes (`--job-name=r7_lane{1,2,3} --dependency=singleton`, vector arms also `afterok:<pool job>`)
  so at most three of our jobs run at once.
- Cost from the raw-prompt runs: 4–22 min per arm, `codesteer_auto` 61 min ⇒ **≈ 3 GPU-h** for 15 arms.
- Stats: `ase_bakeoff_stats.py --dir …/bakeoff_codellama7b_chat --original …/bakeoff_codellama7b_chat/original_unsteered.jsonl --scoring cn --packs …` → kept as `nla/results/2026-09-15_ase_bakeoff/2026-09-15_bakeoff_stats_chat_cn.json`.

## Results / verdict / observations

To be filled by the results entry. Nothing here is a result.
