# 2026-09-15 · CORRECTION — the `parse` column was inflated by phantom keys, and the "why so bad" answer is the runtime

**Thread:** nla-harness · **Experiment:** H-R2 / H-R6 / H-R9 reporting (no GPU) · **Status:** correction, append-only
**Corrects:** the `parse` / `acc/parse` columns and observation 4 ("`role_proto`'s win is entirely compliance") of
[`2026-09-15_cn-rescore-results.md`](2026-09-15_cn-rescore-results.md), the same columns in
[`2026-09-15_better-vector-results.md`](2026-09-15_better-vector-results.md) and
[`2026-09-15_bakeoff-results.md`](2026-09-15_bakeoff-results.md), and §3.3 of
[`../../reports/2026-09-15_ase-bakeoff/REPORT.md`](../../reports/2026-09-15_ase-bakeoff/REPORT.md). Those entries
stand as written; every **accuracy** number and every verdict in them is unaffected (accuracy never used
`parsed_frac`). Prompted by the user's question *"Why are these so bad? These results seem weird and would
contradict the codesteer paper on attention reallocation"*, which this entry answers with banked data.

## Hypotheses (stated before re-reading the rows)

- **Parse.** The runner's `parsed_frac` (`ase_steer_run.py`: `n_parsed = len(pred)`, divided by `len(cases)`)
  counts *every* key in the model's JSON, including case ids that are not in the pack. Prediction: per-case parse
  (`pred.get(case_id) is not None` over the pack's own ids) is **lower** for every arm, and most so for the
  writing arms, whose replies were observed to hallucinate ids.
- **Why so bad.** The bake-off accuracies (0.38–0.67, `original_unsteered` 0.545) are low because the paper's
  `run_llama` feeds CodeLlama-*Instruct* the raw prompt with no chat template. Prediction: the H-R1 runs of the
  **same 50 programs** under our chat-template harness (`gate_codellama7b/`), re-scored `c/n` against the same
  packs, are ≥ 0.15 higher on clean code and show the renaming damage that the bake-off does not.

## Setup (no GPU, nothing re-generated)

- Truth from `gate/packs_ren_subset.jsonl` (434 cases, 8.7 per program, **True-rate 0.500 exactly** → chance 0.5).
- Per arm, over 50 × 3 runs × 434 cases: `acc = mean(pred.get(c) == truth[c])`, `parse_pc = mean(pred.get(c) is
  not None)`, `acc|p = mean(pred[c] == truth[c] | parsed)`, `T-rate|p = mean(pred[c] is True | parsed)`,
  `phantom/run = |keys(pred) − keys(truth)|`. The old `parsed_frac` is averaged straight from the jsonl.
- Same scoring applied to `gate_codellama7b/original.jsonl` and `renamed_unsteered.jsonl` (H-R1, job in
  [`2026-09-14_ase-replication-results.md`](2026-09-14_ase-replication-results.md); same 50 programs, same
  packs, same seed 20260724, same T 0.7 / 3 runs; the only difference is the chat template).

## Results

### Per-case parse vs the runner's `parsed_frac` (all 16 bake-off arms)

| arm | accuracy | parse (per-case) | `parsed_frac` (old) | inflation | acc\|p | T-rate\|p | phantom ids / reply |
|---|--:|--:|--:|--:|--:|--:|--:|
| `swap_oracle` | 0.674 | 0.857 | 0.883 | +0.03 | 0.787 | 0.500 | 0.20 |
| `role_proto` | 0.601 | **0.843** | **0.998** | **+0.16** | **0.713** | 0.496 | 0.49 |
| `unsteered` | 0.596 | 0.828 | 0.902 | +0.07 | 0.720 | 0.526 | 0.47 |
| `codesteer_auto` | 0.581 | 0.797 | 0.863 | +0.07 | 0.728 | 0.496 | 0.36 |
| `codesteer_beta0` | 0.569 | 0.775 | 0.817 | +0.04 | 0.734 | 0.521 | 0.27 |
| `rand_prior_beta0` | 0.569 | 0.775 | 0.817 | +0.04 | 0.734 | 0.521 | 0.27 |
| `ridge_map` | 0.569 | 0.783 | 0.939 | +0.16 | 0.726 | 0.514 | 0.42 |
| `erasure` | 0.567 | 0.808 | 0.829 | +0.02 | 0.702 | 0.520 | 0.32 |
| `uniform_prior` | 0.561 | 0.790 | 0.815 | +0.03 | 0.710 | 0.503 | 0.30 |
| `rand_prior` | 0.557 | 0.790 | 0.861 | +0.07 | 0.705 | 0.528 | 0.50 |
| `codesteer` | 0.549 | 0.791 | 0.838 | +0.05 | 0.694 | 0.531 | 0.32 |
| `combined` | 0.549 | 0.822 | 0.868 | +0.05 | 0.668 | 0.510 | 0.30 |
| `original_unsteered` | 0.545 | 0.715 | 0.760 | +0.05 | 0.763 | 0.523 | 0.39 |
| `prompt` | 0.541 | 0.805 | 0.836 | +0.03 | 0.673 | 0.554 | 0.21 |
| `foreign` | 0.523 | 0.783 | 0.962 | +0.18 | 0.668 | 0.517 | 0.54 |
| `prompt_types` | 0.379 | 0.545 | 0.633 | +0.09 | 0.695 | 0.532 | 0.55 |

### Same 50 programs, same packs, same model — their runtime vs ours

| harness | condition | accuracy (c/n) | Pass@1 (run 1) | parse (per-case) | acc\|p |
|---|---|--:|--:|--:|--:|
| **ours, chat template** (H-R1, `gate_codellama7b/`) | original | **0.780** | 0.800 | 0.919 | 0.849 |
| ours, chat template | renamed | **0.714** | 0.687 | 0.897 | 0.796 |
| **theirs, raw prompt** (bake-off) | original | **0.545** | 0.558 | 0.715 | 0.763 |
| theirs, raw prompt | renamed | **0.596** | 0.532 | 0.828 | 0.720 |

Renaming damage: **−0.066** under the chat template, **+0.051** (wrong sign) in their runtime. Clean-code cost of
dropping the chat template: **−0.235**. Chance on this task: **0.500**.

## Hypothesis verdict

- **Parse ✓.** Per-case parse is lower for all 16 arms (+0.02 to +0.18 inflation), most for the writing arms
  `foreign` (+0.18), `role_proto` (+0.16), `ridge_map` (+0.16). The "`role_proto` answers 99.8 % of the time at
  `acc/parse` 0.60" reading is **withdrawn**: per-case it is 0.843 / 0.713 against `unsteered` 0.828 / 0.720 — a
  wash on both axes, exactly as its +0.005 accuracy Δ says. No arm's accuracy-among-parsed leaves 0.67–0.79 and
  no arm's True-rate leaves 0.50–0.55, so neither compliance nor label bias separates the arms.
- **Why so bad ✓.** The chat-template runs of the identical items score 0.780 clean, and renaming costs 0.066
  there; their raw-prompt runtime sits at 0.545 clean — 4.5 points above chance — and the renaming "damage" has
  the wrong sign. The bake-off is being run on a host that their harness has already pushed to the floor.

## Observations

1. **This does not contradict the CodeSteer paper; it fails its precondition.** CodeSteer's claim is that
   re-allocating attention *restores* accuracy that renaming removed (Qwen2.5-7B 76.5 → 40.2 → 78.3). On the
   only model families we may run there is little to restore under a chat template (H-R1: +0.03 / +0.08 / +0.11)
   and nothing at all in their own runtime. A steering method cannot be ranked on a −0.051 deficit. The paper's
   models (Qwen2.5-7B/14B, DeepSeek-Coder-6.7B, DeepSeek-V2-Lite) are all excluded by the standing rule.
2. **Their runtime, not their method, is the problem for CodeLlama.** `run_llama` tokenises the raw
   `build_counterfactual_instruction` text; CodeLlama-Instruct without `[INST]…[/INST]` loses 24 points on clean
   code and 21 points of parse. The paper's own models may tolerate the raw prompt; CodeLlama-7B-Instruct does not.
3. **Sampling at T 0.7 on a chance-0.5 binary task** is the other half of the noise: single-run Pass@1 CIs were
   ±0.145 on 434 cases, and the ranking flipped between run 1 and the 3-run mean (H-R9).
4. **Where the earlier narrative went wrong, mechanically.** `n_parsed = len(pred)` in `ase_steer_run.py:` the
   parser accepts any JSON object, and a steered reply that invents case ids (`role_proto` 0.49 per reply,
   `foreign` 0.54) is credited for them. The fix is scoring-side (this entry and the report use per-case rates);
   the runner is left untouched so the banked jsonl stay byte-identical to their job ids.

## New questions / hypotheses

- **H-R7 (re-scoped).** The damage-bearing screen should be run **under the chat template**, where CodeLlama-7B
  shows −0.066 on these 50 (and +0.113 on the full 1 930-case set): re-run the six informative arms
  (`unsteered`, `original_unsteered`, `codesteer_auto`, `prompt`, `role_proto`, `swap_oracle`) through
  `SteeredCausalLM` with the chat template applied at tokenisation — everything else (seed, T, packs, parser,
  3 runs, `c/n`) unchanged. Pre-register: `CODESTEER-RESTORES` if `codesteer_auto − unsteered` ≥ +0.03 with CI
  excluding 0 on the chat-template run. This is the cheapest test that keeps the paper's machinery and gives it a
  deficit to work on; ~6 GPU-h.
- **H-R11 (Instrument-1 territory).** Does the paper's *attention fixation* diagnostic (`Level2Effect` /
  fixation score in their artifact) even fire on CodeLlama-7B under renaming? If fixation is absent, there is no
  attention to re-allocate and the null is expected on their own theory. Zero generation cost: one forward pass
  per condition per snippet.
- **H-R8** is withdrawn as posed (no compliance gap to explain); H-R10 stays blocked on H-R7.
