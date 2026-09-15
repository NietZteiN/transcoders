# 2026-09-15 · correction: the bake-off arms sampled at T 0.7, they were not greedy

**Thread:** nla-harness · **Type:** correction (no new run, no new number)
**Corrects:** [`2026-09-15_bakeoff-results.md`](2026-09-15_bakeoff-results.md) and
[`2026-09-15_better-vector-results.md`](2026-09-15_better-vector-results.md), whose setup lines describe the
decoder as "raw prompt, greedy, 3 runs × cases per snippet". Those entries are append-only and stand as written;
this entry is the correction of record.

## What is wrong

"Greedy" is wrong. `nla/src/ase_steer_run.py:414` calls the paper's runtime with `do_sample=True`, and
`ase_steer_run.py:71-72` pins `SEED = 20260724`, `TEMP, TOP_P = 0.7, 1.0` — the ASE-2026 runtime's own defaults
(`models.py:58 DEFAULT_TEMP = 0.7`). `torch.manual_seed(SEED)` is set once per arm
(`ase_steer_run.py:319`). So every arm decoded by **sampling at T 0.7, top_p 1.0**, three runs per program.

## Why it does not move any number

Nothing about the runs changes — the description was wrong, not the execution, and every arm (theirs and ours)
went through the same sampled decoder, so the paired contrasts and all verdicts in the two entries above stand
unchanged. The mislabel matters only for how the metric reads:

- **`pass@1` is the accuracy of ONE sampled reply**, not of a deterministic one. Per program,
  `pass@k` = share of its boolean cases that *any* of the first k runs got right
  (`ase_steer_run.py:438-440`), so `pass@1` = "run 1 got this case right".
- It is therefore **not** the Chen et al. `c/n` estimator: three runs exist per program, but the headline
  consumes only the first. Sampling noise in the headline is larger than it would be if all three were averaged —
  which is part of why the β = 0 identity arms drift +0.067 from `unsteered` (the ±0.14 noise floor).
- `pass@2` / `pass@3` are in the jsonl and are **not** accuracy: "any of k correct" is a ceiling, and it rises
  with k for a *worse* model too. They were never used for a verdict and must not be.

## Consequence for the reporting

`reports/2026-09-15_ase-bakeoff/REPORT.md` is corrected in place (living doc): the Runtime row now states the
sampling defaults, and the Metric row spells out the `any-of-first-k` definition, the case weighting, the
departure from `c/n`, and the warning about `pass@2`/`pass@3`.

## New question raised

**H-R9 (cheap, no new GPU).** The three runs per program are already banked for all 16 arms. Re-score every arm
with `c/n` over the 3 runs (the unbiased single-sample estimator) instead of run 1 alone, and re-run the same
frozen contrasts. Prediction: every verdict survives (the arms are ranked well inside the noise floor either
way), but the noise floor itself should shrink by roughly √3 — which would tighten the `swap_oracle` interval
and give H-R7's damage screen a better-powered readout for free. Rule to freeze before looking: the H-R2a/H-R6
verdicts are re-read on the `c/n` scoring only if the β = 0 identity arms come within ±0.05 of `unsteered`; if
they do not, the floor is not sampling noise and the re-score is descriptive only.
