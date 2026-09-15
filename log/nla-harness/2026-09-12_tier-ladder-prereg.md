# 2026-09-12 · PRE-REGISTRATION — the prompt-space tier ladder on the host (H-A1 / H-A2 / H-A3)

**Thread:** nla-harness · **Experiment:** `A_tier_ladder`, Gemma-3-4B-it · **Status:** pre-registered, nothing run.

## Why this exists

Every steering result in this family is a repair applied *in activation space* to an L1b prompt. The ladder
itself — what the same host scores when the decoy is simply **absent from the prompt** — has never been
measured on this host beyond L0 and L1b. That number is the ceiling for any *erasure* lever (NLA edit,
cross-item erasure vector, attention masking, attention reallocation alike), because **L1 is perfect erasure
done in token space**: nonsense names, original structure, no trap. Likewise L2/L3 size the second failure
route (control-flow flattening) on this host before Step 4 (dispatcher state reads) is designed.

This is the *prompt-space* sibling of the activation-space decomposition already banked
([`2026-09-07_repaired-rerun-results.md`](2026-09-07_repaired-rerun-results.md): clean-state transport splits into
**meaning ≈ 51 %, decoy removal ≈ 49 %, structure ≈ 1 %** of the logprob effect). The question here is what those
shares are worth in **accuracy**, which the logprob readout cannot say.

## Hypotheses (falsifiable, frozen before the run)

Readouts per tier T ∈ {L0, L1, L1b, L2, L3}: greedy accuracy `acc_T` (n = 1, comparable with every banked
number) and per-item pass rate `rate_T` (n = 8 samples, temperature 0.8 / top-p 0.95 as in
[`2026-09-12_accuracy-prereg.md`](2026-09-12_accuracy-prereg.md)). Paired contrasts are **per-item differences
in `rate_T`** over the same 60 items, cluster bootstrap N_BOOT 10 000, seed 20260724.

- **H-A1 — erasure headroom.** `rate_L1 − rate_L1b` is the most any erasure lever can deliver;
  `rate_L0 − rate_L1` is what only *reconstruction* (installing the true meaning) can add. Both are reported as
  the bounds they are. Prediction: **both small on this host** — `acc_l0 0.567 → acc_l1b 0.533` leaves 2/60 net,
  so the two shares together are ≤ ~0.03 greedy and the rate readout is where any separation would show.
  Verdict words: **`ERASURE-HEADROOM`** if `rate_L1 − rate_L1b` > 0 with CI excluding 0; else
  **`ERASURE-FLOOR`** (this host has nothing for an erasure lever to recover in accuracy, whatever it does to
  logprob — which would bound every arm of job 391968 from above).
- **H-A2 — flattening penalty.** Paper 3's **OR = 0.57** for L2 predicts `0.567 → ~0.43` here, i.e. a paired
  `rate_L0 − rate_L2` ≈ **+0.13**. **`FLATTENING-PENALTY`** if `rate_L0 − rate_L2` > 0 with CI excluding 0;
  **`FLATTENING-FLAT`** otherwise. Prediction: PENALTY. If FLAT, the L2/L3 route is not expressed on this host
  and Step 4 must move to a host where it is.
- **H-A3 — do the routes compound?** **`ROUTES-COMPOUND`** if `rate_L3 − rate_L2` < 0 with CI excluding 0
  (the decoy costs extra *on top of* flattening); **`ROUTES-NOT-ADDITIVE`** otherwise. Prediction: NOT-ADDITIVE
  on this host, because the L1b penalty alone is within churn.

Descriptive, no rule: the full ordering `rate_L0 ≥ rate_L1 ≥ rate_L1b ≥ rate_L2 ≥ rate_L3` (expected), per-tier
parse rate, and the number of items whose tier rows could not produce a call string (`task_bank.build_call`).

## Identity gate (exit 3, nothing reportable on failure)

The L0 and L1b prompts rebuilt here through `steer_run.build_user` + the chat template must be **bit-identical**
to `l0_prompt_ids` / `l1b_prompt_ids` in the banked traces on every item. Any mismatch means the ladder is not
on the banked footing. Sanity (not a gate): greedy `acc_L0` / `acc_L1b` should land within the measured churn
band of 0.567 / 0.533 (|Δ| ≤ 9/60); a larger gap is reported as a discrepancy, not silently accepted.

## Setup (frozen)

Same 60 items as every run here (`select_pairs`, seed 20260724), tier rows from
`data/stimuli/dataset_{a,b}/dataset_{a,b}.jsonl` via `nla_tiers.tier_rows`; truth is the tier row's own
`expected_output`, asserted equal to the L0 truth (functional equivalence is execution-validated; a mismatch
skips the item and is counted). Grading `steer_run.graded`, `MAX_NEW_GEN` 1100, unchanged. No steering, no AV/AR,
no vectors — forward generation only. Multi-sample via `num_return_sequences` at batch 1 (no padding; kept
identical to the steering runner so rates are comparable across the two). Code `nla/src/nla_tier_accuracy.py`;
output `data/nla/ml/gemma4b/gate/tier_ladder/{tier_rows.jsonl,tier_stats.json}`. `--deterministic` OFF.

## Cost

5 tiers × 60 items × (1 greedy + one 8-sample call) ≈ 600 generate calls ≤ 1100 tokens on a 4B host:
**~3–4 GPU-h**, 8 h wall. One GPU, `h200,h100`.
