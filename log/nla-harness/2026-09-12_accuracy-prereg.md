# 2026-09-12 · PRE-REGISTRATION — does steering the NLA improve ACCURACY? (H-S9, run at user request)

**Thread:** nla-harness · **Experiment:** `S9_accuracy_at_user_request`, Gemma-3-4B-it · **Status:**
pre-registered, nothing run.

## Why this runs despite its own gate saying not to

H-S9 was pre-registered in [`2026-09-11_steering-sweep-prereg.md`](2026-09-11_steering-sweep-prereg.md) as a
**gated** stage and the gate **did not fire**: best config `band` +9.85 vs best single layer +8.18, gap **+1.67 <
+3.00** ([`2026-09-12_beta-layerset-results.md`](2026-09-12_beta-layerset-results.md)). The user has asked for the
accuracy result directly and twice — *"The result I want to see is if accuracy improved by steering the nla"* —
so it runs. **That is the user's call and it is recorded as such**, not as the gate having been met. The gate's
number stands unchanged in the banked stats.

## The bound, restated before any number exists

This is the fourth entry to state it and it does not improve with restating: this host scores `acc_l0` **0.567**
→ `acc_l1b` **0.533**, i.e. **2 of 60 items of net headroom** and **7 flippable**, against a measured
same-condition greedy churn of **6–9 items of 60**
([`2026-09-07_flippable-census-results.md`](2026-09-07_flippable-census-results.md),
[`2026-08-29_determinism-floor-structural.md`](2026-08-29_determinism-floor-structural.md)). Exact McNemar at
n = 60: a **perfect** rescue of all 7 flippable items against that churn gives **p = 0.092**. It cannot reach 0.05.

**Therefore this entry pre-commits to reporting NO verdict word and NO significance claim in either direction.**
Effect sizes, Wilson intervals, paired bootstrap intervals and exact McNemar p-values are reported as measured.
A positive point estimate will be reported as a point estimate with its interval, explicitly not as evidence of
improvement; a null will be reported as uninformative, explicitly not as evidence of absence. The full arithmetic
and what would fix it is in [`../../docs/nla_flippable_corpus_scoping.md`](../../docs/nla_flippable_corpus_scoping.md)
(~20 flippable items, ~170 screened, and the corrected finding that **no adversarial-rename generator is reachable**
on this machine).

## Setup (frozen)

Host Gemma-3-4B-it text ckpt; same **60 items** and banked vectors as every run in this family
(`data/nla/ml/gemma4b/gate/vectors/L{K}.npz`); traces `data/nla/p0/trace_llr/gemma4b/traces.jsonl` supply
`truth`, `l0_correct`, `l1b_correct` (so the flippable subset is the banked one, not re-derived). Grading is
`steer_run.graded` / `ANSWER_RE` with `MAX_NEW_GEN = 1100`, **unchanged**, so numbers are comparable with the
banked corpus. Seed 20260724, `--deterministic` OFF. Code `nla/src/nla_accuracy.py`.

**Arms (6):**
| arm | what it is |
|---|---|
| `noop` | unsteered L1b — the baseline |
| `c3_band` | `AR(AV(h_L0))` written at `band` L2–L13, β = 0.35 — **the strongest NLA write this project has** |
| `edit_band` | the NLA **edit** at the same config — *the intervention the question is about* |
| `foreign_band` | decoy → a different item's true term — the content null |
| `edit_single` | the banked standard: single layer **L2**, β = 1.0 (`best_single_edit_layer` from the Phase-B gate) |
| `prompt` | mandatory prompting baseline (CLAUDE.md §4): *"the identifiers in this code may be misleading…"* |

`c3_band` is included deliberately: if even the strongest possible NLA-transported write — which recovers ~91 % of
the clean state's `G_sum` effect — does not move accuracy, then `edit_band` cannot, and that is the informative
shape of a null. Configuration chosen from [`2026-09-12_beta-layerset-results.md`](2026-09-12_beta-layerset-results.md)
(`band` best on the specific effect at β = 0.35) and the paired follow-up
([`2026-09-12_beta-paired-followup.md`](2026-09-12_beta-paired-followup.md)).

**Two readouts:**
1. **Greedy n = 1** — directly comparable with every banked accuracy number here.
2. **Sampled n = 8 per item → a per-item pass rate.** ~**2.8×** smaller per-item SE than one greedy draw
   (0.177 vs 0.500 at p = 0.5) — the only cheap lever on the churn floor. Implementation constraint:
   `num_return_sequences` at **batch 1, unpadded**. `answer_entropy.py:186-190` batches multi-sample generation
   with **left padding**, which shifts every absolute position and would silently break `PositionReplacer`'s span
   targeting; that path is not used.

**Guards.** `n_positions_written` is asserted against the expected span count on every generation (the counter is
per-forward; span positions live in the prompt, so only the prefill forward writes them). The `prompt` arm is
built by decode → insert → re-encode and is **skipped for any item whose decode→encode does not round-trip
exactly**, rather than risk a malformed prompt; the skip count is reported.

## What will be reported

Per arm: greedy accuracy over all 60 with Wilson 95 %, greedy accuracy on the 7 flippable, sampled pass rate with
bootstrap 95 %, and against `noop`: items fixed, items broken, exact McNemar p, and the paired pass-rate
difference with its interval. **No verdict word.**

## Cost

Greedy 6 arms × 60 items ≈ 360 generations; sampled 6 × 60 calls of 8. ~**3–5 GPU-h** on one GPU, 10 h wall
limit, banked vectors, no AV/AR loaded.
