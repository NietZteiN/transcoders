# 2026-09-10 · PRE-REGISTRATION — does training budget explain the 12B/4B fidelity gap? (H-C1…H-C3)

**Thread:** nla-harness · **Experiment:** Phase C · **Status:** pre-registered, nothing run yet
**Written BEFORE** any 12B activation extraction or SFT. Rules below are frozen; a change after the first
submission is a new dated entry, not an edit.

## The question

[`2026-09-10_multilayer-gate-results.md`](2026-09-10_multilayer-gate-results.md) reported that our 4B pairs reach
C3 fidelity `S_c3/S_swap` = **0.68** at relative depth 0.65 (L22/34), where the released
`kitft/nla-gemma3-12b-L32` pair reaches **0.98** at L32/48 (relative depth 0.67, W-family). In that entry I attributed
the gap to **training budget** — ours is 86 k AV rows, 1 epoch, no RL; the vendored recipe is 100 k docs × 10 positions
with RL. **That attribution is an untested guess**, and it is the kind of guess that has been wrong three times in this
project already (the L22 fve forecast, the spectral-difficulty caveat, and the PR↔fve inference — all withdrawn on
2026-09-10). This entry makes it falsifiable.

Two explanations are confounded in the 0.68-vs-0.98 comparison: **host size** (4B vs 12B) and **budget/method**
(our SFT-only pipeline vs the full recipe with RL). Training OUR pipeline on the SAME host at the SAME layer
de-confounds them.

## Setup (frozen)

Host `google/gemma-3-12b-it`, text-only ckpt `/scratch/juno/jvl210002/nla_ml_gemma12b/host_text` (48 layers, d 3840).
Config `nla/configs/nla_ml_12b.yaml` sha256 `c44d7ca2e60bf2be…` — **every hyperparameter identical to the 4B run**
(lr 1.41e-5, global batch 128, micro 8, 1 epoch, no RL, seed 20260724). Differences are only host id, n_layers,
d_model and root.
**Reused unchanged** (the 4B and 12B share a tokenizer — `㈜` = token 246566 in both): the 20 000-doc corpus and the
**174 423 explanations**. The explanations describe the *text context*, not activations, so they are layer- and
model-independent; they were written by Gemma-3-12B itself. Caveat to carry: they were written without reference to
either host's actual predictions.
Training uses `--fsdp` (2 GPUs, one node), verified equivalent to the single-GPU recipe in
[`2026-09-10_fsdp-equivalence-gate.md`](2026-09-10_fsdp-equivalence-gate.md) (step-1 loss 1.9e-16, grad_norm ratio
1.0000, checkpoint parity).
Layers: **L32** (the released pair's layer, the primary comparison) plus **L2** and **L7**.
Scoring: `nla_ml_gate.py` sha `1f7a9d31…` on the same 60 items / 471 spans as every other run in this family.

## Hypotheses and decision rules (frozen, declared before any data)

- **H-C1 — budget explains the gap.** Our 12B L32 pair's `S_c3/S_swap` is compared with the released pair's 0.98 on the
  same items. **BUDGET-EXPLAINS** if our ratio ≤ 0.80 (i.e. we fail to reproduce the released pair's fidelity on the
  same host and layer, so the gap follows our budget rather than host size). **HOST-EXPLAINS** if our ratio ≥ 0.90
  (our budget suffices at 12B, so the 4B 0.68 was about the smaller host). **INDETERMINATE** in between (0.80, 0.90).
  CI via the gate's cluster bootstrap over items, N_BOOT 10 000, seed 20260724.
- **H-C2 — is "early is better" host-invariant?** At 4B, three independent measures pointed at L2–L7: fidelity ratio
  peak (0.93 at L2), `best_single_edit_layer = L2`, and the largest geometric clean-vs-decoy separation
  `cos(h0,h1b)` (0.68–0.80 at L0–L3). **EARLY-INVARIANT** if at 12B `S_c3/S_swap` at L2 and L7 both exceed L32's by
  ≥ 0.05 with CIs excluding zero. **EARLY-IS-4B-ONLY** if both are ≤ L32's. Otherwise **MIXED**.
- **H-C3 — liveness at 12B** (descriptive, no verdict word): report rules (a)–(d) per trained layer. Prediction stated
  now: L2 will be live (the 4B L2 had fve 0.358) and L0-like death is not expected at L2.

**Declared in advance, so it cannot be read as a result later:** a ratio between 0.80 and 0.90 is INDETERMINATE and will
be reported as such rather than argued toward either side. If only L32 completes (scheduler or quota), H-C1 is still
decidable and H-C2 is simply not adjudicated.

## Known limitations, stated up front

1. **Not a clean budget experiment.** The released pair differs from ours in data scale AND in having RL, and its exact
   data is not ours. H-C1 therefore tests "our pipeline at 12B L32" vs "the released artefact", not one isolated factor.
2. **Explanations are ours, not the recipe's.** 174 k local Gemma-3-12B explanations vs the recipe's API-written set at
   ~10× the scale. This is part of what "our budget" means here.
3. **`S_swap` depends on the host's own behaviour.** The 12B host answers the 60 items differently from the 4B one, so
   `G_prompt_swap` is not numerically comparable across hosts; only the RATIO `S_c3/S_swap` is, which is why H-C1 is
   framed on the ratio.
4. **Four of 33 live 4B layers were causally inert** (a prompt-position write needs attention above it; L33 was exactly
   0.000). At 48 layers the equivalent dead zone is roughly L45–L47; L32 is far from it, so this does not affect H-C1.

## Cost and plan

Extract 12B acts (48 layers × 3.07 GB = 147 GB, scratch) → train L32, L2, L7 with `--fsdp` (~4–5 h each at 2 GPUs) →
`nla_ml_gate.py` vectors + score. **~15–20 GPU-h total**, against ~194 GPU-h for all 48 layers, which is deliberately
not being done: the targeted layers discriminate between hypotheses, 48 more pairs of unknown quality would not.
