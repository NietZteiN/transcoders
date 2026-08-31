# 2026-08-27 — PRE-REGISTRATION: Phase-0 triage (P0.1 layer rotation · P0.2 channel · P0.3 site)

**Frozen before any of the three runs. Nothing below is written with knowledge of the results.**

## Why this exists

B4 is refuted (V1 0.550 = V3 0.550, p = 1.00, across a 16× α range) and B5 is null (every CI
spans zero). Both negatives are currently read as *"the belief is not there to edit."* That reading
is not licensed by the experiments, because a second explanation predicts identical tables:
**the injection channel cannot deliver anything, whatever it carries.**

The evidence for the second explanation is inside B4's own control battery. **V4 — the oracle,
which has seen the un-obfuscated program and therefore knows the answer — gained only +0.083, less
than the +0.100 from adding one sentence to the prompt.** If ground truth cannot clear a prompt
sentence through this channel, no direction was going to, and the gate measured delivery rather
than belief.

The channel is narrow by construction (established 2026-08-26): **one vector, at one token position
(`last_prompt`), at one layer (20 of 28)**, whose edit propagates only to layers 21–27 and leaves
the KV entries at layers 0–20 for that position unedited — so every later token can still read the
decoy through the bottom 21 layers.

Phase 0 decides which of the two explanations holds, before more GPU time is spent on either the
position-allocation work or B5's second seed.

## The three experiments

### P0.1 — layer rotation of the task direction *(no NLA; forward passes only)*

**Question.** Is the single-layer constraint hard or soft? And is layer 20 even where the
clean/obfuscated distinction is strongest?

**Method.** The same 60 L0/L1b pairs `steer_run.load_pairs` yields (`dataset_a` + `dataset_b`,
ground truth on the L1b side). One forward pass per prompt with `output_hidden_states=True`, read
at the **final prompt token** — the site B4 steers. For layer ℓ ∈ [0, 27], with
δ_i,ℓ = h_clean,i[ℓ] − h_obf,i[ℓ] and Δ_ℓ = mean_i δ_i,ℓ:

| metric | definition |
|---|---|
| **rotation** | cos(Δ₂₀, Δ_ℓ) |
| **coherence** | mean_i cos(δ_i,ℓ, Δ_ℓ^(−i)) — leave-one-out, so an item never contributes to the mean it is scored against |
| **relative magnitude** | ‖Δ_ℓ‖ / mean_i ‖h_i,ℓ‖ |

**Validation gate (must pass before any metric is believed).** The project's extractor hooks
`layers[K]` and documents that as `hidden_states[K+1]`. This script uses `hidden_states`, so at
layer 20 it must reproduce `ActivationExtractor(…, 20).extract_chat(...).activations[-1]` to within
**1e-4 relative**. A miss means the layer indexing is off by one and every curve is mislabelled.

**Decision rule.**
- **SOFT** — cos(Δ₂₀, Δ_ℓ) ≥ **0.50** for every ℓ ∈ [21, 27]. Injecting the L20-derived vector
  across the 20–27 band is licensed, and Phase 1a gains a depth axis with no training.
- **HARD** — cos < 0.50 anywhere in [21, 27]. Re-using the L20 vector above layer 20 is **not**
  licensed; the single-layer limit is reported as a measured constraint on single-layer NLAs.
- **Reported either way:** argmax_ℓ coherence. If layer 20 is not at or near the maximum, the
  instrument sits at a suboptimal depth for this contrast — a finding about the checkpoint, not
  about the method.

*(0.50 is arbitrary. It is frozen here precisely because it is arbitrary.)*

### P0.2 — is the channel the bottleneck?

**Question.** Does widening the write from one position to many recover the oracle?

**Method.** `steer_run.py --positions all_reply --alphas 1.0`, conditions **V4_oracle, V1_gloss,
V3_taskvec, R_random**, on the same 60 pairs, against the banked `last_prompt` run at the same α.
Primary contrast: **V4 at `all_reply` − V4 at `last_prompt`**.

**Decision rule.**
- **CHANNEL-LIMITED** — V4 improves by **≥ +0.10** balanced Δaccuracy, and its CI excludes zero.
  B4 measured delivery, not belief; the gate must be re-run at the better allocation before its
  negative can be reported as a claim about representations.
- **NOT CHANNEL-LIMITED** — V4 does not improve. The site cannot carry an intervention even with
  perfect information at every reply position, and B4's negative stands as a bounded claim about
  *where a correction can be written*, which is the stronger and more defensible version.
- R_random is read alongside: if random *also* jumps, the gain is generic perturbation and not
  delivery. A V4 gain that R_random matches does **not** count as CHANNEL-LIMITED.

### P0.3 — is layer 20 a live site for any intervention?

**Question.** Can a *different* lever — attention reallocation — move accuracy when restricted to
layer 20 alone?

**Method.** The B5 composed runner (its α=0 validation gate passed 2026-08-26 at |Δ| = 0.0104),
`adversarial_rename`, 164 Java programs / 1,930 cases. Attention steering at layer ranges
**[20,20], [20,21], [20,23], [20,27]** — a scope *curve*, because a single point null cannot
distinguish a dead site from a lever below its operating point. Belief channel off (α = 0)
throughout. **Two seeds: 1000 and 2000.**

Requires a `--steer-layers A:B` flag; `steer_layer_start` / `steer_layer_end` already exist on the
config and `--steer-last-n-layers` is only a suffix wrapper (n=1 gives layer 27, not 20).

**Decision rule.**
- **SITE LIVE** — the [20,20] cell differs from the unsteered baseline by more than the
  seed-to-seed spread of the baseline itself, in the same direction at both seeds.
- **SITE DEAD** — [20,20] is within seed noise while [20,27] is not. Then no single-layer
  intervention of either kind moves this task, and both B4 and B5 are explained by scope.
- **UNINFORMATIVE** — [20,27] is also within seed noise, i.e. the whole curve is flat. Then this
  corpus cannot discriminate and P0.3 says nothing; it is *not* to be read as SITE DEAD.

## What Phase 0 licenses

| P0.2 | P0.3 | conclusion | next |
|---|---|---|---|
| channel-limited | site live | the channel was the bottleneck | Phase 1a — position-allocation paper; re-run the B4 gate at the better allocation |
| not channel-limited | site dead | single-layer late intervention cannot move this task | Phase 1b — readout paper with a bounded, mechanistic causal negative |
| channel-limited | site dead | contradictory — do not report either; diagnose | — |
| not channel-limited | site live | the site works but belief-shaped writes do not | strongest support for the "no item-level belief" reading |

## Standing constraints, restated so they are not renegotiated later

- **Seeds recorded on every run.** P0.3 is two seeds by construction; E5b showed a five-transform
  family average flipping sign between draws on this exact pipeline.
- **`recovered` and `damaged` reported separately**, never netted into Δaccuracy.
- **The banked `last_prompt` numbers are the comparison**, resumed rather than recomputed, so any
  difference is the manipulation and not a re-run.
- **This is exploratory relative to the B4/B5 pre-registrations.** It does not revive a refuted
  hypothesis; it establishes what the refutation was about.
- If P0.1's validation gate fails, **P0.1 is not reported at all** — a mislabelled layer axis is
  worse than no layer axis.

## Provenance

```
GPUs      0, 2, 3 (GPU 1 held by an unrelated vLLM job, 40.7 GB — untouched)
seed      20260724 (P0.1, P0.2) · 1000 + 2000 (P0.3)
corpus    P0.1/P0.2: 60 L0/L1b pairs, dataset_a + dataset_b
          P0.3: 164 Java programs / 1,930 cases, adversarial_rename
models    P0.1/P0.2 Qwen2.5-7B-Instruct · P0.3 Qwen2.5-Coder-7B-Instruct
```
