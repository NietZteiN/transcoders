# Scoping: a multi-layer NLA pair on a small host (Gemma-3-4B-it)

*Last updated: 2026-09-07 — scoping only. No GPU has been spent, no trainer code exists, the host
is not yet downloaded. This document sizes the work so a go/no-go can be taken once H-W31 is in.*

## 1. Why

The W family on `google/gemma-3-12b-it` with the released one-layer pair
`kitft/nla-gemma3-12b-L32-{av,ar}` settled two things about the channel (thread
[`log/nla-harness/`](../log/nla-harness/README.md), 2026-09-03 → 09-07):

| fact | number | entry |
|---|---|---|
| The NLA channel is causally faithful | `C3pure` +44.94 vs `P_patch` +45.71 G_sum nats → **0.983** [0.967, 0.999] | `2026-09-05_nla-fidelity-results` |
| The *edit* step delivers nothing above random | W1 edit **+19.47** vs random vector at the same positions **+19.60** | `2026-09-04_writeback-results` |
| The effect is item-specific, accumulative, saturating | foreign-item clean span +15.18 (gap +30.53, 49/49 items) | `2026-09-05_null-battery-results` |

So transport through English is ~lossless, but *editing* the English does not move the model in
the direction the edit describes. Two readings compete, and a one-layer pair cannot tell them
apart:

- **(R1) One-layer artefact.** Layer 32 of 48 is a late site; what the verbalizer reads there is
  already a compressed, answer-shaped summary, and the reconstructor maps an *edited* sentence to
  a point that is on-manifold for the sentence but off-manifold for the surrounding computation.
  A pair trained at an earlier layer, or a stack of pairs at several depths written jointly,
  would give the edit a place to take hold.
- **(R2) Channel property.** The AR is trained on `2(1−cos)` against activations whose
  explanations describe the *source text*, so it learns "sentence → the activation such text
  would produce", not "sentence → the activation the model would need for this claim". Editing
  the sentence then produces a plausible activation for a *different* text, which is exactly what
  "matches a random vector" looks like. More layers would not change this.

Training a small host with NLA pairs at three depths is the cheapest experiment that separates
R1 from R2: if a multi-layer edit beats the one-layer edit on the same corpus and readout, R1 is
alive; if it still lands on the random benchmark, R2 is favoured and the edit protocol should be
dropped from the programme.

**H-W31 feeds this directly.** The head-mediation map (job 382366, running) says *where* the
L32 write is read: if the effect is `CONCENTRATED` in a few late global heads (35/41/47), the
informed second write site is immediately *after* those heads; if `DISTRIBUTED`, depth spacing
is the right default. Section 5 conditions the layer choice on that verdict.

## 2. Host: `google/gemma-3-4B-it` (decided 2026-09-07)

| property | Gemma-3-4B-it | Gemma-3-12B-it (current) |
|---|---|---|
| layers | 34 | 48 |
| d_model | 2560 | 3840 |
| q / kv heads, head_dim | 8 / 4, 256 | 16 / 8, 256 |
| sliding window / global cadence | 1024, every 6th layer (global: 5, 11, 17, 23, 29) | 1024, every 6th (…, 35, 41, 47) |
| vendored preset default layer (2·L/3) | **22** | 32 (the released pair's site) |
| bf16 weights | ~8.6 GB | ~24.4 GB |
| in `HF_HOME` (`/work/jvl210002/migration/hf_home`) | **no — step 0 is the download** (gated, token at `$HF_HOME/token`) | yes |

Same family and tokenizer as the released pair, so the injection token (`㈜`), chat template,
`local_av.py` in-process injection path, and the whole W harness (`steer.py`, `nla_writeback.py`,
`nla_tiers.py`, `arm_guard.py`, `head_patch.py`) carry over with only a `HOSTS` entry and the
`d_model`/`n_heads` constants changed. Qwen hosts are excluded by the standing constraint.

## 3. What the vendored recipe actually costs

From `nla/vendor/nla-repo/configs/TRAINING_NOTES.md`, `configs/datagen/gemma3_12b_ultrafineweb_100k.yaml`
and `nla/datagen/stage2_api_explain.py`:

- **Stage 0 (activations).** 100 k Ultra-FineWeb docs × 10 positions/doc = ~1 M vectors, chunk
  256 tokens. Forward passes of the host only. **This is the only stage that scales with the
  number of layers, and it scales for free** — one forward yields every layer's residual; the
  extractor must simply keep three hidden states instead of one.
- **Stage 1 (split).** 25 % AV-SFT / 25 % AR-SFT / 50 % RL.
- **Stage 2 (explanations).** An API model (`claude-sonnet-4-6`, ≤300 tokens, ~500 k calls for
  the SFT half) explains `detokenized_text_truncated` — the **source text**, not the activation.
  **Explanations are therefore layer-independent**: one stage-2 corpus serves every layer, and
  its cost is paid once regardless of how many pairs are trained.
- **Stage 3 (build).** Parquet + sidecar per layer (cheap).
- **SFT.** Per layer per side. The profiled Qwen2.5-7B point: AV (full 28-layer fine-tune) ~5 s/step
  at batch 256, 1000 steps ≈ 1.4 h on 2×H100; AR (20-layer truncated + identity-init `value_head`,
  loss `2(1−cos)`) ~3 s/step ≈ 0.8 h on 2×H100. A 4B host at the same batch is ~0.5× those.
- **RL (GRPO, 50 % of the data).** Optional; the released checkpoints used it. Needs Miles +
  SGLang + Ray — **SGLang is absent on juno** (why `local_av.py` exists) and none of it is in
  `nla-mi`. Skipped in this scoping.

### Budget at smoke scale (10 k docs, 3 layers) — what a go decision buys

| stage | compute | wall (h200 node, 1 GPU unless stated) |
|---|---|---|
| 0 download 4B-it | — | minutes |
| 0 activations, 10 k docs × 10 pos, 3 layers kept | 100 k forwards of a 4B at T≤256 | ~20 min |
| 2 explanations, 50 k (SFT half) | **local** Gemma-3-12B-it, ≤300 new tokens, batched | ~2 GPU-h — **or** API |
| AV-SFT × 3 layers | 25 k rows each, batch 128 | ~3 × 15 min |
| AR-SFT × 3 layers | 25 k rows each | ~3 × 10 min |
| W-harness gate (§4) on 60 items | 6 arms × 60 items × (AV read + AR + scoring forward) | ~1 h |
| **total** | | **≈ 5 GPU-h** + queue |

At full scale (100 k docs) multiply stages 0/2/SFT by 10: ~40 GPU-h without RL. The explanation
source is the one real cost decision — local 12B generations cost ~20 GPU-h at full scale and no
API budget; `claude-sonnet-4-6` costs ~500 k calls and no GPU. Local is the default here because
the explanations are text descriptions and the W harness measures a *relative* effect (edited vs
random vs one-layer) that does not depend on their absolute quality.

**Recommendation on quality:** the released 12B pair's stage-0 gate on this project's own stimuli
gave centred cosine +0.68 matched vs +0.05 within-item (`2026-09-04_cycle-gate-results`). A 10 k-doc
smoke pair will be far below that; the gate in §4 must be run on the smoke pair *before* the
edit test, with `W0`-style thresholds relaxed only by pre-registration, never after the fact.

## 4. Pre-registered success gate (frozen here, before any training)

Reuse the W harness unchanged — same 60 items, same repaired anchoring (`--repair`), same
teacher-forced `G_sum` readout over `l1b_prompt_ids + l0_reply_ids`, same `PositionReplacer`
norm-matched write, same `arm_guard` stamps.

Arms, each written at the L1b identifier spans (all layers of the pair written jointly for the
multi-layer arms):

| arm | what | role |
|---|---|---|
| `M_edit` | multi-layer: read at each of the 3 layers, apply the *same* English edit, reconstruct, write all 3 | the test |
| `S_edit_ℓ` | single-layer edit at each layer ℓ alone | the one-layer baseline on this host |
| `M_c3` | multi-layer round trip of the *clean* state (no edit) | fidelity ceiling |
| `M_random` | random direction at the same positions and layers, norm-matched | the generic-perturbation benchmark (the one W1 matched) |
| `M_foreign` | edit text from a different item | content null |
| `SELF` | write each position's own activation back at every layer | identity, tol 1.0 nats as banked |

Rules (all on summed `G_sum`, cluster bootstrap over items, N_BOOT 10 000, seed 20260724):

- **Gate 0 (channel alive on this host):** `M_c3 / M_prompt-swap` ≥ 0.25 with a CI excluding 0 —
  the 12B pair sat at 0.40 of the swap. Below this the pair is too weak to ask the edit question.
- **H-M1 (multi-layer edit beats one-layer):** `M_edit − max_ℓ S_edit_ℓ` > 0 with a 95 % CI
  excluding 0. Anything else → `M-NO-GAIN`.
- **H-M2 (edit beats the random benchmark):** `M_edit − M_random` ≥ 0.30 · `M_c3` with a CI
  excluding 0 → `M-EDIT-LIVE`. If H-M1 passes but H-M2 fails → `M-GAIN-BUT-GENERIC` (more layers
  perturb more; the edit still carries nothing). If both fail → `M-GENERIC`, and R2 is favoured.
- **Specificity:** `M_edit − M_foreign` > 0 with a CI excluding 0, else the edit is not
  content-bearing whatever H-M2 says.
- **Priors, recorded now:** `M-GENERIC` (R2) at ~60 %, `M-GAIN-BUT-GENERIC` ~25 %, `M-EDIT-LIVE`
  ~15 %. A clean `M-GENERIC` is publishable: it says the NLA edit protocol fails for a reason that
  is not depth.

## 5. Layer choice, conditioned on H-W31

Default (no H-W31 information): 25 / 50 / 75 % depth → layers **8 / 17 / 26** of 34 (17 is a
global-attention layer; 22 is the preset default and the natural "released-pair-equivalent"
site). Adjust after `heads_stats.json` lands:

| H-W31a verdict | implication for the 12B | choice on the 4B |
|---|---|---|
| `CONCENTRATED` in global heads at 35/41 | the write is read by a few heads shortly above the site; a second pair *just above* those heads gives the edit a second, later handle | 22 (preset) + the layer after the first global layer above it (**24**) + one early site (8) |
| `CONCENTRATED` in local heads at 33–34 | read immediately; depth spacing does nothing for that route | 22 + 23 + one early site |
| `DISTRIBUTED` | the state is re-read everywhere above the site; the edit fails because it is *one* write, so space the writes | **8 / 17 / 26** |
| `INTERMEDIATE` | as DISTRIBUTED, but weight the middle site toward the layer where the summed `nec` profile peaks (rescaled 48 → 34) | 8 / peak / 26 |

H-W31c (`SAME-CIRCUIT` vs `DIFFERENT`) matters too: if C3pure routes differently from the raw
state, the reconstructor is already producing an off-route vector at *one* layer and stacking
layers compounds it — that would move the prior further toward R2 before anything is trained.

## 6. What a minimal plain-PyTorch SFT trainer needs (spec, not code)

Miles/SGLang/Ray are not on juno and are not worth porting for a 4B smoke. The vendored
`nla/` package already holds everything except the optimisation loop:

- **Data:** the vendored stage 0–3 pipeline (`nla.datagen.run_pipeline`) runs without Miles;
  `AnthropicProvider` swapped for a local-generation provider (same interface: text in,
  explanation out). Emit the standard parquet + `nla_meta.yaml` sidecar per layer so
  `local_av.py` / `NLACritic` load the result unchanged.
- **AV trainer:** `AutoModelForCausalLM` (text-only path as for the 12B), full fine-tune, bf16,
  AdamW lr 2e-5 → 2e-6 cosine (√-scale by batch/256), warmup 5 %, prompt `Explain:
  <concept>㈜</concept>`, injection by replacing the `㈜` embedding with `injection_scale ·
  unit(h)` (`NLAClient._build_embeds` is the reference — reuse it, do not reimplement), loss on
  response tokens only. 1 GPU, micro-batch 8–16 at T≈250 for a 4B without checkpointing.
- **AR trainer:** truncate to layers `0..ℓ`, add `value_head = Linear(d, d)` **identity-initialised**
  (`torch.eye`; the notes measured a 17 % worse start otherwise), loss `2(1−cos)` on the last
  token, same schedule. Report `fve` against the **raw-mean** baseline.
- **Multi-layer:** three independent pairs (one per layer), *not* one shared model — the released
  design is per-layer and sharing would confound the depth question with a capacity change.
  Joint writing at inference is just three `PositionReplacer` hooks.
- **Checks before the gate:** real-vs-shuffled-activation loss gap (the notes' ~0.21 signal),
  `n_no_tags` on reads (the 09-04 `MAX_NEW_READ` defect), CJK rate, and the stage-0 centred-cosine
  gate per layer.

## 7. Go / no-go

Go if **all** of: H-W31 has landed with the identity gate passed (so §5 can be applied);
Gemma-3-4B-it downloads and produces sane greedy replies on the 60 items (accuracy within the
12B's 0.633 ± 0.15 band — otherwise the flippable ceiling that already binds accuracy claims on
the 12B becomes worse); the 5 GPU-h smoke budget is acceptable on h200. No-go / defer if H-W31
returns `W31-HARNESS-FAULT`, or if the queue makes a multi-job pipeline impractical this week.

## Changelog
- **2026-09-07** — created (scoping only; decisions: host Gemma-3-4B-it, no trainer code, no GPU).
