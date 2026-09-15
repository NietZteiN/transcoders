# 2026-09-10 · FSDP memory path for a 12B host — equivalence gate **PASSES**

**Thread:** nla-harness · **Experiment:** Phase C prep (12B NLA) · **Status:** gate passed; 12B training unblocked
**Follows:** [`2026-09-10_multilayer-gate-results.md`](2026-09-10_multilayer-gate-results.md) (Phase B results, `M-NO-GAIN`)

## Why a gate at all

Gemma-3-12B text-only is ~11.2 B params. The trainer holds **fp32 weights + fp32 grads + fp32 Adam m/v**
(`load_gemma_text(..., dtype=torch.float32)` + bf16 autocast), i.e. **16 bytes/param**, with gradient checkpointing
already enabled in both SFT stages. That is **179 GB + ~27 GB activations ≈ 206 GB against 143 GB on an H200** — single
GPU impossible. Rejected alternatives: **8-bit Adam** (bitsandbytes absent; ~139 GB, too tight) and **bf16 weights**
(~161 GB, still over) — both change the update rule, and the entire point of a 12B run is a like-for-like comparison
against the 33 existing 4B pairs and the released `kitft/nla-gemma3-12b-L32` pair. **FSDP keeps the recipe intact** —
but that argument is worthless unless the multi-GPU path is *verified*, hence this gate.

## Result (jobs 389293 arm A / 389326 arm B, 4B host, L2, 20 steps, seed fixed, mb=8, 4000 rows)

| check | measured | pre-declared tolerance |
|---|---|---|
| step-1 loss | **1.90e-16** (bit-identical) | < 1e-4 |
| step-1 grad_norm | **5.47e-07**, ratio **1.0000** | < 5e-3 |
| mean rel loss diff (steps 1/10/20) | 1.71e-04 | < 2e-2 |
| final rel loss diff | 2.82e-04 | < 3e-2 |
| checkpoint | **444 tensors / 7.76 GB — identical to single-GPU** | parity |
| peak memory | 68.1 → **56.2 GB/GPU** | — |
| **EQUIVALENCE** | **PASS** | |

Tolerances were written into `nla/scripts/nla_fsdp_equiv.sh` before any output existed. The **grad_norm ratio is the
sharp diagnostic**: 1.0000 rules out both 0.5 (the normalizer trap) and 0.707 (per-shard clipping).
*Caveat:* the trainer logs every 10th step, so the trajectory comparison is **3 logged points (1/10/20), not 20**.
Holdout agreement: `holdout_loss_real` 1.36e-04, `_permuted` 3.24e-04, and **`gap` 1.59e-02** — the gap is a difference
of two nearly-equal numbers so it amplifies; harmless for liveness rule (a) (which only tests gap > 0) and measured on a
32-row smoke holdout rather than the real 858.

## Design: shard ONLY the decoder blocks

`av_loss` calls **`model.model(...)` and `model.lm_head(...)` directly**, not `model(...)`, because applying the
262k-vocab head only at label positions is what makes the loss affordable. Entering an inner module **bypasses a root
wrapper's pre-forward unshard hook**, so root params stay DTensors and the embedding dies with
`aten.embedding.default got mixed torch.Tensor and DTensor` (job 389302). Gemma3 also **ties `lm_head.weight` to
`embed_tokens.weight`**, and the head is used *after* `model.model(...)` has re-sharded, so wrapping those modules
individually does not help either. Rewriting the loss to call the root would change the method.
**Blocks are invoked through their own module boundary (the layer loop), so block-level `fully_shard` hooks fire
correctly — and blocks are where the parameters are** (~10.2 B of 12B's 11.2 B). The tied embedding (~1.0 B) stays
replicated at 16 GB/GPU. Projected 12B: **2-way ≈ 112 GB/GPU (fits 143 GB H200)**, **4-way ≈ 64 GB (fits 80 GB H100)**.
The 4B memory model predicted 39 GB and measured 39.07 GB at step 1, which is what licenses those projections.

## Five defects the gate caught, all in minutes-long 4B runs

1. **My harness**: `--root` overrides the trainer's *input* root too, so arm A died on a missing `corpus/rows.parquet`
   (job 389065). Fixed with a scratch root symlinking `corpus/`+`explain/` out of /work, plus an input pre-flight check.
2. **`acts/` had been deleted** that morning to free quota — correct for the gate *scoring* stages (they never read
   `acts/`) but it also disables any *training*, including verification. Re-extracted (job 389106); the rebuilt
   activations reproduce the frozen `injection_scale` for L2 **exactly (1200)**, so the gate compares against the same
   inputs the 33 pairs were trained on.
3. **MIG slices cannot host FSDP** (job 389294, `invalid device ordinal`): a process can use only ONE MIG instance, so
   every rank sees `device_count()==1`, and MIG instances cannot do IPC/NVLink peer transfers so NCCL collectives fail
   anyway. `dist_setup` now raises a message naming the limitation. **12B routes are therefore only**: 2-way h200 or
   4-way g-04-02 (4 × H100 80 GB, uncapped).
4. **Root-hook bypass** (job 389302) → block-only sharding, above. Its consequence: replicated params are outside FSDP's
   reduction, so `reduce_replicated_grads()` averages them before the step, or the embedding would train on one rank's
   microbatches instead of the global batch.
5. **Mixed-tensor gradient norm** (job 389310): `clip_grad_norm_` dies on a list mixing DTensor and plain grads
   (`aten._foreach_norm.Scalar`). Clipping the groups separately would be two local norms instead of one global norm —
   a silent change to training — so the norm is assembled explicitly: sharded part SUM-reduced, replicated part counted
   once (already averaged) and deliberately not reduced again.
6. **Tied-weight duplication in the save path**: handing `save_pretrained` an explicit gathered state dict bypasses its
   tied-weight de-duplication, producing **445 tensors / 9.10 GB vs 444 / 7.76 GB** — a redundant `lm_head.weight`.
   At d=3840 that is **2.0 GB per layer, ~96 GB across 48 layers**, in a layout unlike the existing pairs. Now popped
   when `config.tie_word_embeddings`; parity re-verified (job 389326).

## Provenance

| | |
|---|---|
| `nla/src/fsdp_util.py` (new) | sha256 `892aba9913143245…` |
| `nla/src/nla_train.py` | `b924ff21…` → `cae2a2efe5be1cd4…` (119 lines, all behind `--fsdp`; without the flag every helper is an identity and `fused=True` is preserved) |
| `nla/tests/test_fsdp_util.py` (new) | 8 CPU tests, incl. a control proving that omitting `scale_for_world` yields gradients wrong by exactly 1/world |
| `nla/scripts/nla_fsdp_equiv_arm.sh`, `nla/src/fsdp_equiv_cmp.py` (new) | split arms + CPU-only comparison |
| 12B text host | `/scratch/juno/jvl210002/nla_ml_gemma12b/host_text`, 626 keys, 23.5 GB, `㈜` = token 246566 (same tokenizer as 4B) |

## Cluster facts worth reusing

`juno` QOS **MaxJobsPU=4** (partitions `normal`, `h200`); `juno-dev` **=1** (`dev`); **`h100` and `a30` have `QoS=N/A`
→ uncapped**. Two traps: a site plugin **routes gres-less jobs off GPU partitions back under the cap**, so an uncapped
slot costs `--gres=gpu:1` even for CPU work; and login-node **`ulimit -v` is 8 GB**, which defeats the 12B safetensors
remap *and* any `import torch` (`libtorch_cuda.so: failed to map segment`). A 2-GPU-on-one-node request is also hard to
schedule: most h200 nodes have ≤1 GPU free and the idle one (g-07-05) is `IDLE+RESERVED` — splitting the gate into a
1-GPU arm and a 2-GPU arm is what got it running.

## Next

1. Extract 12B activations for the target layers (3.07 GB/layer at d=3840) — **the 174,423 explanations are reusable
   as-is** (they describe the text context, are model-independent, and were written by the 12B itself; ~20 GPU-h saved).
2. Train **L32 first** and compare against the released `kitft/nla-gemma3-12b-L32` pair on the same 60 items / 471
   spans: same host, same layer, same readout, only the training budget differs. That tests the limitation claimed in
   the Phase B entry (our 4B reached **0.68** fidelity at relative depth 0.65 vs the released pair's **0.98** at L32/48),
   which is currently an **untested guess**. Add **L2** and **L7** for the "early is better" question. ~15–20 GPU-h.
3. Pre-register that comparison before running it.
