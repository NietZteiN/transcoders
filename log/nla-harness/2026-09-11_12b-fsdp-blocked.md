# 2026-09-11 · Phase C blocked: 12B FSDP training hangs after step 1 (four attempts, cause not yet isolated)

**Thread:** nla-harness · **Experiment:** Phase C (12B budget test, pre-registered in
[`2026-09-10_12b-budget-prereg.md`](2026-09-10_12b-budget-prereg.md)) · **Status:** BLOCKED, diagnosis in progress
**Does not affect** Phase B, which is complete: [`2026-09-10_multilayer-gate-results.md`](2026-09-10_multilayer-gate-results.md).

## What works

Everything except the 12B training loop itself:
- **12B text host** materialised (`/scratch/juno/jvl210002/nla_ml_gemma12b/host_text`, 626 keys, 23.5 GB, `㈜` = token
  246566 — the SAME tokenizer as the 4B host, which is why the corpus and all 174,423 explanations are reused unchanged).
- **12B activations, all 48 layers**, 138 GB on scratch, ~49 min (job 389378). Injection scales verified against the
  frozen `ceil_2sf_of_mean_norm` rule on `mean_av_train`: L2 860, L7 2200, L32 74000, L47 180000.
- **FSDP equivalence gate passed THREE times** across three code revisions
  ([`2026-09-10_fsdp-equivalence-gate.md`](2026-09-10_fsdp-equivalence-gate.md)): step-1 loss **1.9e-16**
  (bit-identical), step-1 grad_norm ratio **1.0000**, checkpoint parity 444 tensors / 7.76 GB.
- **Memory model accurate to 0.3 %**: predicted 134 GB/GPU for 2-way 12B, measured **133.7 GB** (and 39 GB predicted /
  39.07 measured at 4B; 112 / 113.5 with sharded grads).
- Step 1 of the 12B AV runs and is **bit-identical across all four attempts and three different nodes**
  (loss 3.606, grad_norm 106.2, mem 133.7 GB) — so the distributed numerics are reproducible.

## The blocker

Four attempts, all failing the same way after completing step 1:

| job | fsdp_util | node | failure |
|---|---|---|---|
| 389416 | 31927c7b (per-microbatch sync) | g-08-05 | watchdog timeout, `_REDUCE_SCATTER_BASE` 224 M elems, 600 s |
| 389569 | 6a4dbb1d (+no-sync accumulation) | g-08-12 | watchdog, `ALLREDUCE NumelIn=1`, 600 s |
| 389666 | 9eb9c772 (+determinism, +timeout, +device_id) | g-07-08 | watchdog, `ALLREDUCE NumelIn=1`, **ran for 3 600 062 ms** |
| 389781 | 9eb9c772 (+`PYTHONUNBUFFERED`) | g-07-08 | watchdog, `ALLREDUCE NumelIn=1`, 600 s |

**`SeqNum=486` in every attempt.** A flaky interconnect would not fail at an identical collective count four times, so
this is a **deterministic divergence**, not a hardware flake. The 1-element ALLREDUCE is the `dist.barrier()` in
`fsdp_util.dist_cleanup`, reached from `main()`'s `finally` — i.e. **rank 0 leaves `stage_sft_av` after ~485 collectives
(roughly one step) while rank 1 keeps training**, and the barrier then waits out the whole timeout.
With `PYTHONUNBUFFERED=1` there is **no Python exception**, and no `--force` skip message, so rank 0 appears to return
*normally*. The only normal returns in the stage are the `eval.json` exists-check (not triggered; its message is absent)
and `if not DC.is_main: return` in the save path — which rank 0, being main, should not take.

Instrumented attempt **389786** prints, from every rank, `len(tr)/gb/accum/steps_per_epoch/total` and
`TRAIN LOOP EXITED after step=S of total=T`, to distinguish: (i) ranks disagreeing on the loop bounds, (ii) an exit that
is neither the break nor an exception, (iii) rank 0 actually stuck mid-step with the "exits early" reading being wrong.

## Process failure worth recording

The barrier appeared in all four logs and I treated it as the fault three times running, building a traffic fix, a
determinism fix and a timeout fix on top of a symptom. Two of those were genuinely needed (below), but the
`SeqNum=486` invariance was visible after the **second** attempt and should have sent me straight to per-rank
instrumentation. Compounding it, raising the timeout to 3600 s turned one diagnostic cycle into an hour of waiting for a
barrier I already expected to fail; the short 600 s timeout is the better diagnostic setting.

## Genuine findings, kept regardless of how the blocker resolves

1. **These `h200_nvl` nodes have NO NVLink.** `nvidia-smi topo -m` reports **SYS** between the two GPUs with NUMA
   affinity **3 vs 4** — peer traffic crosses GPU → socket 3 → UPI → socket 4 → GPU. The product name misleads; I had
   been reasoning from "NVLink-local, FSDP never touches the network", which was wrong.
2. **Per-microbatch gradient sync is pathological here.** 48 blocks × 8 microbatches × 896 MB ≈ **344 GB of
   reduce-scatter per optimizer step**; job 389416's step 1 moved that in ~41 s, implying **~8.4 GB/s** — exactly a
   host-mediated cross-socket path. `set_grad_sync` (no-sync accumulation, syncing only on the last microbatch) cuts it
   **8×** to ~43 GB/step with the arithmetic unchanged (re-gated: loss still 1.9e-16, throughput 0.70× → 0.79× of one
   GPU, memory 56.2 → 62.2 GB at 4B).
3. **Collective schedules must be data-independent.** `reduce_replicated_grads` originally skipped params whose `.grad`
   was None; ranks process different microbatches, so the counts could diverge and deadlock. Now every param is reduced
   (zero grad materialised if absent), with a regression test. `rank_microbatches` also raises if the microbatch count is
   not divisible by world size, instead of hanging — a latent hazard, since global_batch 128 / micro 8 happens to give 16
   (even at world 2) and the holdouts 108 and 106.
4. **`TORCH_NCCL_WATCHDOG_TIMEOUT_SEC` is not honoured** (389569 still showed `Timeout(ms)=600000` with it exported).
   The timeout must be passed to `init_process_group(timeout=...)`; `device_id=` should be set there too, which silences
   "Guessing device ID based on global rank" — a warning the docs say can itself cause hangs. **It was not the fix
   here**: 389666 already ran with both and hung anyway.
5. **A watchdog SIGABRT loses buffered stdout**, so the exception that sends a rank into `finally` is invisible without
   `PYTHONUNBUFFERED=1`. Set it in any multi-rank job.
6. **MIG slices cannot host FSDP** (job 389294): a process can use only one MIG instance, so every rank sees
   `device_count()==1`, and MIG instances cannot do IPC/NVLink peer transfers.

## If the blocker resolves

L32 alone decides **H-C1**; L2/L7 only add H-C2. Remaining cost ~5–6 h for L32 (AV ≈ 3.2 h at the measured 465 tok/s
first-step rate, AR ≈ 1.0–1.5 h, check, then ~100 min vectors + ~5 min score). The alternative route is 4-way FSDP on
**g-04-02** (4 × H100 80 GB HBM3, uncapped partition, likely NVLink) — but note that at 4-way on 80 GB cards the
no-sync unsharded gradients (~41 GB) do **not** fit, so that route needs per-microbatch sync, which NVLink would make
affordable. g-04-02 was held by another user's 22 h job throughout.

---

## RESOLVED (same day): the cause was a CUDA OOM, not communication — and 2-way FSDP cannot fit this recipe

**How it stayed hidden for five attempts.** Every log showed a hung NCCL collective because of a defect in my own error
path: Python generates a traceback only *after* `finally` completes, and `finally` called `dist_cleanup()`, whose
`dist.barrier()` blocks forever while the other rank is still training. The watchdog then SIGABRTed the process, so the
exception was never formatted — `PYTHONUNBUFFERED=1` could not help, because the text did not exist to be flushed.
Fixed by printing the traceback in `main()`'s `except` **before** any collective, and by making `dist_cleanup` **skip the
barrier when `sys.exc_info()` shows an exception propagating**. Attempt 6 (job 389819) then printed the real error
immediately.

**The error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB.
GPU 1 has a total capacity of 139.72 GiB of which 26.69 MiB is free.
Including non-PyTorch memory, this process has 139.68 GiB
```

**Why my memory model missed it.** I validated against `torch.cuda.max_memory_allocated()` — 133.7 GB measured against
134 predicted — and treated the agreement as proof the budget was understood. That metric **excludes allocator reserve
and non-PyTorch memory**, chiefly NCCL's device and pinned staging buffers, which are large here precisely because the
node has no NVLink. Measured gap between allocated peak and process usage: **~24 GB**.

| attempt | config | peak allocated | process | died on |
|---|---|---|---|---|
| 389819 | no-sync, mb 8 | 133.7 GB | 139.68 / 139.72 GiB | 30 MiB request |
| 389847 | no-sync, mb 4, `expandable_segments` | 131.6 GB | ~139.7 GiB | 30 MiB request |
| 389859 | **per-microbatch sync**, mb 8 | **113.3 GB** | 137 GiB | **3.75 GiB** request, 2.36 GiB free |

Halving the micro-batch bought only 2.1 GB, so activations were never the large term. And the fix for the slow
interconnect turned out to be what broke the budget: **no-sync accumulation keeps gradients UNSHARDED across a step,
costing 20.4 GB at 12B.** Removing it brings allocation down to the predicted 113.3 GB — and it still OOMs, now on
fragmentation headroom rather than on totals.

**The arithmetic, which settles the route:** blocks (params+grads+Adam, fp32, 16 B/param) ÷ 2 ranks = 81.6 GB, plus the
replicated tied embedding 16.1 GB, is a **97.7 GB floor** before a single activation; with ~24 GB of NCCL/reserve
overhead that leaves **~18 GB** on a 139.7 GiB card, which 12B transients exceed. **2-way FSDP cannot train this recipe
on a 143 GB H200, in either sync mode.**

**Routes that do fit** (none attempted; each needs a decision and a re-gate):
| route | footprint | cost |
|---|---|---|
| CPU-offload Adam states (FSDP2 `CPUOffloadPolicy`) | 57 + 24 ≈ **81 GB** | slower per step; math unchanged |
| 4-way multi-node (2 × h200 nodes, InfiniBand `mlx5_{0,1,2}` present) | 40.8 + 16.1 + 24 ≈ **81 GB** | torchrun rendezvous work |
| 4-way on g-04-02 (4 × H100 80 GB) | ≈ 81 GB vs 80 GB | **probably does not fit** once overhead is counted |

**Process lesson, recorded deliberately.** Eight attempts: one on the real problem, five lost to a symptom my own error
path manufactured, and two spent on memory knobs that could not close a 24 GB gap. Two things would have short-circuited
it: reading `SeqNum=486`'s invariance across attempts as *deterministic logic* rather than a flaky link (visible after
attempt 2), and not treating agreement between a prediction and `max_memory_allocated()` as evidence that the full
hardware budget was understood.

---

## FIXED: CPU-offload the sharded optimizer state (the only lever that does not change the method)

`fully_shard(block, offload_policy=CPUOffloadPolicy())` moves the 81.6 GB of sharded fp32 block state (params 20.4 +
grads 20.4 + Adam m/v 40.8) into host memory and all-gathers per block for compute. **Verified at 4B against the banked
single-GPU arm (job 389921) with the tightest agreement of any configuration tried:**

| check | offload | (per-microbatch sync) | (no-sync) |
|---|---|---|---|
| step-1 loss rel | **1.90e-16** | 1.90e-16 | 1.90e-16 |
| step-1 grad_norm rel / ratio | 1.56e-06 / **1.0000** | 6.25e-07 / 1.0000 | 1.87e-06 / 1.0000 |
| mean rel loss diff | **4.68e-05** | 1.27e-04 | 2.33e-04 |
| final rel loss diff | **1.19e-05** | 1.93e-04 | 1.70e-04 |
| peak GPU mem (4B) | **32.3 GB** | 56.2 GB | 62.2 GB |
| throughput vs 1 GPU | 0.29x | 0.70x | 0.79x |

Agreement is *tighter* with offload, presumably because the CPU fp32 optimizer step is more deterministic than the fused
CUDA path. **Rejected alternatives** that would also fit but change the update rule, and so would confound the budget
comparison H-C1 exists to make: **bf16 master weights** (12 B/param) and **8-bit Adam**. Two incidental fixes were
required: `clip_grad_norm` accumulated into a CUDA scalar and raises once gradients live on the host (now accumulates as
Python floats, device-agnostic), and the host memory request had to come down to the node's 375 GB cap.

**At 12B (job 389957) this is the first configuration that fits and trains:**
- **GPU 73.2 → 80.8 GB of 139.7** (was 137 → OOM), ~60 GB of headroom.
- **Passed step 2 for the first time** in nine attempts, and reached step 10 with loss 3.606 → 2.167 and
  grad_norm 106.2 → 13.4 — i.e. it is genuinely learning, not merely surviving.
- **Cost: 114 s/step measured** (step 1 at 04:11:03, step 10 at 04:28:13) → **AV 21.5 h**.

389957 was cancelled rather than allowed to run its 20 h limit: the AV stage writes nothing until it completes, so
hitting the wall at step ~630 would have produced no output at all. Resubmitted as **389994 with a 40 h limit**
(h200 MaxTime is 2 days), giving AV 21.5 h + AR ~4 h + check.

**Possible accelerator, gated separately (job 389983): offload + no-sync together.** With offload, per-microbatch
syncing ships gradients host-ward on all 16 microbatches; deferring to the last one cuts that 8x, and the ~20 GB of
unsharded gradients now fits inside the headroom. Both components are individually verified; the gate checks the
combination before the primary run is switched to it. If it holds, AV drops to roughly 6-9 h.

**Accelerator hypothesis REFUTED (job 389983).** I expected offload + no-sync to beat offload alone, reasoning that
per-microbatch syncing ships gradients host-ward on all 16 microbatches. Measured at 4B: **323 tok/s vs 495**, i.e.
**0.65x — 35 % slower** — for +11.9 GB of memory (equivalence still PASSES: loss 1.9e-16, grad_norm ratio 1.0000).
The reasoning was backwards: with per-microbatch syncing FSDP **overlaps** each block's reduce-scatter and D2H copy with
the next microbatch's compute, whereas deferring them all to the final microbatch creates a serial burst with nothing to
hide behind. Projected to 12B this would have turned AV from 21.5 h into ~33 h. **Offload-only is the fastest verified
configuration**, and job 389994 was already running it, so nothing was switched. Fourth trend-style prediction of this
project to be refuted by measurement (after the L22 fve forecast, the spectral-difficulty caveat and the PR-fve
inference) — the pattern is mine: plausible mechanism, asserted before measuring.
