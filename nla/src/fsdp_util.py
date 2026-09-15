"""FSDP2 helpers for the NLA SFT stages — memory sharding for a 12B host on 2 NVLink H200s.

WHY THIS EXISTS
    Measured on Gemma-3-4B (2026-09-09/10): peak 68.5 GB/GPU = 16 bytes/param with **fp32
    weights + fp32 grads + fp32 Adam m/v** (`load_gemma_text(..., dtype=torch.float32)` plus bf16
    autocast; gradient checkpointing is already enabled in both stages). Gemma-3-12B text-only is
    ~11.2 B params -> 179 GB + ~27 GB activations ~= 206 GB, against 143 GB on an H200 NVL.
    Sharding over the 2 GPUs a juno h200 node provides gives ~103 GB/GPU.

    8-bit Adam (139 GB, and bitsandbytes is not installed) and bf16 weights (161 GB) were both
    rejected: they change the update rule, and the entire purpose of the 12B run is a
    like-for-like comparison against the 33 existing 4B pairs and the released
    `kitft/nla-gemma3-12b-L32` pair. FSDP keeps the optimizer and precision recipe intact.

WHY FSDP2 (`fully_shard`) AND NOT FSDP1
    `stage_sft_ar` selects optimizer parameters BY NAME
    (`[p for n, p in model.named_parameters() if not n.startswith("lm_head")]`).
    FSDP1 flattens parameters into opaque FlatParameters and that selection silently breaks.
    FSDP2 keeps one DTensor per original parameter, so `named_parameters()` still works.

THE NORMALIZER TRAP (read before touching the training loops)
    Both stages divide by a GLOBAL quantity: AV by `n_tok_global` (tokens in the whole global
    batch), AR by `gb` (global batch size). Single-GPU, every microbatch contributes
    `tot_i / N_global` and the sum over microbatches is the exact per-token (or per-row) mean.

    FSDP **averages** gradients across ranks. So if each rank divides its own microbatches by the
    same `N_global`, the reduced gradient is 1/world_size of the correct value:

        single GPU : g  = grad[ sum_i tot_i / N ]
        2 ranks    : (g0 + g1)/2 = (1/2) * grad[ sum_i tot_i / N ]

    `scale_for_world()` multiplies the per-rank loss by `world_size` to cancel that average
    exactly. This is the one error that would leave training "working" while no longer being the
    same method -- which would silently destroy the comparability FSDP was chosen to preserve.
    `nla/scripts/nla_fsdp_equiv.sh` is the gate: it compares 1-GPU vs 2-GPU loss trajectories on
    the 4B host at a fixed seed, and must pass before any 12B layer is trained.

KNOWN, DOCUMENTED DEVIATION
    `torch.optim.AdamW(..., fused=True)` does not support DTensor parameters, so under `--fsdp`
    the optimizer falls back to the `foreach` implementation. That is mathematically the same
    AdamW; it is not bit-identical. The equivalence test therefore checks agreement to a
    tolerance, not to exact equality, and the tolerance is stated in the test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DistCtx:
    """Where this process sits in the job. `world == 1` means the untouched single-GPU path."""
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}" if self.enabled else "cuda"


def dist_setup(enable: bool) -> DistCtx:
    """Join the torchrun process group. No-op (and no torch.distributed import) when disabled."""
    if not enable:
        return DistCtx()
    import torch.distributed as dist
    lr = int(os.environ.get("LOCAL_RANK", 0))
    n_vis = torch.cuda.device_count()
    ws = int(os.environ.get("WORLD_SIZE", 1))
    # MIG slices cannot host FSDP: a process can only ever use ONE MIG instance, so every rank sees
    # device_count()==1 however many MIG UUIDs Slurm put in CUDA_VISIBLE_DEVICES, and MIG instances
    # cannot do IPC/NVLink peer transfers, so NCCL collectives across them do not work either.
    # Observed 2026-09-10 (job 389294, g-06-01, 2 x nvidia_h100_nvl_3g.47gb): rank 1 died with
    # "CUDA error: invalid device ordinal" inside set_device. Fail loudly rather than cryptically.
    if n_vis < ws:
        raise RuntimeError(
            f"--fsdp needs {ws} visible CUDA devices, found {n_vis} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
            "MIG slices cannot be used for FSDP -- request whole GPUs, e.g. "
            "--gres=gpu:nvidia_h200_nvl:2 or --gres=gpu:nvidia_h100_80gb_hbm3:4.")
    torch.cuda.set_device(lr)
    if not dist.is_initialized():
        # The watchdog timeout MUST be set here. `TORCH_NCCL_WATCHDOG_TIMEOUT_SEC` is not honoured
        # (job 389569 still reported Timeout(ms)=600000 with it exported), and on these no-NVLink
        # nodes a single 448-896 MB collective can exceed 10 minutes. `device_id` also silences
        # "Guessing device ID based on global rank", which the docs warn can itself cause hangs.
        import datetime
        secs = int(os.environ.get("NLA_NCCL_TIMEOUT_SEC", "3600"))
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=secs),
                                device_id=torch.device(f"cuda:{lr}"))
    return DistCtx(rank=dist.get_rank(), world=dist.get_world_size(), local_rank=lr, enabled=True)


def dist_cleanup(ctx: DistCtx) -> None:
    """Leave the process group. Skips the barrier when an exception is propagating.

    The barrier exists so ranks exit together, but on the error path it is actively harmful: the
    failing rank blocks on it while the others keep training, the NCCL watchdog eventually SIGABRTs
    the process, and the traceback that explains the failure never gets printed. Jobs 389416-389786
    were all diagnosed as "barrier timeout" for this reason."""
    if not ctx.enabled:
        return
    import sys
    import torch.distributed as dist
    if not dist.is_initialized():
        return
    if sys.exc_info()[0] is None:
        dist.barrier()
    dist.destroy_process_group()


def shard_model(model: torch.nn.Module, ctx: DistCtx, extra: tuple[torch.nn.Module, ...] = ()) -> torch.nn.Module:
    """Shard ONLY the decoder blocks. The embedding/norm/lm_head stay replicated — see below.

    WHY NOT `fully_shard(model)` AT THE ROOT (observed 2026-09-10, job 389302):
        `av_loss` calls `model.model(...)` and `model.lm_head(...)` **directly**, not `model(...)`,
        because applying the 262k-vocab lm_head only at label positions is what makes the loss
        affordable. Entering an inner module bypasses the ROOT wrapper's pre-forward unshard hook,
        so root-level parameters stay DTensors and the embedding dies with
            "aten.embedding.default got mixed torch.Tensor and DTensor".
        Gemma3 also TIES lm_head.weight to embed_tokens.weight, and the lm_head calls happen after
        `model.model(...)` has already re-sharded, so wrapping those modules individually does not
        fix it either. Rewriting the loss to call the root would change the method — the one thing
        FSDP was chosen to avoid.

    Each decoder block IS called through its own module boundary (the layer loop inside
    Gemma3TextModel), so a block-level `fully_shard` has its hooks fire correctly. The blocks are
    also where the parameters are: for 12B, blocks+norm are ~10.2 B of the 11.2 B total, leaving
    ~1.0 B in the tied embedding. Replicating that costs 16 GB/GPU at 16 bytes/param, so
    2-way: 10.2*16/2 + 16.1 + ~14 activations ~= 112 GB  (fits a 143 GB H200)
    4-way: 10.2*16/4 + 16.1 + ~7            ~=  64 GB  (fits an 80 GB H100)

    CONSEQUENCE: replicated parameters are outside FSDP's reduction, so their gradients are
    per-rank only. `reduce_replicated_grads()` must be called before the optimizer step or the
    embedding would train on one rank's microbatches instead of the global batch.
    """
    if not ctx.enabled:
        return model
    from torch.distributed.fsdp import fully_shard
    blocks = getattr(getattr(model, "model", model), "layers", None)
    assert blocks is not None, "expected model.model.layers (Gemma3ForCausalLM)"
    # NLA_FSDP_CPU_OFFLOAD=1 keeps the SHARDED block params/grads/optimizer state in host memory and
    # all-gathers to GPU per block for compute. This is the only way found to fit 12B 2-way on a
    # 143 GB H200 WITHOUT changing the method: the floor is 81.6 GB of sharded fp32 state (params
    # 20.4 + grads 20.4 + Adam m/v 40.8) plus a 16.1 GB replicated embedding, and ~24 GB of
    # NCCL/allocator overhead leaves only ~18 GB for activations -- jobs 389819/389847/389859 all
    # OOMed. Offloading moves that 81.6 GB off the device.
    # Rejected alternatives, both of which would fit but change the update rule and so confound the
    # budget comparison this run exists to make: bf16 master weights (12 B/param) and 8-bit Adam.
    kw = {}
    if os.environ.get("NLA_FSDP_CPU_OFFLOAD", "0") == "1":
        from torch.distributed.fsdp import CPUOffloadPolicy
        kw["offload_policy"] = CPUOffloadPolicy()
    for b in blocks:
        fully_shard(b, **kw)
    for m in extra:
        fully_shard(m, **kw)
    return model


def set_grad_sync(model: torch.nn.Module, enabled: bool, ctx: DistCtx) -> None:
    """Turn FSDP's per-microbatch gradient reduce-scatter on/off (gradient-accumulation no-sync).

    WHY THIS IS ESSENTIAL ON THESE NODES, not an optimisation. `nvidia-smi topo -m` on a juno h200
    node (job 389475) reports **SYS** between the two GPUs and NUMA affinity 3 vs 4: despite the
    "h200_nvl" product name there is **no NVLink** and not even a shared PCIe switch, so peer traffic
    crosses GPU -> CPU socket 3 -> UPI -> socket 4 -> GPU.

    By default FSDP reduce-scatters gradients on EVERY microbatch. With 48 blocks x 8 microbatches
    per rank x 896 MB per block that is ~344 GB of reduce-scatter per optimizer step; over a
    host-mediated cross-socket link (~5-10 GB/s) that is 35-70 s/step, which is where job 389416's
    7.7 h ETA came from, and a single 896 MB collective under that contention blew the 600 s NCCL
    watchdog (timeout on _REDUCE_SCATTER_BASE in post_backward at step ~2).

    Disabling sync for all but the LAST microbatch of a step reduces that to ~43 GB/step (8x less)
    and cuts the collective count 8x, which also shrinks the window for ordering problems.

    The MATH IS UNCHANGED: this is ordinary gradient accumulation — gradients accumulate unsharded
    across microbatches and are reduced once at the end, the same sum in a different order. The cost
    is memory: unsharded fp32 block gradients instead of sharded (~+21 GB/GPU at 12B, taking the
    measured 113.5 GB to ~134 GB against 143 GB). Re-run the equivalence gate after changing this."""
    if not ctx.enabled:
        return
    # NLA_FSDP_NOSYNC=0 forces a sync on EVERY microbatch, trading 8x the communication for 20.4 GB
    # of memory at 12B. Needed because no-sync keeps gradients UNSHARDED across a step, and at 12B
    # 2-way that is what pushes the process over a 143 GB H200: jobs 389819/389847 OOMed at step 2
    # with the process at 139.68 of 139.72 GiB. Without no-sync the footprint is ~118 GB and fits,
    # at ~41 s/step (~7.7 h for an AV stage) on this no-NVLink node.
    if os.environ.get("NLA_FSDP_NOSYNC", "1") == "0":
        enabled = True
    blocks = getattr(getattr(model, "model", model), "layers", None)
    if blocks is None:
        return
    for b in blocks:
        fn = getattr(b, "set_requires_gradient_sync", None)
        if fn is not None:
            fn(enabled)


def replicated_params(model: torch.nn.Module, ctx: DistCtx, extra_params=()) -> list[torch.nn.Parameter]:
    """Trainable params NOT managed by FSDP (plain tensors, i.e. everything outside the blocks).

    Identified by not being a DTensor, which is exactly FSDP2's marker for a sharded parameter."""
    if not ctx.enabled:
        return []
    from torch.distributed.tensor import DTensor
    out = [p for p in model.parameters() if p.requires_grad and not isinstance(p.data, DTensor)]
    out += [p for p in extra_params if p.requires_grad and not isinstance(p.data, DTensor)]
    return out


def reduce_replicated_grads(params, ctx: DistCtx) -> None:
    """Average the gradients of replicated params across ranks — with a DATA-INDEPENDENT schedule.

    FSDP reduce-scatters block gradients with a MEAN, and `scale_for_world` is calibrated against
    that mean, so replicated params must be averaged too for the two halves of the model to receive
    consistent gradients. Without this the embedding would see only the local rank's microbatches.

    CRITICAL: every rank must issue EXACTLY the same number of collectives in the same order.
    The first version skipped params whose `.grad` was None:

        for p in params:
            if p.grad is not None:          # <-- count can differ between ranks
                dist.all_reduce(p.grad, ...)

    Ranks process different microbatches, so a replicated parameter can receive a gradient on one
    rank and not the other; the counts then diverge and the ranks deadlock. That is what killed job
    389569: a watchdog timeout on `OpType=ALLREDUCE, NumelIn=1` — four bytes, so not bandwidth, but
    one rank waiting at a collective the other never issued. Materialising a zero gradient makes the
    schedule depend only on the parameter list, which is identical everywhere."""
    if not ctx.enabled or not params:
        return
    import torch.distributed as dist
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)     # keeps the collective count data-independent
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def scale_for_world(loss: torch.Tensor, ctx: DistCtx) -> torch.Tensor:
    """Cancel FSDP's gradient averaging — see THE NORMALIZER TRAP above.

    The loops normalise by a *global* batch quantity, so each rank's contribution must be
    multiplied by `world` for the mean-reduced gradient to equal the single-GPU gradient."""
    return loss * ctx.world if ctx.enabled else loss


def rank_microbatches(n_micro: int, ctx: DistCtx) -> list[int]:
    """Microbatch indices this rank owns: a strided partition, so the union over ranks is exactly
    `range(n_micro)` with no overlap and sizes differing by at most one."""
    if not ctx.enabled:
        return list(range(n_micro))
    # Every microbatch triggers FSDP collectives (all-gather in forward, plus recompute), so an
    # UNEVEN split desyncs the ranks and deadlocks them. With global_batch 128 / micro 8 the training
    # loop has 16 microbatches and the holdout loops 108 and 106 -- all even at world=2 -- but a
    # different batch size or world size could make it odd, and the symptom would be a 10-minute
    # watchdog hang rather than an error. Fail loudly instead. (Job 389569 was the same class of bug
    # in reduce_replicated_grads: a data-dependent collective count.)
    if n_micro % ctx.world:
        raise RuntimeError(
            f"microbatch count {n_micro} is not divisible by world {ctx.world}: ranks would issue "
            f"different numbers of FSDP collectives and hang. Adjust global_batch/micro_batch so "
            f"global_batch/micro_batch is a multiple of {ctx.world}.")
    return list(range(ctx.rank, n_micro, ctx.world))


def all_reduce_sum(*vals: float, ctx: DistCtx) -> tuple[float, ...]:
    """Sum scalars across ranks (used for held-out loss accumulators, which must be global:
    a per-rank holdout number would silently change the liveness rule (a)/(b) inputs)."""
    if not ctx.enabled:
        return tuple(float(v) for v in vals)
    import torch.distributed as dist
    t = torch.tensor(list(vals), dtype=torch.float64, device=f"cuda:{ctx.local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return tuple(t.tolist())


def all_gather_tensor(t: torch.Tensor, ctx: DistCtx) -> torch.Tensor:
    """Concatenate a per-rank tensor along dim 0 across all ranks, in rank order.

    The AR held-out block needs the COMPLETE prediction/target matrices to compute `fve`, its
    shuffled control (a permutation over the *global* holdout) and the train-mean baseline. It
    cannot simply be run on rank 0: FSDP all-gathers parameters inside forward, which is a
    collective, so a single-rank forward would hang. Every rank therefore evaluates its shard and
    the results are gathered here (holdout is ~850 x d, i.e. megabytes -- cheap)."""
    if not ctx.enabled:
        return t
    import torch.distributed as dist
    dev = f"cuda:{ctx.local_rank}"
    n = torch.tensor([t.shape[0]], device=dev)
    sizes = [torch.zeros_like(n) for _ in range(ctx.world)]
    dist.all_gather(sizes, n)
    sizes = [int(x.item()) for x in sizes]
    src = t.to(dev).contiguous()
    bufs = [torch.empty((sz, *t.shape[1:]), dtype=src.dtype, device=dev) for sz in sizes]
    dist.all_gather(bufs, src)
    return torch.cat([b.cpu() for b in bufs], dim=0)


def gather_full_state_dict(model: torch.nn.Module, ctx: DistCtx) -> dict | None:
    """Full (unsharded) fp32->caller's-dtype state dict on rank 0, None elsewhere.

    Needed because `save_pretrained` on a sharded model would otherwise write DTensor shards."""
    if not ctx.enabled:
        return model.state_dict()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if not ctx.is_main:
        return None
    # Drop tied duplicates. `save_pretrained` de-duplicates tied weights itself, but ONLY when it
    # builds the state dict; handing it an explicit one bypasses that. Gemma3 sets
    # tie_word_embeddings=True, so without this the checkpoint gains a redundant lm_head.weight:
    # measured 2026-09-10 on the 4B gate (job 389317) as 445 tensors / 9.10 GB against the
    # single-GPU path's 444 / 7.76 GB -- 1.34 GB of duplicate embedding, and 2.0 GB per layer at
    # d=3840, i.e. ~96 GB across a 48-layer 12B run, in a layout unlike the 33 existing 4B pairs.
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    return sd


def clip_grad_norm(params, max_norm: float, ctx: DistCtx) -> torch.Tensor:
    """One GLOBAL gradient norm over a mix of sharded (DTensor) and replicated (plain) grads.

    `torch.nn.utils.clip_grad_norm_` cannot do this: handed both kinds it dies with
        "aten._foreach_norm.Scalar: got mixed torch.Tensor and DTensor"
    (observed 2026-09-10, job 389310). Clipping the two groups separately would be two local norms
    instead of the single global norm the single-GPU recipe computes — a silent change to training,
    so the norm is assembled explicitly instead:

        sharded part   : each rank contributes its own shard's sum of squares, SUM-reduced so every
                         rank sees the whole thing.
        replicated part: identical on every rank (reduce_replicated_grads already averaged it), so
                         it is counted ONCE per rank and must NOT be reduced again.

    The clip coefficient then matches single-GPU, and the returned norm is the true global value
    (it is what gets logged as grad_norm, which the equivalence gate compares at step 1)."""
    plist = [p for p in (params if isinstance(params, (list, tuple)) else list(params)) if p.grad is not None]
    if not ctx.enabled:
        return torch.nn.utils.clip_grad_norm_(plist, max_norm)
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    dev = f"cuda:{ctx.local_rank}"
    # Accumulate as Python floats, not as a CUDA scalar: under NLA_FSDP_CPU_OFFLOAD the sharded
    # gradients are HOST tensors, and adding those into a CUDA accumulator raises. Floats also keep
    # the reduction device-agnostic for the mixed sharded/replicated case.
    sharded_sq = 0.0
    rep_sq = 0.0
    for p in plist:
        g = p.grad
        if isinstance(g, DTensor):
            sharded_sq += float(g.to_local().float().pow(2).sum())
        else:
            rep_sq += float(g.float().pow(2).sum())
    t = torch.tensor([sharded_sq], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)               # replicated part deliberately excluded
    total_f = float((t.item() + rep_sq) ** 0.5)
    coef = max_norm / (total_f + 1e-6)
    if coef < 1.0:
        for p in plist:
            g = p.grad
            if isinstance(g, DTensor):
                g.to_local().mul_(coef)
            else:
                g.mul_(coef)
    return torch.tensor(total_f)
