#!/bin/bash
#SBATCH --job-name=nla12b_layer
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G   # host holds the offloaded sharded state (~82 GB x 2 ranks = 164 GB); node caps at 375G
#SBATCH --time=40:00:00   # measured 114 s/step with CPU offload -> AV 21.5 h, + AR ~4 h (h200 max 2 days)
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla12b_L%a.out
# Phase C: one 12B NLA pair (AV -> AR -> check) at layer $LAYER, pre-registered in
# log/nla-harness/2026-09-10_12b-budget-prereg.md.
#
# --fsdp is MANDATORY here: 11.2 B params x 16 bytes (fp32 weights+grads+Adam m/v) is 179 GB +
# activations vs 143 GB on one H200. Block-only sharding with the tied embedding replicated gives
# ~112 GB/GPU at 2-way -- which is why this asks for 2 H200s and NOT an 80 GB H100 (2-way would not
# fit; a 4-way H100 run needs g-04-02, held by a 22 h job at time of writing).
# Equivalence to the single-GPU recipe verified in
# log/nla-harness/2026-09-10_fsdp-equivalence-gate.md (step-1 loss 1.9e-16, grad_norm ratio 1.0000,
# checkpoint parity). FSDP here buys FEASIBILITY, not speed: measured w=2 throughput is 0.70x of one
# GPU, so expect AV ~2.3-4.0 h and (at L32's 33-layer trunk) AR ~1.0-1.5 h.
#
# The `check` stage is deliberately NOT run under torchrun: it only generates 96 reads from the
# saved AV+AR (~41 GB bf16), which fits a single GPU.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS=8
# These nodes have NO NVLink (topo SYS, cross-socket NUMA 3/4 -- job 389475), so collectives are
# host-mediated and slow. Gradient-accumulation no-sync cuts traffic 8x, but keep a generous
# watchdog: the default 600 s is what job 389416 tripped on a single 896 MB reduce-scatter.
# PYTHONUNBUFFERED is essential here: jobs 389416/389569/389666 all ended with rank 0 sitting in
# dist_cleanup's barrier for the full NCCL timeout, and the Python exception that sent it there
# was LOST because the watchdog kills the process with SIGABRT while stdout is still buffered.
# Three attempts were spent diagnosing the barrier (a symptom) instead of the real error.
export PYTHONUNBUFFERED=1
# ROOT CAUSE of jobs 389416-389819 (six attempts): not communication -- a plain CUDA OOM at
# step 2 inside av_loss. `torch.cuda.max_memory_allocated()` read 133.7 GB (my prediction was
# 134), but that metric EXCLUDES allocator fragmentation and non-PyTorch memory: the process was
# actually at 139.68 of 139.72 GiB and died on a 30 MiB request. The gap is mostly NCCL's device
# and pinned staging buffers, which are large here because the node has no NVLink (topo SYS) so
# every collective is host-mediated across 48 blocks.
# Two independent mitigations:
#   expandable_segments  - reclaims fragmentation the caching allocator would otherwise strand
#   MB=4 (below)         - halves activation memory; the global batch stays 128, so this is the
#                          same gradient accumulation with more, smaller microbatches (32 -> 16
#                          per rank, still even, which rank_microbatches requires)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NLA_NCCL_TIMEOUT_SEC is read by fsdp_util.dist_setup and passed to init_process_group();
# TORCH_NCCL_WATCHDOG_TIMEOUT_SEC is NOT honoured (389569 still showed Timeout(ms)=600000).
# Keep it short now: a long timeout just delays the real traceback by an hour.
# no-sync OFF: it costs 20.4 GB of unsharded gradients, which is exactly what OOMed 389819/389847.
# With per-microbatch sync the footprint is ~118 GB (fits) but traffic is 8x -> ~41 s/step, so the
# watchdog must be generous: attempt 1 (389416) ran step 1 fine and died only on the 600 s default.
# CPU OFFLOAD is what makes 12B 2-way fit at all: it moves the 81.6 GB of sharded fp32 block state
# (params 20.4 + grads 20.4 + Adam m/v 40.8) to host memory, leaving the GPU with the replicated
# embedding, transient all-gathered blocks and activations. Without it the floor is 97.7 GB and
# ~24 GB of NCCL/allocator overhead leaves ~18 GB for activations -> OOM (389819/389847/389859).
# NOSYNC stays OFF: it would add 20.4 GB of unsharded grads, and with offload the traffic cost is
# no longer the binding constraint.
export NLA_FSDP_CPU_OFFLOAD=1 NLA_FSDP_NOSYNC=0 NLA_NCCL_TIMEOUT_SEC=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=SYS NCCL_DEBUG=WARN
K=${LAYER:?set LAYER}; CFG=nla/configs/nla_ml_12b.yaml; MB=${MB:-8}
echo "# 12B layer $K · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_train.py nla/src/fsdp_util.py "$CFG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_fsdp_util.py -q -p no:faulthandler || exit 1

echo; echo "=== sft_av L$K (FSDP, 2 GPUs) ==="
torchrun --standalone --nproc_per_node=2 nla/src/nla_train.py --stage sft_av --layer "$K" \
  --config "$CFG" --fsdp --micro-batch $MB || exit 1
echo; echo "=== sft_ar L$K (FSDP, 2 GPUs) ==="
torchrun --standalone --nproc_per_node=2 nla/src/nla_train.py --stage sft_ar --layer "$K" \
  --config "$CFG" --fsdp --micro-batch $MB || exit 1
echo; echo "=== check L$K (single GPU) ==="
python nla/src/nla_train.py --stage check --layer "$K" --config "$CFG" || exit 1
echo "# done L$K $(date -u +%FT%TZ)"
