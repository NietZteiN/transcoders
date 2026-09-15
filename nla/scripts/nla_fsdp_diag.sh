#!/bin/bash
#SBATCH --job-name=fsdp_diag
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_fsdp_diag.out
# Why job 389416 (12B L32) died: NCCL watchdog timeout on FSDP's own gradient reduce-scatter
# (_REDUCE_SCATTER_BASE, 224M elements = 896 MB, one 12B block) inside post_backward at step ~2.
# Rank 1 had enqueued 832 collectives, rank 0 had not reached 822 -> rank 0 stalled 600 s.
#
# Two candidate causes, distinguished here:
#   (a) NO WORKING PEER LINK. 48 blocks x 8 microbatches x 896 MB = ~344 GB of reduce-scatter per
#       step. Over NVLink (~400 GB/s) that is ~1 s; over PCIe/host (~10 GB/s) ~34 s, consistent with
#       the 7.7 h ETA -- but a SINGLE 896 MB collective exceeding 600 s means ~1.5 MB/s, a stall.
#   (b) ORDERING DEADLOCK from FSDP + `use_reentrant=False` activation checkpointing: recompute
#       during backward re-triggers all-gather, and the two ranks can schedule them differently.
# Step 1 of both the 4B gate (34 blocks, 20 steps, PASSED) and this 12B run completed, so whatever
# it is, it bites later and at scale.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
echo "# fsdp diag · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo; echo "=== (1) GPU topology: is there a peer link? ==="
nvidia-smi topo -m 2>&1 | head -8
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1 | head -4
echo; echo "=== (2) raw NCCL bandwidth, 896 MB reduce-scatter x5 (the exact failing shape) ==="
torchrun --standalone --nproc_per_node=2 - <<'PY'
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); w = dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
n = 224148992
inp = torch.ones(n, dtype=torch.float32, device="cuda")
out = torch.empty(n // w, dtype=torch.float32, device="cuda")
for i in range(5):
    torch.cuda.synchronize(); t = time.time()
    dist.reduce_scatter_tensor(out, inp)
    torch.cuda.synchronize()
    if r == 0:
        dt = time.time() - t
        print(f"  reduce_scatter {n*4/1e9:.2f} GB: {dt*1000:.1f} ms -> {n*4/1e9/dt:.1f} GB/s", flush=True)
dist.destroy_process_group()
PY
echo; echo "=== (3) 12B AV, 3 steps, NCCL_DEBUG=WARN, 30 min timeout ==="
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN TORCH_NCCL_TRACE_BUFFER_SIZE=2000
torchrun --standalone --nproc_per_node=2 nla/src/nla_train.py --stage sft_av --layer 32 \
  --config nla/configs/nla_ml_12b.yaml --fsdp --max-steps 3 \
  --root /scratch/juno/jvl210002/fsdp_diag_12b 2>&1 | grep -vE "^\[train\] step=(4|5|6)" | tail -25
echo "# done $(date -u +%FT%TZ)"
