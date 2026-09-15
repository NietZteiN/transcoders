#!/bin/bash
#SBATCH --job-name=fsdp_arm
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_fsdp_arm.out
# ONE arm of the FSDP equivalence gate. Split from nla_fsdp_equiv.sh because that script wanted a
# single node with 2 FREE GPUs, and no such node exists right now: every h200 node has at most one
# GPU free (mostly to other users) and the one idle node, g-07-05, is IDLE+RESERVED.
#
# The two arms have very different needs, so they schedule far more easily apart:
#   ARM=A  1 GPU, needs >= ~64 GB  (4B fp32 weights+grads+Adam is 59.5 GB of irreducible state,
#                                   which no micro-batch reduction touches) -> one h200 GPU
#   ARM=B  2 GPUs at ~35 GB each   -> fits g-06-01's 47 GB H100 MIG slices, uncapped partition
#
# Both arms MUST use the same K, STEPS, LIMIT and MB. Arms may land on different Hopper GPU models;
# both run the same bf16 kernels and the declared tolerances concern fp summation order, so that is
# acceptable -- and the arm's actual device is printed so the comparison can record it.
# Compare afterwards with:  python nla/src/fsdp_equiv_cmp.py <A_root> <B_root> <K> <world>
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS=8
ARM=${ARM:?set ARM=A or ARM=B}; K=${K:-2}; STEPS=${STEPS:-20}; LIMIT=${LIMIT:-4000}
NPROC=${NPROC:-2}; MB=${MB:-8}
SRC=/scratch/juno/jvl210002/fsdp_equiv_root
OUT=/scratch/juno/jvl210002/fsdp_equiv
D=$OUT/$( [ "$ARM" = A ] && echo single || echo fsdp2 )
for n in "$SRC/corpus" "$SRC/explain" "$SRC/acts/L$K.npy"; do
  [ -e "$n" ] || { echo "MISSING INPUT $n"; exit 1; }
done
rm -rf "$D"; mkdir -p "$D"
ln -sfn "$SRC/corpus" "$D/corpus"; ln -sfn "$SRC/explain" "$D/explain"; ln -sfn "$SRC/acts" "$D/acts"
echo "# FSDP arm $ARM · job $SLURM_JOB_ID on $SLURMD_NODENAME · K=$K steps=$STEPS mb=$MB · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_train.py nla/src/fsdp_util.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_fsdp_util.py -q -p no:faulthandler || exit 1
if [ "$ARM" = A ]; then
  python nla/src/nla_train.py --stage sft_av --layer $K --limit $LIMIT --max-steps $STEPS \
    --micro-batch $MB --root "$D" || exit 1
else
  torchrun --standalone --nproc_per_node=$NPROC nla/src/nla_train.py --stage sft_av --layer $K \
    --fsdp --limit $LIMIT --max-steps $STEPS --micro-batch $MB --root "$D" || exit 1
fi
echo "# done arm $ARM $(date -u +%FT%TZ)"
