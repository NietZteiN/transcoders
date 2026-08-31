#!/bin/bash
# P0.4-det — can the per-item reproducibility floor be LOWERED?
#
# 2026-08-29 measured the floor at 0.85-0.90 per-item agreement between two runs of the identical
# command on the SAME physical card. That entry attributed it to float kernel execution rather
# than to anything a seed controls. This tests the claim directly: run the same replicate pair
# again with cuBLAS/cuDNN/SDPA pinned to deterministic kernels.
#
#   D1, D2 same job => same allocated card, --deterministic
#   compare D1<->D2 (deterministic same-card) against the banked A1<->A2 (0.900 / 0.850)
#
# PRE-STATED READING, frozen before the run:
#   D1<->D2 == 1.000   the floor is autotuning, it is a CHOICE, and every future paired run
#                      should set --deterministic.
#   still < 1.000      something structural survives kernel pinning; the floor is a fact about
#                      the pipeline and must be quoted in every paired per-item claim.
# Also reported: D1 vs A1 — whether deterministic mode CHANGES the answers, not just stabilises
# them. If it does, deterministic runs are not comparable with the banked corpus.
#
#SBATCH --job-name=p04_det
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_det.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_det.out
set -uo pipefail

source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# MUST be exported before python starts: cuBLAS reads it when it creates its handle, and setting
# it later is silently ineffective. steer_run.py refuses --deterministic if it is unset.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "# p04-det · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "# CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader || true
echo

for REP in D1 D2; do
  OUT="$PROJ/data/nla/p0/p04/repro/$REP"
  mkdir -p "$OUT"
  echo "=== replicate $REP -> $OUT · $(date -u +%FT%TZ)"
  python nla/src/steer_run.py \
    --layer 20 \
    --positions last_prompt \
    --alphas 1.0 \
    --only-conditions V4_oracle \
    --deterministic \
    --out-dir "$OUT" \
    --max-hours 2.0
  echo "=== replicate $REP exit $? · $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$OUT/gpu.txt"
  echo "$SLURMD_NODENAME" > "$OUT/node.txt"
done
