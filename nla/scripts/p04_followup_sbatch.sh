#!/bin/bash
# P0.4 follow-ups, back to back on ONE card so both comparisons are same-card by construction.
#
#   steps 1-2  read reproducibility (R1, R2)  — is the floor a GENERATION problem only?
#   steps 3-4  generation with MAX_NEW_GEN raised 1100 -> 2048 (E1, E2) — is the floor
#              truncation censoring? 3 of 10 disagreeing items had one run emit no answer line.
#
# PRE-STATED READING, frozen before the run:
#   reads   R1<->R2 bit-identical  => the floor is generation-only; read-side measures (rt_cos,
#           act_norm, the P0.1 curves) carry no caveat. Any drift => it is pipeline-wide.
#   gens    E1<->E2 agreement materially above the banked A1<->A2 (0.900 baseline / 0.850 steered)
#           => truncation censoring was a real component and MAX_NEW_GEN should be raised for
#           future runs. Comparable => censoring is not the mechanism and the floor stands.
# Default mode throughout (NOT --deterministic): that flag was shown to fork the corpus.
#
#SBATCH --job-name=p04_follow
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_follow.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_follow.out
set -uo pipefail

source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "# p04-followup · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader || true
echo "$SLURMD_NODENAME" > "$PROJ/data/nla/p0/p04/followup_node.txt"
echo

for TAG in R1 R2; do
  echo "=== read-repro $TAG · $(date -u +%FT%TZ)"
  python nla/src/p04_read_repro.py --layer 20 --tag "$TAG"
  echo "=== read-repro $TAG exit $?"
done

for REP in E1 E2; do
  OUT="$PROJ/data/nla/p0/p04/repro/$REP"
  mkdir -p "$OUT"
  echo "=== gen replicate $REP (max-new-gen 2048) -> $OUT · $(date -u +%FT%TZ)"
  python nla/src/steer_run.py \
    --layer 20 \
    --positions last_prompt \
    --alphas 1.0 \
    --only-conditions V4_oracle \
    --max-new-gen 2048 \
    --out-dir "$OUT" \
    --max-hours 2.5
  echo "=== gen replicate $REP exit $? · $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$OUT/gpu.txt"
  echo "$SLURMD_NODENAME" > "$OUT/node.txt"
done
