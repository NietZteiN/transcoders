#!/bin/bash
# P0.4-repro — how reproducible is a per-item outcome, actually?
#
# WHY. P0.4 found greedy bf16 decoding agrees only 0.83-0.95 per item across cards while marginal
# accuracies match to <=0.033. Every paired test in this programme treats per-item correctness as
# a fixed property of (item, condition): B4's primary gate (V1 vs V3, McNemar p = 1.00), N13's
# matched 2x2, and P0.4's own criterion (a). None carries an error bar for this. This job measures
# the floor so those tables can quote one.
#
# DESIGN, frozen before the run. A replicate = the full unsteered baseline (60 L0 + 60 L1b) plus
# V4_oracle at alpha = 1.0 on all 60 items, layer 20, last_prompt, seed 20260724 — everything
# identical to the P0.4 L20 arm.
#   A1, A2  same job => same allocated GPU  => run-to-run nondeterminism
#   B1      a different node                => card-to-card nondeterminism
# Reported: per-item agreement A1<->A2 and A1<->B1, separately for the unsteered baseline and for
# the steered condition, alongside the marginal accuracies. No decision rule — this is calibration,
# not a hypothesis test. The pre-declared use is that the CROSS-NODE figure becomes the floor any
# paired per-item claim in this programme must report.
#
#SBATCH --job-name=p04_repro
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_repro.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_repro.out
set -uo pipefail

source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

REPS="${REPS:?set REPS to a space-separated list of replicate tags}"

echo "# p04-repro reps [$REPS] · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader || true
echo

for REP in $REPS; do
  OUT="$PROJ/data/nla/p0/p04/repro/$REP"
  mkdir -p "$OUT"
  echo "=== replicate $REP -> $OUT · $(date -u +%FT%TZ)"
  # Each replicate gets its own out-dir: steer_run resumes from steer_results.jsonl, so sharing
  # one would make the second replicate a no-op and report perfect agreement with itself.
  python nla/src/steer_run.py \
    --layer 20 \
    --positions last_prompt \
    --alphas 1.0 \
    --only-conditions V4_oracle \
    --out-dir "$OUT" \
    --max-hours 1.5
  echo "=== replicate $REP exit $? · $(date -u +%FT%TZ)"
  # Record which physical card produced it — the whole point of the measurement.
  nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$OUT/gpu.txt"
  echo "$SLURMD_NODENAME" > "$OUT/node.txt"
done
