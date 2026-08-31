#!/bin/bash
# P0.4 — the depth arms. Three layers, one GPU each, run as an array so they are independent:
# a failure in one arm does not take the other two with it, and each resumes on its own.
#
# Pre-registration: log/nla-harness/2026-08-28_p04-depth-prereg.md
# DO NOT launch this until data/nla/p0/p04/p04_gate.json says PASS. The gate proves the read
# site and the write site are the same decoder block; without it the depth axis is unlabelled.
#SBATCH --job-name=p04_depth
#SBATCH --partition=h200
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_p04_depth.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_p04_depth.out
set -uo pipefail

source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LAYERS=(6 13 20)
L=${LAYERS[$SLURM_ARRAY_TASK_ID]}
OUT="$PROJ/data/nla/p0/p04/L$(printf %02d "$L")"
mkdir -p "$OUT"

GATE="$PROJ/data/nla/p0/p04/p04_gate.json"
if ! grep -q '"verdict": "PASS"' "$GATE" 2>/dev/null; then
  echo "REFUSING TO RUN: $GATE does not say PASS. See the prereg — a failed indexing gate"
  echo "means P0.4 is not reported at all, so generating its data would be pure waste."
  exit 2
fi

echo "# p04 layer $L · job $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "# node $SLURMD_NODENAME ($SLURM_JOB_PARTITION) · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "# sha256 $(sha256sum nla/src/steer_run.py | cut -c1-16) nla/src/steer_run.py"
echo "# out $OUT"
echo

# --out-dir is passed EXPLICITLY. steer_run's default is data/nla/n12 — the banked B4 run —
# and a run that inherited it would append P0.4 rows into the comparison bank.
#
# Conditions restricted to the three that are pure activation arithmetic: the AR reconstructor
# is trained at layer 20 and has no meaning at 6 or 13, so V1/V2/F/A are not merely omitted for
# cost, they are undefined off-layer. steer_run refuses --layer != 20 with an AR condition.
python nla/src/steer_run.py \
  --layer "$L" \
  --positions last_prompt \
  --alphas 0.25,0.5,1.0,2.0,4.0 \
  --only-conditions V3_taskvec,V4_oracle,R_random \
  --out-dir "$OUT" \
  --max-hours 6.5

rc=$?
echo "# steer_run exit $rc · $(date -u +%FT%TZ)"

# Per-arm scoring only. The cross-layer contrast is p04_score.py and needs all three arms,
# so it is NOT run here — a per-task invocation would race and would score a partial grid.
# --results/--baseline/--out are all explicit: every steer_stats default points at n12, and
# `--out` alone would overwrite the banked B4 steer_stats.json.
python nla/src/steer_stats.py \
  --results "$OUT/steer_results.jsonl" \
  --baseline "$OUT/baseline.jsonl" \
  --out "$OUT/steer_stats.json" \
  --primary-alpha 1.0 > /dev/null && echo "# scored -> $OUT/steer_stats.json"

exit $rc
