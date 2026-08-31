#!/bin/bash
# One unsteered baseline pass that records BOTH tiers' reply lengths.
# Closes the limitation on the 2026-08-30 tier result: no L0 length baseline existed on disk.
# Budget 2048 to match the position/depth run whose lengths the L1b baseline came from, so the
# two tiers' baselines are computed from the same generation regime.
#SBATCH --job-name=p1b_baselen
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_baselen.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_baselen.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="$PROJ/data/nla/p0/p1b/baselen"
mkdir -p "$OUT"
echo "# p1b baseline lengths · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader || true
python nla/src/steer_run.py --layer 20 --baseline-only --max-new-gen 2048 \
  --out-dir "$OUT" --max-hours 1.5
echo "# exit $? · $(date -u +%FT%TZ)"
