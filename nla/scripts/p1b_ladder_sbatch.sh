#!/bin/bash
#SBATCH --job-name=p1b_ladder
#SBATCH --partition=h200
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_p1b_ladder.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_p1b_ladder.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TIERS=(L0 L1 L1b L2 L3)
T=${TIERS[$SLURM_ARRAY_TASK_ID]}
echo "# ladder $T · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "# sha256 $(sha256sum nla/src/p1b_ladder.py | cut -c1-16)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader || true
python nla/src/p1b_ladder.py --tier "$T" --draws 10
