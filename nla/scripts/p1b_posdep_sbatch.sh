#!/bin/bash
#SBATCH --job-name=p1b_posdep
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_posdep.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_posdep.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "# p1b read probe · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "# sha256 $(sha256sum nla/src/p1b_position_depth.py | cut -c1-16)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader || true
echo
python nla/src/p1b_position_depth.py
