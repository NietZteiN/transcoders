#!/bin/bash
#SBATCH --job-name=p1b_pdL2
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_pdL2.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_pdL2.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "# position x depth on L2 · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader || true
python nla/src/p1b_position_depth.py --tier L2
