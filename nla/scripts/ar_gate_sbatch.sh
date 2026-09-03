#!/bin/bash
#SBATCH --job-name=ar_gate
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_ar_gate.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_ar_gate.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "# Gemma AR gate · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ar_gate.py nla/src/steer_run.py
python nla/src/ar_gate.py --host gemma12b
