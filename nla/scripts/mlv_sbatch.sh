#!/bin/bash
#SBATCH --job-name=mlv
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_mlv.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_mlv.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "# per-layer V3/V4 bank + energy match · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/multilayer_vectors.py nla/src/steer_multilayer.py
python nla/src/multilayer_vectors.py --host gemma12b
