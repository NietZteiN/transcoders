#!/bin/bash
#SBATCH --job-name=nla_cputest
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_cputest.out
# CPU-only pytest runner: the login node's 8 GB virtual-memory cap cannot hold torch + a toy Llama.
# $@ = pytest args (file paths / -k)
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
echo "# cpu tests · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ) · args: $*"
python -m pytest "$@" -q 2>&1 | tail -30
