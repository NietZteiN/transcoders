#!/bin/bash
#SBATCH --job-name=p1b_null
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_null.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_null.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
echo "# ladder selection-free null · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
python nla/src/p1b_ladder_null.py --perm-tiers L1b,L2,L3
