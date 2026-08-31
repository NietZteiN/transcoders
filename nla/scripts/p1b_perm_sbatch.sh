#!/bin/bash
#SBATCH --job-name=p1b_perm
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_perm.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p1b_perm.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
echo "# p1b corrected permutation control · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
python nla/src/p1b_perm_control.py --layers 13,20
