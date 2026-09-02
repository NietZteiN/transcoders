#!/bin/bash
# The read-side battery on all three hosts, now that the censoring confound is removed and
# non-terminating rows are excluded rather than silently scored wrong.
#SBATCH --job-name=battery
#SBATCH --partition=normal
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_battery.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_battery.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
HOSTS=(qwen7b gemma12b llama8b)
DIRS=(ladder ladder_gemma12b ladder_llama8b)
H=${HOSTS[$SLURM_ARRAY_TASK_ID]}; D=${DIRS[$SLURM_ARRAY_TASK_ID]}
R="$PROJ/data/nla/p0/p1b/$D"
echo "# battery · $H · $R · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "===== ladder score (selection-free mean rho, own length baseline per tier) ====="
python nla/src/p1b_ladder_score.py --root "$R" --out "$PROJ/data/nla/p0/p1b/battery_${H}_score.json"
echo "===== permutation nulls over mean AND max statistics ====="
python nla/src/p1b_ladder_null.py --root "$R" --perm-tiers L1b,L2,L3 \
  --out "$PROJ/data/nla/p0/p1b/battery_${H}_null.json"
