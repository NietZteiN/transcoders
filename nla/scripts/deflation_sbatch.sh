#!/bin/bash
# The deflation controls that turned the Qwen relational positive into a surface statistic,
# now run per host. Two questions, in order:
#   1. does the residual stream beat a baseline that already knows reply length AND the code's
#      static shape?  (on Qwen it added only +0.0245 over that, bar +0.10)
#   2. is the dispatcher-span signal more than text repetition?  (on Qwen five surface counts
#      alone reached rho +0.8346 against the residual's +0.8842)
#SBATCH --job-name=deflation
#SBATCH --partition=normal
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_deflation.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_deflation.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
HOSTS=(qwen7b gemma12b llama8b); DIRS=(ladder ladder_gemma12b ladder_llama8b)
H=${HOSTS[$SLURM_ARRAY_TASK_ID]}; R="$PROJ/data/nla/p0/p1b/${DIRS[$SLURM_ARRAY_TASK_ID]}"
echo "# deflation · $H · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "===== 1. beyond static complexity? ====="
python nla/src/p1b_l2_mechanism.py --root "$R" --tiers L2,L3,L1b \
  --out "$PROJ/data/nla/p0/p1b/deflation_${H}_mechanism.json"
echo "===== 2. beyond size AND repetition? ====="
python nla/src/p1b_span_probe.py --root "$R" --tiers L2,L3,L1b \
  --out "$PROJ/data/nla/p0/p1b/deflation_${H}_span.json"
