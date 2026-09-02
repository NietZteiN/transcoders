#!/bin/bash
# How big does the generation budget need to be?
#
# 100% of unparsed replies sit exactly at the 2048 cap while the median parsed reply is ~520
# tokens, so censoring is truncation, not failure to answer. But those items ran 4x the median
# without finishing, so they may be in a degenerate loop no budget rescues. Only 49 distinct
# items are ever censored, so re-running exactly those at 8192 answers the question directly
# instead of guessing a new cap and discovering later that it was still too small.
#SBATCH --job-name=censor_probe
#SBATCH --partition=h200,h100
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_censor_probe.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_censor_probe.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

HOSTS=(gemma12b llama8b)
H=${HOSTS[$SLURM_ARRAY_TASK_ID]}
echo "# censor probe · host $H · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"

for T in L0 L1 L1b L2 L3; do
  SN=$(python3 -c "
import json;d=json.load(open('data/nla/p0/p1b/censored_items.json'))
print(','.join(d.get('$H/$T', [])))")
  [ -n "$SN" ] || { echo \"$T: no censored items\"; continue; }
  echo "===== $H $T : $(echo $SN | tr ',' '\n' | wc -l) item(s) at 8192 ====="
  python nla/src/p1b_ladder.py --host "$H" --tier "$T" --draws 1 --max-new-gen 8192 \
    --snippets "$SN" --out-dir "$PROJ/data/nla/p0/p1b/censor_probe_$H"
done
