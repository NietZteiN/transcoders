#!/bin/bash
# Regenerate ONLY the ever-censored items, at a budget the probe showed is sufficient.
#
# WHY THIS IS NOT REGIME-MIXING. Greedy decoding stops at EOS; max_new_tokens only bounds the
# loop. Any generation that terminated before 2048 is therefore token-identical under a larger
# cap — raising the budget cannot change it. So only the censored items are affected, and
# regenerating just those is mathematically equivalent to regenerating all 6,000 at the new
# budget, at 1/12th the cost. Whole items are redone (all 10 draws), never individual draws, so
# no item ends up straddling two budgets.
#
# Probe results that set the budget: Gemma 17/17 finished, max 4,154 tokens. Llama 17/32
# finished; the remaining 15 loop indefinitely and no budget rescues them — they are handled as
# an explicit non-terminating category downstream, not silently as wrong answers.
#SBATCH --job-name=censor_fix
#SBATCH --partition=h200,h100
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_censor_fix.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_censor_fix.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

HOSTS=(gemma12b llama8b)
H=${HOSTS[$SLURM_ARRAY_TASK_ID]}
echo "# censor fix · host $H · budget 8192 · job $SLURM_JOB_ID on $SLURMD_NODENAME"

for T in L0 L1 L1b L2 L3; do
  SN=$(python3 -c "
import json;d=json.load(open('data/nla/p0/p1b/censored_items.json'))
print(','.join(d.get('$H/$T', [])))")
  [ -n "$SN" ] || continue
  echo "===== $H $T : $(echo $SN | tr ',' '\n' | wc -l) item(s) x 10 draws at 8192 ====="
  python nla/src/p1b_ladder.py --host "$H" --tier "$T" --draws 10 --max-new-gen 8192 \
    --snippets "$SN" --out-dir "$PROJ/data/nla/p0/p1b/censor_fix_$H"
done
