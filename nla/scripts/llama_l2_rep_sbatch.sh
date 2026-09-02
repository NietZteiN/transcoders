#!/bin/bash
# Pre-registered replication of the one surviving cell.
# Rule frozen in log/nla-harness/2026-09-02_llama-l2-prereg.md BEFORE these draws existed:
#   REPLICATED iff beats-strict >= +0.075 with permutation p < 0.05, on draws 10-14 ONLY.
#   Discovery value +0.1507, so the bar is half of it. L3 runs as the sibling control.
#SBATCH --job-name=llama_l2_rep
#SBATCH --partition=h200,h100,a30
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_llama_l2_rep.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_llama_l2_rep.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TIERS=(L2 L2 L3 L3); STARTS=(10 13 10 13); NDRAWS=(3 2 3 2)
T=${TIERS[$SLURM_ARRAY_TASK_ID]}; S=${STARTS[$SLURM_ARRAY_TASK_ID]}; N=${NDRAWS[$SLURM_ARRAY_TASK_ID]}
echo "# llama L2 replication · tier $T draws $S..$((S+N-1)) · $SLURMD_NODENAME"
python nla/src/p1b_ladder.py --host llama8b --tier "$T" --draws "$N" --draw-start "$S" \
  --max-new-gen 8192
