#!/bin/bash
#SBATCH --job-name=l2_rep_score
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_l2_rep_score.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_l2_rep_score.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
echo "===== REPLICATION: draws 10-14 only (frozen rule: beats-strict >= +0.075, p < 0.05) ====="
python nla/src/p1b_l2_mechanism.py --root data/nla/p0/p1b/ladder_llama8b --tiers L2,L3 \
  --draws 10-14 --out data/nla/p0/p1b/llama_l2_replication.json
echo
echo "===== DISCOVERY for reference: draws 0-9 ====="
python nla/src/p1b_l2_mechanism.py --root data/nla/p0/p1b/ladder_llama8b --tiers L2,L3 \
  --draws 0-9 --out data/nla/p0/p1b/llama_l2_discovery.json
