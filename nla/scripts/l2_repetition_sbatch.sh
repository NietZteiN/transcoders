#!/bin/bash
#SBATCH --job-name=l2_repctl
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_l2_repctl.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_l2_repctl.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
echo "===== REPLICATION draws 10-14, baseline now includes repetition ====="
python nla/src/p1b_l2_mechanism.py --root data/nla/p0/p1b/ladder_llama8b --tiers L2,L3 \
  --draws 10-14 --out data/nla/p0/p1b/llama_l2_replication_rep.json
echo
echo "===== DISCOVERY draws 0-9, same baseline ====="
python nla/src/p1b_l2_mechanism.py --root data/nla/p0/p1b/ladder_llama8b --tiers L2,L3 \
  --draws 0-9 --out data/nla/p0/p1b/llama_l2_discovery_rep.json
echo
echo "===== all three hosts, all draws, for the full matrix ====="
for pair in "qwen7b ladder" "gemma12b ladder_gemma12b" "llama8b ladder_llama8b"; do
  set -- $pair
  python nla/src/p1b_l2_mechanism.py --root "data/nla/p0/p1b/$2" --tiers L2,L3,L1b \
    --out "data/nla/p0/p1b/repctl_$1.json"
done
