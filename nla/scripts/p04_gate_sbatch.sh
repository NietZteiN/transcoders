#!/bin/bash
# P0.4 validation gate — must PASS before the three depth arms are believed.
# See log/nla-harness/2026-08-28_p04-depth-prereg.md, "Two validation gates", gate 1.
#SBATCH --job-name=p04_gate
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_gate.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_p04_gate.out
set -uo pipefail

source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"

# Compute nodes have no route out; every weight this touches is already in HF_HOME.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "# p04 gate · job $SLURM_JOB_ID on $SLURMD_NODENAME ($SLURM_JOB_PARTITION) · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "# sha256 $(sha256sum nla/src/p04_gate.py | cut -c1-16) nla/src/p04_gate.py"
echo

python nla/src/p04_gate.py --layers 6,13,20 --out "$PROJ/data/nla/p0/p04/p04_gate.json"
