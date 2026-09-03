#!/bin/bash
#SBATCH --job-name=kv_gate
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_kv_gate.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_kv_gate.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "# KV-bypass gate · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
python nla/src/kv_bypass_gate.py --host gemma12b \
  --out /work/jvl210002/migration/transcoders/data/nla/p0/steerv2/gemma12b/kv_bypass_gate.json
