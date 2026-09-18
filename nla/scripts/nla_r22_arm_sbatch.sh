#!/bin/bash
#SBATCH --job-name=r22_arm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r22_%x.out
# H-R22b -- one GREEDY arm on the DERANGED corpus (146 PACKS-PAIRED snippets / 1,722 cases).
# The L0 side is already banked (H-R18 greedy original_unsteered), so the damage costs one short job.
# Pre-registered: log/nla-harness/2026-09-18_adversarial-rename-prereg.md
# $1 = arm
set -uo pipefail
ARM="${1:?usage: sbatch nla_r22_arm_sbatch.sh <arm>}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/swap_greedy; mkdir -p "$OUT"
echo "# H-R22b greedy arm=$ARM · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py nla/src/ase_rename_swap.py
nvidia-smi --query-gpu=index,name --format=csv,noheader
python nla/src/ase_steer_run.py --packs $F/packs_swap_paired.jsonl --arm "$ARM" --greedy --runs 1 \
  --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
  --vectors-config nla/configs/ase_vectors_full.yaml --out "$OUT/$ARM.jsonl" --max-hours 3.5; rc=$?
echo "# rows: $(wc -l < "$OUT/$ARM.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
