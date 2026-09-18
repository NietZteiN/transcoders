#!/bin/bash
#SBATCH --job-name=nla_r28_arm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r28_%x.out
# H-R28b -- one model on the FLATTENED corpus (102 PACKS-PAIRED snippets / 1188 cases), greedy,
# unsteered. The L0 side is already banked from H-R23, re-scored on this subset.
# Pre-registered: log/nla-harness/2026-09-18_flatten-prereg.md
# $1 = model id
set -uo pipefail
M="${1:?model}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/flat_greedy; mkdir -p "$OUT"
TAG=$(echo "$M" | tr '/' '_')
echo "# H-R28b model=$M cond=flat · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py nla/src/ase_flatten_java.py
nvidia-smi --query-gpu=index,name --format=csv,noheader
python nla/src/ase_steer_run.py --packs $F/packs_flat_paired.jsonl --arm unsteered --greedy --runs 1 \
  --model-id "$M" --chat-template --vectors-config nla/configs/ase_vectors_full.yaml \
  --out "$OUT/$TAG.flat.jsonl" --max-hours 3.5; rc=$?
echo "# rows: $(wc -l < "$OUT/$TAG.flat.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
