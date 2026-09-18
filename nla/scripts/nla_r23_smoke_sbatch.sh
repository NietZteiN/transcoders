#!/bin/bash
#SBATCH --job-name=r23_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r23_smoke.out
# H-R23 feasibility: their SteeredCausalLM is Llama-derived and the chat-template ids gate encodes
# Llama/SentencePiece BOS behaviour. Try each panel model on 2 snippets; a model that fails is recorded
# PROTOCOL-BROKEN and excluded (precedent: StarCoder2 in H-R1), never worked around.
# Pre-registered: log/nla-harness/2026-09-18_panel-damage-prereg.md
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/panel_smoke; mkdir -p "$OUT"
echo "# H-R23 smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
head -2 $F/packs_swap_paired.jsonl > $OUT/smoke_packs.jsonl
for M in google/codegemma-7b-it meta-llama/Llama-3.1-8B-Instruct microsoft/Phi-3.5-mini-instruct codellama/CodeLlama-13b-Instruct-hf; do
  TAG=$(echo "$M" | tr '/' '_')
  echo "=================== $M ==================="
  timeout 1500 python nla/src/ase_steer_run.py --packs $OUT/smoke_packs.jsonl --arm unsteered --greedy --runs 1 \
    --model-id "$M" --chat-template --vectors-config nla/configs/ase_vectors_full.yaml \
    --out "$OUT/$TAG.jsonl" --max-hours 0.2
  echo "### $M rc=$? rows=$(wc -l < "$OUT/$TAG.jsonl" 2>/dev/null || echo 0)"
done
echo "=== VERDICT ==="
for M in google/codegemma-7b-it meta-llama/Llama-3.1-8B-Instruct microsoft/Phi-3.5-mini-instruct codellama/CodeLlama-13b-Instruct-hf; do
  TAG=$(echo "$M" | tr '/' '_'); n=$(wc -l < "$OUT/$TAG.jsonl" 2>/dev/null || echo 0)
  if [ "$n" -ge 2 ]; then echo "  USABLE          $M"; else echo "  PROTOCOL-BROKEN $M"; fi
done
echo "# done $(date -u +%FT%TZ)"
