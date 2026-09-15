#!/bin/bash
#SBATCH --job-name=r2_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r2_smoke.out
# Smoke: does THEIR SteeredCausalLM load and steer on this host after the cache-dir fix?
# One snippet, steering ON (slice_hybrid). Verifies model load + steering install + AST prior +
# generation + parse, before five full arms are spent.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
head -2 "$A/gate/packs_ren_subset.jsonl" > "$A/smoke_packs.jsonl"
echo "# R2 smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
rm -f "$A/smoke_codesteer.jsonl" "$A/smoke_unsteered.jsonl"   # two small files from a prior failed smoke
sha256sum nla/src/ase_steer_run.py
python nla/src/ase_steer_run.py --packs "$A/smoke_packs.jsonl" --arm codesteer \
  --model-id codellama/CodeLlama-7b-Instruct-hf --runs 1 \
  --out "$A/smoke_codesteer.jsonl" --max-hours 0.5; rc1=$?
echo "--- steered smoke rc=$rc1"
python nla/src/ase_steer_run.py --packs "$A/smoke_packs.jsonl" --arm unsteered \
  --model-id codellama/CodeLlama-7b-Instruct-hf --runs 1 \
  --out "$A/smoke_unsteered.jsonl" --max-hours 0.5; rc2=$?
echo "--- unsteered smoke rc=$rc2"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
