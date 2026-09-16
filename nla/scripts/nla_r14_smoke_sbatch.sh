#!/bin/bash
#SBATCH --job-name=r14_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r14_smoke.out
# H-R14 smoke (CLAUDE.md §4: micro-scale first). 3 snippets NEVER scored before (outside the H-R7 50),
# exercising the three things that are actually new: the full pack files, the shard file path, and the
# residual path reading the FULL packs-orig. Not a result -- a wiring check.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/smoke; mkdir -p "$OUT"
echo "# H-R14 smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
rc=0
run() { echo "--- $1 ---"; python nla/src/ase_steer_run.py --packs "$2" --arm "$3" ${4:-} \
  --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
  --vectors-config nla/configs/ase_vectors_full.yaml --out "$OUT/$1.jsonl" --max-hours 0.4 || rc=$?; }
run unsteered          $F/packs_ren_smoke.jsonl  unsteered
run original_unsteered $F/packs_orig_smoke.jsonl unsteered
run swap_oracle        $F/packs_ren_smoke.jsonl  swap_oracle "--packs-orig $F/packs_orig_full.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0"
# the sharded code path, one shard file, attention arm: this is the one that will run 8 times for real
run codesteer_shardpath $F/packs_ren_full.shard0.jsonl codesteer
for f in "$OUT"/*.jsonl; do echo "$(basename $f): $(wc -l < $f) rows"; done
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
