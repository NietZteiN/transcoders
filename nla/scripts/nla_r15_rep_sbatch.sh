#!/bin/bash
#SBATCH --job-name=nla_r15_rep
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r15_%x.out
# H-R15 -- one REPLICATE of an H-R14 arm on the SAME frozen 148 snippets, same vectors, same runtime:
# only --seed differs (and for replicate A2, not even that -- it is the same-seed control).
# Pre-registered: log/nla-harness/2026-09-17_replication-floor-prereg.md
# $1 = replicate id (A2|B1|C1|B2|B3)   $2 = arm   $3 = seed   $4 = optional shard
set -uo pipefail
REP="${1:?usage: sbatch nla_r15_rep_sbatch.sh <rep-id> <arm> <seed> [shard]}"
ARM="${2:?arm}"; SEED="${3:?seed}"; SHARD="${4:-}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/repro; mkdir -p "$OUT"
SUF=""; [ -n "$SHARD" ] && SUF=".shard$SHARD"
echo "# H-R15 replicate=$REP arm=$ARM seed=$SEED shard=${SHARD:-none} · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py; sha256sum $F/snippets_148.json
nvidia-smi --query-gpu=index,name --format=csv,noheader
if [ -n "$SHARD" ]; then PACKS="$F/packs_ren_full.shard$SHARD.jsonl"; else PACKS="$F/packs_ren_full.jsonl"; fi
EXTRA=""
case "$ARM" in
  ridge_map)
    # the SAME cross-fitted vectors as H-R14 -- this is a decoding replicate, NOT a re-fit
    ls -la $F/vectors_codellama7b_L7_full.pt
    EXTRA="--packs-orig $F/packs_orig_full.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0 --vectors $F/vectors_codellama7b_L7_full.pt";;
esac
python nla/src/ase_steer_run.py --packs "$PACKS" --arm "$ARM" $EXTRA --seed "$SEED" \
  --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
  --vectors-config nla/configs/ase_vectors_full.yaml --out "$OUT/$REP.$ARM$SUF.jsonl" --max-hours 7.0; rc=$?
echo "# rows: $(wc -l < "$OUT/$REP.$ARM$SUF.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
