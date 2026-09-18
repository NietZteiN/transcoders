#!/bin/bash
#SBATCH --job-name=nla_r18_arm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r18_%x.out
# H-R18 -- one arm of the GREEDY pass on the frozen 148 snippets. Same packs, runtime, chat template and
# cross-fitted vectors as H-R14; ONLY the decoder changes (do_sample=False on their own run_llama).
# One run per case: H-R15b proved greedy runs are byte-identical, and the runner refuses --runs != 1.
# DECLARED DEVIATION from the paper's protocol (their Pass@k presupposes sampling) -- a complement to
# the sampled pass, not a replacement.
# Pre-registered: log/nla-harness/2026-09-17_greedy-pass-prereg.md
# $1 = arm
set -uo pipefail
ARM="${1:?usage: sbatch nla_r18_arm_sbatch.sh <arm>}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/greedy; mkdir -p "$OUT"
echo "# H-R18 greedy arm=$ARM · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py; sha256sum $F/snippets_148.json
nvidia-smi --query-gpu=index,name --format=csv,noheader
PACKS="$F/packs_ren_full.jsonl"; RUNARM="$ARM"
[ "$ARM" = original_unsteered ] && { RUNARM=unsteered; PACKS="$F/packs_orig_full.jsonl"; }
EXTRA=""
case "$ARM" in
  swap_oracle|erasure)
    EXTRA="--packs-orig $F/packs_orig_full.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0";;
  ridge_map)
    EXTRA="--packs-orig $F/packs_orig_full.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0 --vectors $F/vectors_codellama7b_L7_full.pt";;
esac
python nla/src/ase_steer_run.py --packs "$PACKS" --arm "$RUNARM" $EXTRA --greedy --runs 1 \
  --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
  --vectors-config nla/configs/ase_vectors_full.yaml --out "$OUT/$ARM.jsonl" --max-hours 3.5; rc=$?
echo "# rows: $(wc -l < "$OUT/$ARM.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
