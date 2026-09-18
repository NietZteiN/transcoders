#!/bin/bash
#SBATCH --job-name=nla_r23_arm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r23_%x.out
# H-R23 -- one (model, condition) cell of the panel damage screen. Greedy, unsteered, 146 PACKS-PAIRED
# snippets. No steering, no vectors, no alignment: this measures the STIMULUS only.
# Pre-registered: log/nla-harness/2026-09-18_panel-damage-prereg.md
# $1 = model id   $2 = l0 | swap
set -uo pipefail
M="${1:?model}"; COND="${2:?l0|swap}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/panel; mkdir -p "$OUT"
TAG=$(echo "$M" | tr '/' '_')
case "$COND" in
  l0)   PACKS=$F/packs_orig_swapsubset.jsonl;;
  swap) PACKS=$F/packs_swap_paired.jsonl;;
  *) echo "bad condition $COND"; exit 2;;
esac
echo "# H-R23 model=$M cond=$COND · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py; sha256sum $F/snippets_swap.json
nvidia-smi --query-gpu=index,name --format=csv,noheader
python nla/src/ase_steer_run.py --packs "$PACKS" --arm unsteered --greedy --runs 1 \
  --model-id "$M" --chat-template --vectors-config nla/configs/ase_vectors_full.yaml \
  --out "$OUT/$TAG.$COND.jsonl" --max-hours 3.5; rc=$?
echo "# rows: $(wc -l < "$OUT/$TAG.$COND.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
