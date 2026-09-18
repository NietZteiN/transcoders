#!/bin/bash
#SBATCH --job-name=nla_r26
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r26_%x.out
# H-R26 -- is Phi-3.5-mini's parse failure a GENERATION-BUDGET artefact? Same 146 snippets, same packs,
# same greedy decoder; only --max-new-tokens changes (512 -> 1536). DECLARED DEVIATION from the uniform
# panel protocol: a number produced here is not comparable with the panel's 512-token numbers, and if
# this yields a host the whole panel must be re-run at the matched budget.
# Pre-registered: log/nla-harness/2026-09-18_phi-budget-prereg.md
# $1 = l0|swap   $2 = max_new_tokens
set -uo pipefail
COND="${1:?l0|swap}"; MNT="${2:?max_new_tokens}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/phi_budget; mkdir -p "$OUT"
case "$COND" in
  l0)   PACKS=$F/packs_orig_swapsubset.jsonl;;
  swap) PACKS=$F/packs_swap_paired.jsonl;;
  *) echo "bad condition"; exit 2;;
esac
echo "# H-R26 cond=$COND max_new_tokens=$MNT · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
python nla/src/ase_steer_run.py --packs "$PACKS" --arm unsteered --greedy --runs 1 \
  --model-id microsoft/Phi-3.5-mini-instruct --chat-template --max-new-tokens "$MNT" \
  --vectors-config nla/configs/ase_vectors_full.yaml \
  --out "$OUT/phi.$COND.mnt$MNT.jsonl" --max-hours 1.5; rc=$?
echo "# rows: $(wc -l < "$OUT/phi.$COND.mnt$MNT.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
