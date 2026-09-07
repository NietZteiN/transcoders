#!/bin/bash
#SBATCH --job-name=flip_census
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_flip_census.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_flip_census.out
# How many items does Llama-3.1-8B get right clean and wrong obfuscated?
# Gemma gives 6/60, so a perfect rescue is +0.100 and the standing power gate (>=9) fails --
# which is why every W result is a likelihood claim rather than a behavioural one. Dataset B is
# already in the corpus, so a bigger n is not available; only a host where the trap bites harder.
# No steering, no NLA. Llama-3.1-8B is ~16 GB bf16, so h100 is ample.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

OUT="$PROJ/data/nla/p0/census/llama8b"; mkdir -p "$OUT"
echo "# flippable census · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/flippable_census.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo; echo "=== CPU path checks (tokenizer only, no weights) ==="
python -m pytest nla/tests/test_census_path.py -q || exit 1

echo; echo "=== SMOKE (4 items) ==="
python nla/src/flippable_census.py --model llama8b --limit 4 --out-dir "$OUT/smoke" \
  --max-hours 1 || exit 1

echo; echo "=== FULL ==="
python nla/src/flippable_census.py --model llama8b --out-dir "$OUT" --max-hours 4 || exit 1
echo "# done $(date -u +%FT%TZ)"
