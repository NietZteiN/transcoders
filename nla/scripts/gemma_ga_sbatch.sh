#!/bin/bash
# Phase G-A — the two gates that decide whether the Gemma port proceeds.
#
#   1. layer-indexing gate at L32: does the extractor read the block the steerer writes?
#      This is the one the `_layers()` fix could have broken silently — Gemma-3 loads as
#      Gemma3ForConditionalGeneration and both files had to learn a new path.
#   2. G0 stage A: do our per-token ||v|| match the shipped reference for gemma12b?
#      A pure forward-pass check — no NLA involved — so a failure isolates layer indexing and
#      chat templating from anything about the autoencoder.
#
# Stage B (the full round trip) needs the AV server and is a separate step.
#SBATCH --job-name=gemma_ga
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_gemma_ga.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_gemma_ga.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "# Gemma G-A · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

echo "===== gate 1: layer indexing at L32 ====="
python nla/src/p04_gate.py --host gemma12b \
  --out "$PROJ/data/nla/p0/gemma/p04_gate_gemma12b.json"
echo "gate 1 exit $?"
echo

echo "===== gate 2: G0 stage A vs the shipped reference ====="
python nla/src/replicate_example.py --stage A --model gemma12b
echo "gate 2 exit $?"
