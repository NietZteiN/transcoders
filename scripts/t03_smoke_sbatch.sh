#!/bin/bash
#SBATCH --job-name=t03_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_t03_smoke.out
# T0.3 -- the gate in front of E1/E2/E7: is a Llama Scope SAE healthy on Llama-3.1-8B-INSTRUCT,
# given the dictionaries are trained on BASE? Backlog: docs/EXPERIMENT_BACKLOG.md.
#
# NOTE: HF_HUB_OFFLINE is deliberately NOT set. SAELens's official llama_scope loader resolves the
# `fnlp/...` alias while our cache holds `OpenMOSS-Team/...@8dbc1d85`, so the SAE (~513 MB for this
# layer) is fetched. The script then compares the fetched tensors against the pinned local copy, so the
# smoke either transfers to the revision E1 will use or tells us it does not.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
cd "$PROJ"
E="$MI_ENV"
export LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# T0.3 dictionary smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum src/t03_dictionary_smoke.py
nvidia-smi --query-gpu=index,name --format=csv,noheader
"$E/bin/python" src/t03_dictionary_smoke.py \
  --out results/2026-09-20_t03_dictionary_smoke.json; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
