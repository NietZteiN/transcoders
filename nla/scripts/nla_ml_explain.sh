#!/bin/bash
#SBATCH --job-name=nla_ml_explain
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --array=0-9
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_nla_ml_explain.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_nla_ml_explain.out
# Phase B step 2: stage-2 explanations from LOCAL Gemma-3-12B-it (no API), 10 shards in parallel.
# Resumable: re-submitting the array continues each shard's jsonl. --max-hours leaves margin for
# the join. Submit with --dependency=afterok:<extract job>.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
echo "# ML explain shard $SLURM_ARRAY_TASK_ID · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_train.py nla/configs/nla_ml.yaml
python nla/src/nla_train.py --stage explain --shard "$SLURM_ARRAY_TASK_ID" --n-shards 10 --max-hours 9.5 || exit 1
echo "# done $(date -u +%FT%TZ)"
