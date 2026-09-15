#!/bin/bash
#SBATCH --job-name=nla_ml_layer
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-33
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_nla_ml_layer.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_nla_ml_layer.out
# Phase B step 4: one array task per layer K = SLURM_ARRAY_TASK_ID: AV SFT -> AR SFT -> check.
# Each stage skips itself if its output exists, so a re-submit resumes at the failed stage.
# fp32 master weights + Adam for the 4B host need ~70 GB before activations -> H200 (141 GB) only.
# Submit with --dependency=afterok:<join job>. The pre-registered gate runs separately afterwards.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
K=$SLURM_ARRAY_TASK_ID; T=nla/src/nla_train.py
echo "# ML layer $K · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $T nla/src/gemma_text.py nla/src/local_av.py nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python $T --stage sft_av --layer $K || exit 1
python $T --stage sft_ar --layer $K || exit 1
python $T --stage check  --layer $K || exit 1
echo "# done $(date -u +%FT%TZ)"
