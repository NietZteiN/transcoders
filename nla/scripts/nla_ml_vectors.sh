#!/bin/bash
#SBATCH --job-name=nla_ml_vec
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --array=0-33
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_nla_ml_vec.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_nla_ml_vec.out
# Phase B step 5: gate vectors, one array task per trained layer K. host + AV_K + AR_K (~24 GB).
# Per span: 2 greedy AV reads (<=180 tokens) + 4 AR reconstructions; ~471 spans -> ~1-1.5 h.
# A PAIR-DEAD layer still gets its vectors (cheap) so the per-layer table can show it; the score
# stage is what excludes it. Submit with --dependency=afterok:<layer array>.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
K=$SLURM_ARRAY_TASK_ID; G=nla/src/nla_ml_gate.py
echo "# ML gate vectors L$K · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $G nla/src/steer.py nla/src/local_av.py nla/src/gemma_text.py nla/configs/nla_ml_gate.yaml nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python $G --stage vectors --layer $K --max-hours 5.5 || exit 1
echo "# done $(date -u +%FT%TZ)"
