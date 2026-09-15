#!/bin/bash
#SBATCH --job-name=nla_ml_gsmoke
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_gate_smoke.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_gate_smoke.out
# Phase B gate SMOKE: the mechanics of nla_ml_gate.py on the 1-step L5 smoke pair
# (data/nla/ml/gemma4b_smoke/L5, job 384622) and 3 items. Its numbers are NOT a result and its
# liveness checks are bypassed (--ignore-liveness) precisely because a 1-step pair is dead by
# construction; what is checked here is that every arm writes the expected positions, the
# SELF identity holds (exit 3 otherwise), and the rules run end to end.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=4
G=nla/src/nla_ml_gate.py; ROOT=$PROJ/data/nla/ml/gemma4b_smoke
echo "# ML gate smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $G nla/src/steer.py nla/src/arm_guard.py nla/src/gemma_text.py nla/configs/nla_ml_gate.yaml nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo; echo "=== unit: frozen rules + arm_guard ==="
python -m pytest nla/tests/test_nla_ml_gate.py nla/tests/test_arm_guard.py -q -p no:faulthandler || exit 1
echo; echo "=== vectors L5 (3 items) ==="
python $G --stage vectors --layer 5 --root $ROOT --smoke || exit 1
cat $ROOT/gate/vectors/L5_manifest.json | head -40
echo; echo "=== score (3 items, --ignore-liveness) ==="
python $G --stage score --root $ROOT --smoke --ignore-liveness; rc=$?
echo "score exit $rc (0 ok, 3 = SELF identity failed)"
[ $rc -eq 0 ] || exit $rc
echo "# done $(date -u +%FT%TZ)"
