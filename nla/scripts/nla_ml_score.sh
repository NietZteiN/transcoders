#!/bin/bash
#SBATCH --job-name=nla_ml_score
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_score.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_score.out
# Phase B step 6: the pre-registered gate. Host only (~8 GB). Per item: 2 references +
# 7 single-layer arms x live layers + 7 arms x 3 layer sets (~260 forwards at <=34 live layers);
# 60 items -> ~16k forwards. Liveness (prereg checks a-d) decides the live set BEFORE any forward.
# EXIT 3 = SELF identity failed -> nothing is reportable. Submit with --dependency=afterok:<vectors array>.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
G=nla/src/nla_ml_gate.py
echo "# ML gate score · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $G nla/src/steer.py nla/src/arm_guard.py nla/src/gemma_text.py nla/configs/nla_ml_gate.yaml nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -m pytest nla/tests/test_nla_ml_gate.py nla/tests/test_arm_guard.py -q -p no:faulthandler || exit 1
python $G --stage score --max-hours 7; rc=$?
echo "score exit $rc"; [ $rc -eq 0 ] || exit $rc
echo "# done $(date -u +%FT%TZ)"
