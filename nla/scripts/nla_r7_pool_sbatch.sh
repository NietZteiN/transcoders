#!/bin/bash
#SBATCH --job-name=r7_pool
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r7_%x.out
# H-R7 -- the H-R6 pool capture and vector fit REDONE under the chat template (ase_vectors_chat.yaml):
# layer-7 states of an [INST]-wrapped prompt are not those of the raw prompt, so the ridge_map /
# role_proto vectors written in the chat-template bake-off must be fitted on chat-template captures.
# Pre-registered: log/nla-harness/2026-09-15_chat-template-rerun-prereg.md
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
CFG=nla/configs/ase_vectors_chat.yaml
echo "# H-R7 pool (chat template) · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_pool_capture.py nla/src/ase_vectors.py nla/src/ase_roles.py nla/src/ase_steer_run.py $CFG
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== tests ==="
python -m pytest nla/tests/test_ase_vectors.py -q || { echo "# tests failed"; exit 3; }
echo "=== capture ==="
python nla/src/ase_pool_capture.py --config $CFG --model-id codellama/CodeLlama-7b-Instruct-hf; rc=$?
[ $rc -ne 0 ] && { echo "# capture rc=$rc"; exit $rc; }
echo "=== fit (CPU) ==="
python nla/src/ase_vectors.py --config $CFG; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
