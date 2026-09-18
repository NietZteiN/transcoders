#!/bin/bash
#SBATCH --job-name=nla_r6_pool
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=01:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r6_pool.out
# H-R6 -- pool capture (layer-7 span states + type/role tags for all 156 renamed snippets) followed by
# the CPU fit of the ridge_map / role_proto vectors for the 50 test snippets.
# Pre-registered: log/nla-harness/2026-09-14_better-vector-prereg.md · config nla/configs/ase_vectors.yaml
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-R6 pool · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_pool_capture.py nla/src/ase_vectors.py nla/src/ase_roles.py nla/src/ase_steer_run.py nla/configs/ase_vectors.yaml
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== tests ==="
python -m pytest nla/tests/test_ase_vectors.py -q || { echo "# tests failed"; exit 3; }
echo "=== capture ==="
python nla/src/ase_pool_capture.py --model-id codellama/CodeLlama-7b-Instruct-hf; rc=$?
[ $rc -ne 0 ] && { echo "# capture rc=$rc"; exit $rc; }
echo "=== fit (CPU) ==="
python nla/src/ase_vectors.py; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
