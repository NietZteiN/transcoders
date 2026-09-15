#!/bin/bash
#SBATCH --job-name=nla_hsweep
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_hsweep.out
# H-S2/H-S3: which attention heads carry the NLA-steered state to the answer tokens?
# Pre-registered in log/nla-harness/2026-09-11_steering-sweep-prereg.md.
#
# Write layer fixed at L7 by H-S1 (job 390856): steering ONE layer beats many (k=33 minus k=1 =
# -11.70 nats, CI [-17.22,-6.51]) and L7 is the best single layer (+65.75 vs +50.26 for a
# depth-matched control). Writing early also maximises the downstream component count:
# 26 layers x 8 heads + 26 MLPs = 234 components.
#
# EXIT 3 = the identity checks failed (SELF_c within 0.05 nats of dG_S, ALL within 5 %); results are
# then not reportable. Needs NO AV/AR checkpoint -- it reuses the gate's banked L7 vectors, which is
# why it runs on the 4B even though AR_CHECKPOINTS has no 4B entry.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-S2 head sweep · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_head_sweep.py nla/src/head_patch.py nla/src/nla_ml_gate.py nla/src/steer.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo; echo "=== unit: the component-patching contracts ==="
python -m pytest nla/tests/test_head_patch.py -q -p no:faulthandler || exit 1

echo; echo "=== SMOKE (3 items: identity checks must pass before the full sweep) ==="
python nla/src/nla_head_sweep.py --limit 3 --max-hours 0.5 \
  --out /scratch/juno/jvl210002/hsweep_smoke || exit 1

echo; echo "=== FULL (60 items) ==="
python nla/src/nla_head_sweep.py --max-hours 5 || exit 1
echo "# done $(date -u +%FT%TZ)"
