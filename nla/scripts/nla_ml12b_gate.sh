#!/bin/bash
#SBATCH --job-name=nla12b_gate
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla12b_gate.out
# H-C1: score OUR 12B L32 pair against the released kitft/nla-gemma3-12b-L32 pair's 0.98 fidelity,
# on the SAME 60 items / 471 spans. Pre-registered thresholds (frozen 2026-09-10):
#   S_c3/S_swap <= 0.80 -> BUDGET-EXPLAINS · >= 0.90 -> HOST-EXPLAINS · between -> INDETERMINATE
#
# --traces IS MANDATORY HERE. nla_ml_gate.py defaults to the 4B path; scoring this pair against the
# 4B host's banked replies would be silently wrong rather than an error.
#
# Single GPU: the vectors stage holds host + AV + AR (23.5 + 21.9 + ~17 GB bf16 = ~62 GB), which fits
# a 143 GB H200. L2/L7 were descoped (user, 2026-09-11: stop after one layer), so H-C2 is NOT
# adjudicated and the only set is primary=[32].
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
CFG=nla/configs/nla_ml_gate_12b.yaml
TRACES=$PROJ/data/nla/p0/trace_llr/gemma12b/traces.jsonl
echo "# H-C1 12B gate · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_ml_gate.py "$CFG" nla/configs/nla_ml_12b.yaml
wc -l "$TRACES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_nla_ml_gate.py -q -p no:faulthandler || exit 1

echo; echo "=== vectors L32 (host + AV + AR) ==="
python nla/src/nla_ml_gate.py --stage vectors --layer 32 --config "$CFG" --traces "$TRACES" \
  --max-hours 5 || exit 1

echo; echo "=== score (L32 only) ==="
python nla/src/nla_ml_gate.py --stage score --config "$CFG" --traces "$TRACES" --max-hours 2; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
