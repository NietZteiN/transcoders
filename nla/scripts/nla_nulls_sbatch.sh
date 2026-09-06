#!/bin/bash
#SBATCH --job-name=nla_nulls
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_nulls.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_nulls.out
# Null battery — H-W9 (item) / H-W12 (span) / H-W6 (operator) specificity of the clean-state effect.
# Pre-registered 2026-09-05 (null-battery-prereg). Decides how H-W7's fidelity result may be
# described. NO AV/AR are loaded: every arm is a raw activation or a random vector. Three 12B models in one process (subject + AV + AR) on one H200.
#
# Generations dominate the cost: 5 per item (4 arms + unsteered) x 60 items at 1100 new tokens.
# MAX_NEW_GEN stays at 1100 to match every banked run -- a smaller budget silently zeroes the
# control (2026-08-30: 12% of items were being scored wrong for not finishing).
# h100 is eligible as well as h200: this job loads ONLY the subject model (~24 GB bf16), no AV and
# no AR, so an 80 GB card is ample. Measured 2026-09-06: queue wait on this cluster has a median of
# 110 min against a median 8 min runtime (3x more time waiting than computing), and widening the
# eligible partitions is the cheapest thing that touches that. The three-model jobs (nla_cycle,
# nla_writeback, nla_fidelity) need ~72 GB and must stay on h200.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
OUT="$PROJ/data/nla/p0/nulls/$HOST"; mkdir -p "$OUT"

echo "# Experiment W9 null battery · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_nulls.py nla/src/nla_writeback.py nla/src/steer.py nla/src/local_av.py nla/src/nla_cycle.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

echo; echo "=== unit: the frozen verdict table ==="
python -m pytest nla/tests/test_writeback.py nla/tests/test_steer.py -q \
  -k "verdict or veto or support or foreign or replacer or no_grad" || exit 1

echo; echo "=== SMOKE (3 items, full path incl. generation) ==="
python nla/src/nla_nulls.py --model "$HOST" --smoke --out-dir "$OUT/smoke" \
  --max-hours 1 || exit 1

echo; echo "=== FULL (60 items) ==="
python nla/src/nla_nulls.py --model "$HOST" --out-dir "$OUT" --max-hours 9 || exit 1
echo "# done $(date -u +%FT%TZ)"
