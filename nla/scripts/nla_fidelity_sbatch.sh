#!/bin/bash
#SBATCH --job-name=nla_fidelity
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_fidelity.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_fidelity.out
# H-W7 — causal fidelity of the NLA round trip: C3pure / P_patch at the 375 locatable spans.
# Pre-registered 2026-09-04 (nla-fidelity-prereg). Raised by stage 1's C3 = +37.91 nats. Three 12B models in one process (subject + AV + AR) on one H200.
#
# Generations dominate the cost: 4 per item (3 arms + unsteered) x 60 items at 1100 new tokens.
# MAX_NEW_GEN stays at 1100 to match every banked run -- a smaller budget silently zeroes the
# control (2026-08-30: 12% of items were being scored wrong for not finishing).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
OUT="$PROJ/data/nla/p0/fidelity/$HOST"; mkdir -p "$OUT"

echo "# Experiment H-W7 causal fidelity · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_fidelity.py nla/src/nla_writeback.py nla/src/steer.py nla/src/local_av.py nla/src/nla_cycle.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

echo; echo "=== unit: the frozen verdict table ==="
python -m pytest nla/tests/test_writeback.py nla/tests/test_steer.py -q \
  -k "verdict or veto or support or foreign or replacer or no_grad" || exit 1

echo; echo "=== SMOKE (3 items, full path incl. generation) ==="
python nla/src/nla_fidelity.py --model "$HOST" --smoke --out-dir "$OUT/smoke" \
  --max-hours 1 || exit 1

echo; echo "=== FULL (60 items) ==="
python nla/src/nla_fidelity.py --model "$HOST" --out-dir "$OUT" --max-hours 9 || exit 1
echo "# done $(date -u +%FT%TZ)"
