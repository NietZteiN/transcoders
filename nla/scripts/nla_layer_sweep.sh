#!/bin/bash
#SBATCH --job-name=nla_lsweep
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_lsweep.out
# H-S1: how many layers should we steer? Pre-registered in
# log/nla-harness/2026-09-11_steering-sweep-prereg.md.
#
# Split-half honest: layers are ranked by single-layer dG_c3 on one half of the items and the top-k
# is evaluated on the OTHER half (both directions, pooled), so no set is chosen and scored on the
# same items. A depth-spaced ladder runs alongside as a selection-free control.
#
# Readout is G_sum, NOT accuracy: this host loses 2 of 60 items to obfuscation and the flippable
# census found 6/60 flippable, so accuracy cannot carry a claim (H-S4 handles it as a secondary).
#
# Reuses the banked 34-layer vectors (gate/vectors/L{K}.npz) and the frozen liveness rules, so a
# PAIR-DEAD layer (L0) is never steered. ~6 800 forwards at the measured 36 ms => a few minutes.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-S1 layer sweep · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_layer_sweep.py nla/src/nla_ml_gate.py nla/src/steer.py nla/configs/nla_ml_gate.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo; echo "=== SMOKE (6 items, full ladder) ==="
python nla/src/nla_layer_sweep.py --limit 6 --max-hours 0.5 \
  --out /scratch/juno/jvl210002/lsweep_smoke || exit 1

echo; echo "=== FULL (60 items) ==="
python nla/src/nla_layer_sweep.py --max-hours 3 || exit 1
echo "# done $(date -u +%FT%TZ)"
