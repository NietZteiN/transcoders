#!/bin/bash
#SBATCH --job-name=nla_erase
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_erasure.out
# H-E1/H-E2/H-E3 -- the cross-item decoy-erasure vector with NO autoencoder, written at the span
# positions (L7 beta=1 primary; band L2-13 beta=0.35 secondary). Pre-registered in
# log/nla-harness/2026-09-12_erasure-vector-prereg.md. Identity gates: erase_own == banked swap,
# edit/foreign/swap reproduce banked rows (<= 0.05 nats), self == 0 -> else exit 3, E-HARNESS-FAULT.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# erasure vector · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_erasure.py nla/src/steer.py nla/src/nla_ml_gate.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_erasure.py nla/tests/test_steer.py -q -k "erasure or erase or loo or rules or identity or replacer or no_grad" || exit 1

echo; echo "=== SMOKE: 3 items (identity gates must pass) ==="
python nla/src/nla_erasure.py --smoke --out /scratch/juno/jvl210002/erase_smoke || exit 1

echo; echo "=== FULL: 60 items ==="
python nla/src/nla_erasure.py || exit 1
echo "# done $(date -u +%FT%TZ)"
