#!/bin/bash
#SBATCH --job-name=nla_acc
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_accuracy.out
# Does steering the NLA improve ACCURACY? Run at the user's explicit request, notwithstanding H-S9's
# gate not firing (+1.67 < +3.00). Pre-registered in log/nla-harness/2026-09-12_accuracy-prereg.md.
#
# THE BOUND TRAVELS WITH THE RESULT: acc_l0 0.567 -> acc_l1b 0.533 is 2/60 net headroom and 7
# flippable, against a same-condition greedy churn of 6-9/60. A PERFECT rescue of all 7 gives
# McNemar p = 0.092. No positive significance claim is possible at n=60; this run reports effect
# sizes and intervals with NO verdict word.
#
# Arms: noop (baseline) · c3_band (strongest NLA write we have: band L2-L13 at beta=0.35) ·
# edit_band (THE intervention) · foreign_band (content null) · edit_single (banked L2/beta=1
# standard) · prompt (mandatory prompting baseline, CLAUDE.md §4, guarded by a decode->encode
# round-trip check that refuses rather than malforming the prompt).
# Two readouts: greedy n=1 (comparable with every banked number) and n=8 pass-rate (~2.8x smaller
# per-item SE, the only cheap lever on the churn floor).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# accuracy under NLA steering · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_accuracy.py nla/src/steer.py nla/src/nla_ml_gate.py nla/src/steer_run.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_steer.py -q -k "replacer or no_grad" || exit 1

echo; echo "=== SMOKE: 2 items, greedy only (validates the whole generate+write path) ==="
python nla/src/nla_accuracy.py --limit 2 --no-sampled --max-hours 0.5 \
  --out /scratch/juno/jvl210002/acc_smoke || exit 1

echo; echo "=== GREEDY pass, 60 items (the comparable headline number) ==="
python nla/src/nla_accuracy.py --no-sampled --max-hours 3 \
  --out /work/jvl210002/migration/transcoders/data/nla/ml/gemma4b/gate/accuracy_greedy || exit 1

echo; echo "=== SAMPLED pass, 60 items x 8 samples (the powered readout) ==="
python nla/src/nla_accuracy.py --n-samples 8 --max-hours 5 || exit 1
echo "# done $(date -u +%FT%TZ)"
