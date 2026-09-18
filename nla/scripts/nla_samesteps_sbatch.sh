#!/bin/bash
#SBATCH --job-name=nla_ss
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_samesteps.out
# H-C10 same-steps control. Pre-registered in log/nla-harness/2026-09-12_same-steps-prereg.md.
# 50 % of the rows for 2 epochs = 678 optimiser steps = exactly the 100 %/1-epoch arm, with half the
# unique rows. Isolates DATA from OPTIMISATION, which is the confound standing between H-C4's
# DATA-LIMITED verdict and a ~30 GPU-h corpus purchase.
# --ignore-liveness per the standing deviation so all dose arms stay comparable.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
T=nla/src/nla_train.py; GATE=nla/src/nla_ml_gate.py; GCFG=nla/configs/nla_ml_gate_dose.yaml
TRACES=$PROJ/data/nla/p0/trace_llr/gemma4b/traces.jsonl
BASE=/scratch/juno/jvl210002/nla_ml_dose; R=$BASE/f050e2; K=7
echo "# H-C10 same-steps · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $T $GATE nla/src/dose_score.py $GCFG nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_dose_score.py -q || exit 1
python -m pytest nla/tests/test_beta_sweep.py -q -k "subsample" || exit 1
mkdir -p "$R"; for d in corpus explain acts; do [ -e "$R/$d" ] || ln -s "$BASE/$d" "$R/$d"; done
python $T --stage sft_av --layer $K --train-frac 0.50 --epochs 2 --root "$R" || exit 1
python $T --stage sft_ar --layer $K --train-frac 0.50 --epochs 2 --root "$R" || exit 1
python $T --stage check  --layer $K --root "$R" || exit 1
if [ -f "$R/gate/vectors/L$K.npz" ]; then echo "[ss] vectors present, skipping"; else
  python $GATE --stage vectors --layer $K --config "$GCFG" --root "$R" --traces "$TRACES" --max-hours 2 || exit 1; fi
python $GATE --stage score --config "$GCFG" --root "$R" --traces "$TRACES" --ignore-liveness --max-hours 1 || exit 1
echo "# done $(date -u +%FT%TZ)"
