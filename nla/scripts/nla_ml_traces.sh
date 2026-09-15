#!/bin/bash
#SBATCH --job-name=nla_ml_traces
#SBATCH --partition=h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_traces.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_traces.out
# Phase B stage 0a — bank gemma4b's own L0/L1b greedy replies on the 60 items (host_traces.py).
# Independent of the NLA training stages; gives the go/no-go accuracy band of the multi-layer
# prereg (docs/nla_multilayer_scoping.md §4: within 0.633 ± 0.15 on the 60 items) and the
# traces.jsonl the per-layer gate scores against. One 4B model in bf16 (~8 GB): any card.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
HOST=gemma4b
OUT="$PROJ/data/nla/p0/trace_llr/$HOST"; mkdir -p "$OUT"
echo "# Phase B traces · $HOST · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/host_traces.py nla/src/steer_run.py nla/src/capture_core.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo; echo "=== SMOKE (3 items) ==="
python nla/src/host_traces.py --model "$HOST" --limit 3 --out-dir "$OUT/smoke" || exit 1
cat "$OUT/smoke/traces_summary.json"
echo; echo "=== FULL (60 items) ==="
python nla/src/host_traces.py --model "$HOST" --out-dir "$OUT" || exit 1
cat "$OUT/traces_summary.json"
echo "# done $(date -u +%FT%TZ)"
