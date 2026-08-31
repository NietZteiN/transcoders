#!/usr/bin/env bash
# N7 driver — judge alignment + the AR baseline. Owns GPU selection and the judge server lifecycle.
#   tmux new-session -d -s nla-n7 'bash nla/scripts/n7.sh'
# Env knobs: PACK_DIR, JUDGE (hf id), TAG (output suffix), SKIP_AR=1
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
PACK_DIR="${PACK_DIR:-$(cd .. && pwd)/data/nla/n7/$(date +%F)}"
JUDGE="${JUDGE:-meta-llama/Llama-3.1-8B-Instruct}"
TAG="${TAG:-llama}"
PORT="${PORT:-30010}"
[ -f "$PACK_DIR/pack.jsonl" ] || { echo "[driver] no pack at $PACK_DIR — run sanitize_windows.py first"; exit 2; }

# --- idle GPUs only (shared box, no scheduler) ---
mapfile -t IDLE < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', *' '$2 < 1000 && $3 < 10 {print $1}')
NEED=2; [ "${SKIP_AR:-0}" = "1" ] && NEED=1
if [ "${#IDLE[@]}" -lt "$NEED" ]; then
  echo "[driver] need $NEED idle GPU(s), found ${#IDLE[@]} — aborting (shared box)"; exit 3
fi
GPU_JUDGE="${IDLE[0]}"
echo "[driver] judge=$JUDGE on GPU$GPU_JUDGE  pack=$PACK_DIR"

CUDA_VISIBLE_DEVICES="$GPU_JUDGE" setsid "$ENV/bin/python" -m sglang.launch_server \
  --model-path "$JUDGE" --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.75 \
  > "$PACK_DIR/judge_server_$TAG.log" 2>&1 &
SRV_PGID=$!
cleanup() { echo "[driver] cleanup: killing judge server pgid $SRV_PGID"; kill -- -"$SRV_PGID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for i in $(seq 1 72); do
  curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "[driver] judge ready (~$((i*5))s)"; break; }
  if ! kill -0 "$SRV_PGID" 2>/dev/null; then echo "[driver] judge server died during startup"; tail -30 "$PACK_DIR/judge_server_$TAG.log"; exit 4; fi
  sleep 5
done

# The AR baseline needs no server and is independent of the judge — run it concurrently on the
# second idle GPU so the night costs one wall-clock pass, not two.
AR_PID=""
if [ "${SKIP_AR:-0}" != "1" ] && [ ! -f "$PACK_DIR/ar_baseline_manifest.json" ]; then
  GPU_AR="${IDLE[1]}"
  echo "[driver] AR baseline on GPU$GPU_AR"
  CUDA_VISIBLE_DEVICES="$GPU_AR" "$ENV/bin/python" -m src.ar_baseline \
    --pack "$PACK_DIR/pack.jsonl" > "$PACK_DIR/ar_baseline.log" 2>&1 &
  AR_PID=$!
fi

"$ENV/bin/python" -m src.judge_align --pack "$PACK_DIR/pack.jsonl" \
  --url "http://localhost:$PORT" --model "$JUDGE" --tag "$TAG" "$@" \
  2>&1 | tee -a "$PACK_DIR/judge_$TAG.log"
rc=${PIPESTATUS[0]}

if [ -n "$AR_PID" ]; then
  echo "[driver] waiting on AR baseline (pid $AR_PID)"; wait "$AR_PID" || echo "[driver] AR baseline exited nonzero"
fi
echo "[driver] judge exited $rc — verdicts at $PACK_DIR/verdicts_$TAG.jsonl"
exit $rc
