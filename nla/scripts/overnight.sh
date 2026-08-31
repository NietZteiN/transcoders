#!/usr/bin/env bash
# Overnight capture driver — run inside detached tmux:
#   tmux new-session -d -s nla-overnight 'bash nla/scripts/overnight.sh'
# Smoke: bash nla/scripts/overnight.sh --limit 4
#
# Owns the whole lifecycle: 2-idle-GPU check (shared box — abort rather than stomp),
# AV server launch on GPU_B (+ trap kill on exit), health wait, runner on GPU_A.
set -euo pipefail
cd "$(dirname "$0")/.."          # nla/

ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
PORT=30000
OUT_DIR="${OUT_DIR:-${OUT:-/data/jvl210002/my_downloads/transcoders/data/nla/overnight/$(date +%F)}}"
mkdir -p "$OUT_DIR"

echo "[driver] $(date -Is) out=$OUT_DIR args=$*"

# ---- pick two idle GPUs (idle = <1GB used AND <5% util) ----------------------------
readarray -t IDLE < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 && $3 < 5 {print $1}')
if (( ${#IDLE[@]} < 2 )); then
  echo "[driver] ABORT: need 2 idle GPUs, found ${#IDLE[@]} (${IDLE[*]:-none}) — shared box, not stomping."
  exit 2
fi
GPU_A="${IDLE[0]}"   # runner (extractor + AR)
GPU_B="${IDLE[1]}"   # AV server
echo "[driver] runner on GPU $GPU_A, AV server on GPU $GPU_B"

# ---- AV server on GPU_B in its OWN session (setsid) so cleanup can kill the whole
# tree by pgid without touching the driver's group; sglang forks scheduler children.
CUDA_VISIBLE_DEVICES="$GPU_B" setsid python -m sglang.launch_server \
  --model-path data/checkpoints/av --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.6 \
  > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/server.pid"
cleanup() {
  echo "[driver] cleanup: killing server session (pgid $SERVER_PID)"
  kill -- -"$SERVER_PID" 2>/dev/null || true          # setsid => pgid == pid
  sleep 2
  pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 48); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[driver] server ready after ~$((i*5))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[driver] ABORT: server process died during startup — see $OUT_DIR/server.log"; exit 3
  fi
  if (( i == 48 )); then echo "[driver] ABORT: server not healthy after 240s"; exit 3; fi
  sleep 5
done

# ---- runner on GPU_A ----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES="$GPU_A" NLA_SERVER_GPU="$GPU_B" \
  python -m src.overnight_capture --out-dir "$OUT_DIR" --sglang-url "http://localhost:$PORT" "$@" \
  2>&1 | tee -a "$OUT_DIR/runner.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[driver] runner exited $CODE — summary at $OUT_DIR/summary.md"
exit "$CODE"
