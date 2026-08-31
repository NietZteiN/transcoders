#!/usr/bin/env bash
# N13 Stage 3 — read instability, AV server + runner co-located on ONE GPU.
#
#   NLA_GPU=1 bash nla/scripts/n13_stage3.sh [--limit N]
#
# Memory budget on a 48 GB A6000, same as the B3 driver this is adapted from:
#   AV server (sglang, 7B bf16) --mem-fraction-static 0.40  -> ~19.2 GB
#   runner: subject 7B bf16 (activation extraction only)     -> ~17 GB
#   AR critic on CPU                                          -> 0 GB
#   total ~36 GB. 0.30 was tried on an earlier run and the server refused to start — the
#   fraction covers the WHOLE server including ~15.2 GB of weights, so 0.30 x 48 = 14.4 GB
#   does not even hold them.
#
# The AV KV cache can stay small: reads are strictly sequential at batch 1 (batching was
# measured to break AV determinism — 6.3x faster, only 15% byte-identical), so nothing here
# benefits from a bigger cache.
set -euo pipefail
cd "$(dirname "$0")/.."          # nla/

ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
GPU="${NLA_GPU:-1}"
PORT="${NLA_PORT:-30007}"
OUT_DIR="${OUT_DIR:-/data/jvl210002/my_downloads/transcoders/data/nla/n13}"
mkdir -p "$OUT_DIR"

echo "[n13s3] $(date -Is) gpu=$GPU port=$PORT args=$*"

CUDA_VISIBLE_DEVICES="$GPU" setsid python -m sglang.launch_server \
  --model-path data/checkpoints/av --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.40 \
  > "$OUT_DIR/s3_server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/s3_server.pid"
cleanup() {
  echo "[n13s3] cleanup: killing server session (pgid $SERVER_PID)"
  kill -- -"$SERVER_PID" 2>/dev/null || true
  sleep 2
  # bracketed so the pattern cannot match this script's own command line and kill the shell
  pkill -9 -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[n13s3] server ready after ~$((i*5))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[n13s3] ABORT: server died during startup — see $OUT_DIR/s3_server.log"; exit 3
  fi
  if (( i == 60 )); then echo "[n13s3] ABORT: server not healthy after 300s"; exit 3; fi
  sleep 5
done

set +e
CUDA_VISIBLE_DEVICES="$GPU" python -m src.read_instability \
  --sglang-url "http://localhost:$PORT" "$@" 2>&1 \
  | tee -a "$OUT_DIR/s3_runner.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[n13s3] runner exited $CODE"
exit "$CODE"
