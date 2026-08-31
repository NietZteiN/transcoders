#!/usr/bin/env bash
# B3 coupling capture on ONE GPU — AV server and runner co-located.
#
#   NLA_GPU=0 bash nla/scripts/deception_1gpu.sh --snippets <file>
#
# Why a separate driver: `deception.sh` demands two idle GPUs and puts the AV server on its own
# card. The box is shared and the other three cards are in use, so this variant fits both on one.
#
# MEMORY BUDGET on a 48 GB A6000 (measured on the two-GPU runs):
#   AV server (sglang, 7B bf16)  --mem-fraction-static 0.40  -> ~19.2 GB
#   runner: subject 7B bf16 ~15 GB (+ activations)                    -> ~17 GB
#   AR on CPU                                                          -> 0 GB
#   total ~36 GB, leaving ~12 GB headroom.
#
# The first attempt used 0.30 and the server refused to start: "Loaded weights leave no GPU
# memory for the KV cache under --mem-fraction-static=0.3 (minimum viable 0.3040)". The
# fraction covers the WHOLE server including the ~15.2 GB of bf16 weights, not just the cache —
# 0.30 x 48 GB = 14.4 GB does not even hold the weights. Raising it alone leaves ~2 GB of
# headroom, too tight to leave running unattended, so the AR also moves to CPU. It is a ~11 GB
# model used once per read for the rt_cos quality diagnostic, so the cost is bounded and this
# run is unattended anyway.
# The AV's KV cache can be small because its prompts are ~450 tokens at batch 1 — the read path
# is strictly sequential (batching was measured to break AV determinism: 6.3x faster but only
# 15% byte-identical), so nothing here benefits from a larger cache.
#
# Sets no GPU-idleness gate: it is pinned to NLA_GPU deliberately, and that card is ours.
set -euo pipefail
cd "$(dirname "$0")/.."          # nla/

ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
GPU="${NLA_GPU:-0}"
PORT="${NLA_PORT:-30003}"
OUT_DIR="${OUT_DIR:-/data/jvl210002/my_downloads/transcoders/data/nla/n11}"
mkdir -p "$OUT_DIR"

echo "[1gpu] $(date -Is) gpu=$GPU port=$PORT args=$*"

CUDA_VISIBLE_DEVICES="$GPU" setsid python -m sglang.launch_server \
  --model-path data/checkpoints/av --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.40 \
  > "$OUT_DIR/server_1gpu.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/server_1gpu.pid"
cleanup() {
  echo "[1gpu] cleanup: killing server session (pgid $SERVER_PID)"
  kill -- -"$SERVER_PID" 2>/dev/null || true
  sleep 2
  # bracketed so the pattern cannot match this script's own command line and kill the shell
  pkill -9 -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[1gpu] server ready after ~$((i*5))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[1gpu] ABORT: server died during startup — see $OUT_DIR/server_1gpu.log"; exit 3
  fi
  if (( i == 60 )); then echo "[1gpu] ABORT: server not healthy after 300s"; exit 3; fi
  sleep 5
done

set +e
CUDA_VISIBLE_DEVICES="$GPU" python -m src.malware_reads \
  --sglang-url "http://localhost:$PORT" "$@" 2>&1 \
  | tee -a "$OUT_DIR/runner_1gpu.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[1gpu] runner exited $CODE"
exit "$CODE"
