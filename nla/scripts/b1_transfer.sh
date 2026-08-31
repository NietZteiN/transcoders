#!/usr/bin/env bash
# B1 cross-model transfer gate — BOTH arms + the comparison, on ONE GPU.
#
#   NLA_GPU=0 bash nla/scripts/b1_transfer.sh [--n-snippets 200] [--reads-per-snippet 5]
#
# Runs the subject (Qwen2.5-Coder-7B-Instruct) and the control (Qwen2.5-7B-Instruct) against a
# SINGLE AV server, then emits the gate table. One server for both arms is not just a saving:
# the AV is the instrument, and holding it fixed across the comparison removes it as a variable.
# The two subject models load in separate python processes, so the first is freed before the
# second loads — they never coexist.
#
# MEMORY BUDGET on a 48 GB A6000 (same accounting as deception_1gpu.sh, which established it):
#   AV server (sglang, 7B bf16) --mem-fraction-static 0.40  -> ~19.2 GB
#   runner: subject 7B bf16 + activations                    -> ~17 GB
#   AR critic on CPU                                         ->  0 GB
#   total ~36 GB, ~12 GB headroom.
# AR must stay on CPU: 19.2 + 17 + 11 = 47.2 GB of 48 leaves nothing, and this runs unattended.
# The AR cost is one truncated 21-layer forward per read, and the read path is dominated by the
# AV generation anyway (measured 3.83 s/read sequential).
#
# The read path is sequential by necessity, not by oversight — batching the AV was measured at
# 6.3x faster but only 15% byte-identical (data/nla/n5/async_gate.json), so it is not usable
# where the numbers must be comparable.
#
# Pinned to NLA_GPU deliberately; sets no idleness gate, so confirm the card is free first.
set -euo pipefail
cd "$(dirname "$0")/.."          # nla/

ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
GPU="${NLA_GPU:-0}"
PORT="${NLA_PORT:-30011}"
CONFIG="${NLA_CONFIG:-configs/b1_transfer_gate.yaml}"
OUT_DIR="${OUT_DIR:-/data/jvl210002/my_downloads/transcoders/data/nla/b1}"
SUBJECT="${NLA_SUBJECT:-Qwen/Qwen2.5-Coder-7B-Instruct}"
CONTROL="${NLA_CONTROL:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p "$OUT_DIR"

echo "[b1] $(date -Is) gpu=$GPU port=$PORT config=$CONFIG args=$*"
echo "[b1] subject=$SUBJECT"
echo "[b1] control=$CONTROL"

CUDA_VISIBLE_DEVICES="$GPU" setsid python -m sglang.launch_server \
  --model-path data/checkpoints/av --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.40 \
  > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/server.pid"
cleanup() {
  echo "[b1] cleanup: killing server session (pgid $SERVER_PID)"
  kill -- -"$SERVER_PID" 2>/dev/null || true
  sleep 2
  # bracketed so the pattern cannot match this script's own command line and kill the shell
  pkill -9 -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[b1] server ready after ~$((i*5))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[b1] ABORT: server died during startup — see $OUT_DIR/server.log"; exit 3
  fi
  if (( i == 60 )); then echo "[b1] ABORT: server not healthy after 300s"; exit 3; fi
  sleep 5
done

# Control arm FIRST. If the NLA cannot reproduce its own host's numbers on this corpus, the
# subject arm is uninterpretable and the run should stop before spending the second hour —
# the same "control condition is the tripwire" rule that caught four bugs in B4.
for arm in "control:$CONTROL" "subject:$SUBJECT"; do
  TAG="${arm%%:*}"; MODEL="${arm#*:}"
  echo "[b1] === $TAG arm: $MODEL ==="
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" python -m src.transfer_gate run \
    --config "$CONFIG" --model "$MODEL" --tag "$TAG" \
    --ar-device cpu --sglang-url "http://localhost:$PORT" "$@" 2>&1 \
    | tee -a "$OUT_DIR/runner_$TAG.log"
  CODE=${PIPESTATUS[0]}
  set -e
  if [ "$CODE" -ne 0 ]; then
    echo "[b1] $TAG arm exited $CODE — stopping before the comparison"; exit "$CODE"
  fi
done

echo "[b1] === gate table ==="
set +e
python -m src.transfer_gate compare --config "$CONFIG" --a subject --b control 2>&1 \
  | tee -a "$OUT_DIR/gate.log"
GATE=${PIPESTATUS[0]}
set -e
echo "[b1] gate exited $GATE (0 = pass, 1 = fail)"
exit "$GATE"
