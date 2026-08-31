#!/usr/bin/env bash
# B4 v2 — alpha grid and injection-position sweep, on ONE GPU.
#
#   NLA_GPU=0 bash nla/scripts/b4_sweep.sh                       # alpha grid, last_prompt
#   NLA_GPU=0 NLA_POSITIONS=all_reply bash nla/scripts/b4_sweep.sh --frozen-alpha 1.0
#
# The banked B4 run had NO driver script — it was launched by hand into tmux `b4`, which is why
# it had no idle-GPU gate and no server lifecycle. This is that driver.
#
# WHY AN AV SERVER AT ALL. V1/V3/V4 and the controls need only the AR (reconstruct a gloss,
# diff two activations), which loads in-process. V2 is also AR-only: the minimal single-word
# edit is applied to the gloss `load_pairs` already builds, not to a verbalized read. So this
# script starts a server purely so `--with-v2` keeps working if the read-and-edit variant is
# ever wired; if you are running without V2 you can skip it. Left in because starting it costs
# ~50 s and debugging a missing server at hour three costs more.
#
# MEMORY on a 48 GB A6000: AV sglang 0.40 -> ~19.2 GB, subject 7B bf16 ~17 GB, AR ~11 GB.
# That is 47.2 GB and does NOT fit, so the AR goes to CPU exactly as in deception_1gpu.sh.
# Measured cost of that choice elsewhere: ~5 s/read of CPU AR on top of the GPU work.
#
# RESUME. `steer_run.py` keys its JsonlSink on `snippet|condition|alpha`, suffixed with the
# position when it is not `last_prompt`, so re-running is free for anything already done — the
# 420 banked alpha=1.0 rows included. Relaunching after a crash costs nothing but the model load.
set -euo pipefail
cd "$(dirname "$0")/.."          # nla/

ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
GPU="${NLA_GPU:-0}"
PORT="${NLA_PORT:-30013}"
POSITIONS="${NLA_POSITIONS:-last_prompt}"
ALPHAS="${NLA_ALPHAS:-0.25,0.5,1.0,2.0,4.0}"
OUT_DIR="${OUT_DIR:-/data/jvl210002/my_downloads/transcoders/data/nla/n12}"
mkdir -p "$OUT_DIR"

echo "[b4] $(date -Is) gpu=$GPU port=$PORT positions=$POSITIONS alphas=$ALPHAS args=$*"

CUDA_VISIBLE_DEVICES="$GPU" setsid python -m sglang.launch_server \
  --model-path data/checkpoints/av --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.40 \
  > "$OUT_DIR/b4_server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/b4_server.pid"
cleanup() {
  echo "[b4] cleanup: killing server session (pgid $SERVER_PID)"
  kill -- -"$SERVER_PID" 2>/dev/null || true
  sleep 2
  # bracketed so the pattern cannot match this script's own command line and kill the shell
  pkill -9 -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[b4] server ready after ~$((i*5))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[b4] ABORT: server died during startup — see $OUT_DIR/b4_server.log"; exit 3
  fi
  if (( i == 60 )); then echo "[b4] ABORT: server not healthy after 300s"; exit 3; fi
  sleep 5
done

set +e
CUDA_VISIBLE_DEVICES="$GPU" python -m src.steer_run \
  --alphas "$ALPHAS" --positions "$POSITIONS" --with-v2 \
  --out-dir "$OUT_DIR" --sglang-url "http://localhost:$PORT" "$@" 2>&1 \
  | tee -a "$OUT_DIR/b4_runner_${POSITIONS}.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[b4] runner exited $CODE"

# Scoring is a separate env (statsmodels/scipy live in transcoders-mi, not nla-mi) and is the
# step that must NOT be skipped: with more than one alpha present, the pre-fix scorer silently
# overwrote each item's flag with whichever alpha came last.
echo "[b4] score with:"
echo "  /data/jvl210002/conda_envs/transcoders-mi/bin/python nla/src/steer_stats.py \\"
echo "      --results $OUT_DIR/steer_results.jsonl --baseline $OUT_DIR/baseline.jsonl \\"
echo "      --out $OUT_DIR/steer_stats_sweep.json --primary-alpha 1.0"
exit "$CODE"
