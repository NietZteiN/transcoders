#!/usr/bin/env bash
# N5 driver — owns GPU selection and the AV server lifecycle. tmux-ready:
#   tmux new-session -d -s nla-dense 'bash nla/scripts/dense.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=/data/jvl210002/conda_envs/nla-mi
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export PATH="$ENV/bin:$PATH"
OUT_DIR="${OUT_DIR:-$(cd .. && pwd)/data/nla/n5/$(date +%F)}"
mkdir -p "$OUT_DIR"

# --- pick two idle GPUs (shared box: never take one another job is using) ---
mapfile -t IDLE < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', *' '$2 < 1000 && $3 < 10 {print $1}')
if [ "${#IDLE[@]}" -lt 2 ]; then
  echo "[driver] need 2 idle GPUs, found ${#IDLE[@]} — aborting (shared box)"; exit 3
fi
GPU_RUN="${IDLE[0]}"; GPU_SRV="${IDLE[1]}"
echo "[driver] runner=GPU$GPU_RUN  AV server=GPU$GPU_SRV  out=$OUT_DIR"

# --- AV server, default config (the regime that reproduces the banked corpus) ---
CUDA_VISIBLE_DEVICES="$GPU_SRV" setsid "$ENV/bin/python" -m sglang.launch_server \
  --model-path data/checkpoints/av --port 30000 \
  --disable-radix-cache --mem-fraction-static 0.6 \
  > "$OUT_DIR/server.log" 2>&1 &
SRV_PGID=$!
cleanup() { echo "[driver] cleanup: killing server pgid $SRV_PGID"; kill -- -"$SRV_PGID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for i in $(seq 1 48); do
  curl -s http://localhost:30000/health >/dev/null 2>&1 && { echo "[driver] AV ready (~$((i*5))s)"; break; }
  if ! kill -0 "$SRV_PGID" 2>/dev/null; then echo "[driver] server died during startup"; exit 4; fi
  sleep 5
done

export NLA_SERVER_GPU="$GPU_SRV"
CUDA_VISIBLE_DEVICES="$GPU_RUN" "$ENV/bin/python" -m src.dense_capture \
  --out-dir "$OUT_DIR" "$@" 2>&1 | tee -a "$OUT_DIR/runner.log"
rc=${PIPESTATUS[0]}
echo "[driver] runner exited $rc — summary at $OUT_DIR/summary.md"
exit $rc
