#!/usr/bin/env bash
# N13 stages 2, 3 and 4 back to back, on ONE GPU, unattended.
#
#   NLA_GPU=1 setsid nohup bash nla/scripts/n13_autopilot.sh > .../autopilot.log 2>&1 &
#
# `setsid` matters: it makes this a session leader with no controlling terminal, so SIGHUP at
# logoff cannot reach it. Verify with `ps -o pid,sid,tty` — sid should equal pid and tty '?'.
#
# Waits for the Stage-1 tmux session to end first, so all stages share ONE card rather than
# racing each other for it. Liveness is checked with `tmux has-session`, NOT by grepping the log
# for a terminal word — the logs are append-only and a previous run's "done" line has twice been
# matched by a grep meant for the current run.
#
# No `set -e`: a failed stage must not silently cancel the ones after it. Each stage records its
# own exit code, and every stage is resumable (JsonlSink skips completed keys), so a rerun after
# a crash costs only the unfinished work.
set -uo pipefail
cd "$(dirname "$0")/.."          # nla/
GPU="${NLA_GPU:-1}"
OUT_DIR=/data/jvl210002/my_downloads/transcoders/data/nla/n13
ENV=/data/jvl210002/conda_envs/nla-mi
ANALYSIS_ENV=/data/jvl210002/conda_envs/transcoders-mi
mkdir -p "$OUT_DIR"

echo "[auto] $(date -Is) pid=$$ gpu=$GPU — waiting for Stage 1 (tmux n13s1)"
while tmux has-session -t n13s1 2>/dev/null; do sleep 60; done
echo "[auto] $(date -Is) Stage 1 finished"

echo "[auto] === Stage 2: act_norm recovery (~5 min) ==="
LD_LIBRARY_PATH="$ENV/lib" HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface \
CUDA_VISIBLE_DEVICES="$GPU" "$ENV/bin/python" -m src.recover_act_norm \
  > "$OUT_DIR/stage2.log" 2>&1
echo "[auto] $(date -Is) Stage 2 exit $?"

echo "[auto] === Stage 3: read instability (~2.2 h) ==="
NLA_GPU="$GPU" NLA_PORT=30007 bash scripts/n13_stage3.sh \
  --k 5 --temperature 0.7 --max-hours 6 > "$OUT_DIR/stage3.log" 2>&1
echo "[auto] $(date -Is) Stage 3 exit $?"

# Stage 4 is CPU-only and takes seconds, but it lives here so the pipeline produces its ANSWER
# unattended rather than a directory of inputs waiting for someone to run the analysis by hand.
echo "[auto] === Stage 4: analysis (seconds, CPU) ==="
cd /data/jvl210002/my_downloads/transcoders
"$ANALYSIS_ENV/bin/python" -m src.analysis.n13_torn > "$OUT_DIR/stage4.log" 2>&1
echo "[auto] $(date -Is) Stage 4 exit $?"

echo "[auto] $(date -Is) ALL STAGES DONE"
