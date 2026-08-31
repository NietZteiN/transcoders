#!/usr/bin/env bash
# N10b benign arm, queued behind B4 on the same GPU.
#
#   tmux new-session -d -s n10b 'bash nla/scripts/n10b_after_b4.sh'
#
# Waits for the B4 tmux session to end (liveness, not log text — an append-only log keeps a
# previous run's terminal word and a grep for it fires immediately), then waits for the card to
# actually free before claiming it, then runs the benign capture in both framing arms so the
# malicious/benign contrast is measured under identical prompts.
set -uo pipefail
cd "$(dirname "$0")/.."          # nla/
PROJ=/data/jvl210002/my_downloads/transcoders
OUT=$PROJ/data/nla/n10b
GPU="${NLA_GPU:-2}"
LOG=$OUT/n10b.log
mkdir -p "$OUT"
say(){ echo "[n10b $(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for B4 to finish"
for i in $(seq 1 720); do
  tmux has-session -t b4 2>/dev/null || { say "B4 done"; break; }
  (( i == 720 )) && { say "timeout waiting for B4"; exit 1; }
  sleep 60
done

say "waiting for GPU $GPU to free"
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
  [ "$used" -lt 2000 ] && { say "gpu $GPU free (${used} MiB)"; break; }
  (( i == 60 )) && { say "gpu $GPU still busy (${used} MiB)"; exit 1; }
  sleep 10
done

for arm in security neutral; do
  say "benign capture, arm=$arm"
  NLA_GPU="$GPU" NLA_PORT=30005 OUT_DIR="$OUT" bash scripts/deception_1gpu_generic.sh \
    --corpus "$OUT/benign_corpus.jsonl" --arm "$arm" \
    --out "$OUT/benign_reads_${arm}.jsonl" --max-hours 6 >> "$LOG" 2>&1 \
    || { say "FAILED on arm=$arm"; exit 1; }
  n=$(wc -l < "$OUT/benign_reads_${arm}.jsonl" 2>/dev/null || echo 0)
  say "arm=$arm captured $n rows"
done
say "DONE — benign arms in $OUT"
