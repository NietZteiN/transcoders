#!/usr/bin/env bash
# Unattended B3 coupling pipeline, pinned to ONE GPU.
#
#   tmux new-session -d -s nla-autopilot 'bash nla/scripts/autopilot.sh'
#
# Stages, each skipped if its output already exists so the whole thing is restartable:
#   1. wait for the behavioural screen (launched separately) to finish
#   2. select deceived items + length/algorithm-matched controls
#   3. capture reads on that set only          (the one GPU stage)
#   4. score with the frozen defences
#
# Everything is logged to autopilot.log with timestamps. No stage needs a human, and any stage
# that fails stops the chain rather than feeding a downstream stage garbage — a half-finished
# capture scored as if complete is exactly the kind of silent error this project keeps finding.
set -uo pipefail
cd "$(dirname "$0")/.."          # nla/

PROJ=/data/jvl210002/my_downloads/transcoders
OUT=$PROJ/data/nla/n11
ENV=/data/jvl210002/conda_envs/nla-mi
STATS_ENV=/data/jvl210002/conda_envs/transcoders-mi
GPU="${NLA_GPU:-0}"
LOG=$OUT/autopilot.log
mkdir -p "$OUT"

say() { echo "[autopilot $(date -Is)] $*" | tee -a "$LOG"; }
die() { say "FAILED at: $*"; exit 1; }

say "start (gpu=$GPU)"

# ── 1. wait for the screen ────────────────────────────────────────────────
say "stage 1: waiting for screen.jsonl to reach 1724 rows"
for i in $(seq 1 720); do          # up to 12 h
  n=$(wc -l < "$OUT/screen.jsonl" 2>/dev/null || echo 0)
  if [ "$n" -ge 1724 ]; then say "stage 1: screen complete ($n rows)"; break; fi
  if (( i % 30 == 0 )); then say "stage 1: $n/1724"; fi
  if (( i == 720 )); then die "stage 1 timeout ($n/1724)"; fi
  sleep 60
done

# The row count reaches 1724 while the screen process is still shutting down and still holding
# ~25 GB on this card. Starting the AV server (~14 GB) plus the runner (~27 GB) on top of that
# would OOM, and the failure would look like a mysterious server crash rather than a race. Wait
# for the memory to actually come back before claiming the GPU.
say "stage 1b: waiting for GPU $GPU memory to free (screen still exiting)"
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
  if [ "$used" -lt 2000 ]; then say "stage 1b: gpu $GPU free (${used} MiB)"; break; fi
  if (( i == 60 )); then die "stage 1b: gpu $GPU still busy (${used} MiB) after 10 min"; fi
  sleep 10
done

# ── 2. select the coupling set ────────────────────────────────────────────
if [ -s "$OUT/coupling_snippets.txt" ]; then
  say "stage 2: selection already present, skipping"
else
  say "stage 2: selecting deceived items + matched controls"
  LD_LIBRARY_PATH="$ENV/lib" "$ENV/bin/python" -m src.select_coupling_set \
    >> "$LOG" 2>&1 || die "stage 2 (select_coupling_set)"
fi
N_SEL=$(wc -l < "$OUT/coupling_snippets.txt" 2>/dev/null || echo 0)
say "stage 2: $N_SEL snippets selected"
if [ "$N_SEL" -lt 6 ]; then
  say "stage 2: too few deceived items ($N_SEL) for a coupling test — stopping before spending"
  say "         GPU time. The honest read is that this corpus has no behavioural variance."
  exit 0
fi

# ── 3. targeted read capture, one GPU ─────────────────────────────────────
say "stage 3: capture on $N_SEL snippets x 3 conditions (gpu $GPU)"
NLA_GPU="$GPU" OUT_DIR="$OUT" bash scripts/deception_1gpu.sh \
  --snippets "$OUT/coupling_snippets.txt" \
  --out "$OUT/coupling_reads.jsonl" --max-hours 11 >> "$LOG" 2>&1 || die "stage 3 (capture)"
N_GOT=$(wc -l < "$OUT/coupling_reads.jsonl" 2>/dev/null || echo 0)
N_WANT=$(( N_SEL * 3 ))
say "stage 3: captured $N_GOT rows (expected $N_WANT)"
# The capture returns 0 when it stops on its wall-clock guard, so a successful exit does NOT
# mean a complete run. Scoring a truncated capture as if finished is precisely the silent
# failure this chain is meant to avoid, so compare counts and let the resume path finish the
# job on the next launch rather than reporting half a result.
if [ "$N_GOT" -lt "$N_WANT" ]; then
  say "stage 3: INCOMPLETE ($N_GOT/$N_WANT) — probably the wall guard. Not scoring."
  say "         Re-run this script to resume; the capture is keyed and idempotent."
  exit 1
fi

# ── 4. score ──────────────────────────────────────────────────────────────
say "stage 4: scoring with the frozen defences"
LD_LIBRARY_PATH="$STATS_ENV/lib" "$STATS_ENV/bin/python" src/deception_stats.py \
  --reads "$OUT/coupling_reads.jsonl" --out "$OUT/coupling_stats.json" \
  >> "$LOG" 2>&1 || die "stage 4 (deception_stats)"

say "DONE — results in $OUT/coupling_stats.json"
