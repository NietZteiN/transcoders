#!/usr/bin/env bash
# Phase-0 triage, end to end, unattended and logout-proof.
# Pre-registration: log/nla-harness/2026-08-27_p0-triage-prereg.md
#
#   setsid nohup bash nla/scripts/p0_autopilot.sh >/dev/null 2>&1 &
#
# WHY setsid AND NOT JUST tmux. `Linger=yes` is set for this user so the tmux server does survive
# logout here, but that is a property of the box's logind config, not of the job. setsid detaches
# the process group from any controlling terminal outright, so the chain cannot be taken down by a
# SIGHUP, a killed tmux server, or a dropped SSH connection. tmux stays useful for WATCHING; it is
# no longer the thing keeping the run alive.
#
# ADOPTS WORK IN FLIGHT. P0.2 and P0.3 were launched by hand before this script existed. It does
# not restart them and does not race them: it waits for each to exit, then re-runs the idempotent
# stage once to pick up any cell that failed, then scores. Every stage is skipped if its output
# already exists, so this is safe to re-run at any point.
set -uo pipefail
cd "$(dirname "$0")/../.."          # transcoders/

PROJ=/data/jvl210002/my_downloads/transcoders
REPL=/data/jvl210002/my_downloads/allocation_replication
NLA_ENV=/data/jvl210002/conda_envs/nla-mi
CS_ENV=/data/jvl210002/conda_envs/codesteer
OUT=$PROJ/data/nla/p0
LOG=$OUT/autopilot.log
LOCK=$OUT/.autopilot.lock
P02_DIR=$OUT/p02_allreply
P03_DIR=$OUT/p03
WITH_L13=${P0_WITH_L13:-0}          # opt-in; see the note at stage 4

mkdir -p "$OUT"
export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip

say() { echo "[p0 $(date -Is)] $*" | tee -a "$LOG"; }

# Single instance. mkdir is atomic on this filesystem; a stale lock from a killed run is cleared
# by hand rather than automatically, because auto-clearing a lock is how two runs end up writing
# the same JSONL.
if ! mkdir "$LOCK" 2>/dev/null; then
  say "another autopilot holds $LOCK — exiting rather than racing it"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say "start (pid $$, WITH_L13=$WITH_L13)"

wait_for_exit() {   # $1 pgrep pattern  $2 human label  $3 max minutes
  local pat="$1" label="$2" maxmin="${3:-720}" i=0
  if ! pgrep -f "$pat" >/dev/null 2>&1; then say "$label: not running"; return 0; fi
  say "$label: in flight, waiting"
  while pgrep -f "$pat" >/dev/null 2>&1; do
    sleep 60; i=$((i+1))
    (( i % 30 == 0 )) && say "$label: still running (${i} min)"
    if (( i > maxmin )); then say "$label: exceeded ${maxmin} min — giving up on it"; return 1; fi
  done
  say "$label: exited"
}

# ── 1. P0.1 — layer rotation (already complete; verify rather than assume) ─────────────
if [ -s "$OUT/layer_rotation.json" ]; then
  say "stage 1 (P0.1): present — $("$NLA_ENV/bin/python" -c "
import json;d=json.load(open('$OUT/layer_rotation.json'))
print(f\"verdict={d['verdict']} min_rot={d['min_rotation_layers_21_27']:.3f} argmax_coh=L{d['argmax_coherence_layer']}\")")"
else
  say "stage 1 (P0.1): missing — running"
  # pick the emptiest card rather than assuming one is free
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2 -n | head -1 | cut -d, -f1)
  CUDA_VISIBLE_DEVICES="${G:-0}" "$NLA_ENV/bin/python" \
    nla/src/layer_rotation.py >> "$OUT/p01.log" 2>&1 || say "stage 1 FAILED"
fi

# ── 2. P0.2 — oracle at all_reply ──────────────────────────────────────────────────────
wait_for_exit "steer_run.py.*p02_allreply" "stage 2 (P0.2)" 480
if [ -s "$P02_DIR/steer_results.jsonl" ]; then
  say "stage 2: scoring"
  "$NLA_ENV/bin/python" nla/src/steer_stats.py --primary-alpha 1.0 \
      --results  "$P02_DIR/steer_results.jsonl" \
      --baseline "$P02_DIR/baseline.jsonl" \
      --out      "$P02_DIR/steer_stats.json" \
      >> "$OUT/p02_score.log" 2>&1 \
    && say "stage 2: scored -> $P02_DIR/steer_stats.json" \
    || say "stage 2: scoring FAILED (see p02_score.log)"
else
  say "stage 2: no results at $P02_DIR — SKIPPED"
fi

# ── 3. P0.3 — attention scope curve ────────────────────────────────────────────────────
wait_for_exit "p03_site.sh" "stage 3 (P0.3)" 720
ndone=$(ls "$P03_DIR"/cell_*.done 2>/dev/null | wc -l)
say "stage 3: $ndone/10 cells complete"
if [ "$ndone" -lt 10 ]; then
  say "stage 3: re-running the idempotent driver to pick up missing cells"
  bash nla/scripts/p03_site.sh >> "$LOG" 2>&1
  ndone=$(ls "$P03_DIR"/cell_*.done 2>/dev/null | wc -l)
  say "stage 3: now $ndone/10"
fi
say "stage 3: scoring cells"
"$NLA_ENV/bin/python" nla/src/p03_score.py --out "$P03_DIR/cells_scored.json" \
    >> "$OUT/p03_score.log" 2>&1 \
  && say "stage 3: scored -> $P03_DIR/cells_scored.json" \
  || say "stage 3: scoring FAILED (see p03_score.log)"

# ── 4. OPTIONAL P0.4 — V3/V4 at layer 13 vs 20 ─────────────────────────────────────────
# P0.1 found the cross-item coherence of the task direction peaks at layer 13, not at the
# instrument's layer 20. V3 and V4 are pure activation differences and ARE defined at every
# layer, so this is testable without any autoencoder. It is NOT part of the frozen Phase-0
# pre-registration, so it is opt-in and must get its own decision rule before it is believed:
#     P0_WITH_L13=1 setsid nohup bash nla/scripts/p0_autopilot.sh >/dev/null 2>&1 &
if [ "$WITH_L13" = "1" ]; then
  say "stage 4 (P0.4): layer-13 task vector — EXPLORATORY, not in the frozen prereg"
  say "stage 4: not implemented in this revision; declare its rule first"
else
  say "stage 4 (P0.4): not enabled (set P0_WITH_L13=1 after pre-registering it)"
fi

# ── 5. summary ─────────────────────────────────────────────────────────────────────────
say "stage 5: summary"
"$NLA_ENV/bin/python" nla/src/p0_summary.py --out "$OUT/P0_SUMMARY.md" >> "$LOG" 2>&1 \
  && say "wrote $OUT/P0_SUMMARY.md" || say "stage 5: summary FAILED"

say "ALL DONE"
