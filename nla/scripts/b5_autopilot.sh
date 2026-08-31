#!/usr/bin/env bash
# B5 / E6 — the composed-lever experiment, end to end and unattended.
#
#   tmux new-session -d -s b5 'bash nla/scripts/b5_autopilot.sh'
#   (or run under systemd; see b5-autopilot.service)
#
# Stages, each skipped if its output already exists so the whole thing is restartable:
#   1. export per-snippet steering directions          (nla-mi, 1 GPU, ~25 min)
#   2. THE VALIDATION GATE — alpha=0 with attention ON must reproduce the banked
#      obf_steer:adversarial_rename accuracy to within +/-0.02                (codesteer, 1 GPU)
#   3. the six composed conditions                     (codesteer, all free GPUs)
#   4. score
#
# THE GATE IS A HARD STOP, NOT A WARNING. If stage 2 misses, the integration is wrong and
# every downstream number would be a measurement of the bug rather than of the levers. The
# script exits non-zero and runs nothing further. That is deliberate: the project's standing
# rule is that B5 gets dropped rather than debugged into existence, and an autopilot that
# "carries on anyway" would convert a caught error into a published one.
#
# Everything is logged with timestamps. No stage needs a human. A stage that fails stops the
# chain rather than feeding a downstream stage garbage.
set -uo pipefail
cd "$(dirname "$0")/.."          # nla/

PROJ=/data/jvl210002/my_downloads/transcoders
REPL=/data/jvl210002/my_downloads/allocation_replication
NLA_ENV=/data/jvl210002/conda_envs/nla-mi
CS_ENV=/data/jvl210002/conda_envs/codesteer
DIRS=$PROJ/data/nla/b5_directions
OUT=$PROJ/data/nla/b5
LOG=$OUT/autopilot.log
GATE_TOL=${GATE_TOL:-0.02}
ALPHA=${B5_ALPHA:-1.0}
mkdir -p "$OUT"

say() { echo "[b5 $(date -Is)] $*" | tee -a "$LOG"; }
die() { say "FAILED at: $*"; exit 1; }

export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export JAVA_HOME=/data/jvl210002/my_downloads/tools/jdk-21.0.4+7
export PATH="$JAVA_HOME/bin:$PATH"
export OBF_JAVA_CLASSPATH=$REPL/artifact/PromptSteering/java_compat

say "start (alpha=$ALPHA, gate tolerance +/-$GATE_TOL)"

# ── wait for a genuinely idle GPU ─────────────────────────────────────────────────────
# Shared box, no scheduler. Never launch onto a card another job is using.
free_gpu() {
  for i in $(seq 1 720); do          # up to 12 h
    for g in 0 1 2 3; do
      m=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')
      [ "${m:-99999}" -lt 2000 ] && { echo "$g"; return 0; }
    done
    (( i % 30 == 0 )) && say "waiting for a free GPU ($((i/2)) min)"
    sleep 30
  done
  return 1
}

# ── 1. directions ─────────────────────────────────────────────────────────────────────
if [ -f "$DIRS/v1.npz" ] && [ -f "$DIRS/manifest.json" ]; then
  say "stage 1: directions already exported, skipping"
else
  G=$(free_gpu) || die "stage 1: no free GPU"
  say "stage 1: exporting directions on GPU $G"
  LD_LIBRARY_PATH="$NLA_ENV/lib" CUDA_VISIBLE_DEVICES="$G" \
    "$NLA_ENV/bin/python" -m src.export_directions --dataset humaneval --out "$DIRS" \
    >> "$OUT/stage1_export.log" 2>&1 || die "stage 1 (see stage1_export.log)"
  say "stage 1: done"
fi
python3 -c "
import json,sys; m=json.load(open('$DIRS/manifest.json'))
print(f\"[b5] directions: {m['n_v1']} v1 / {m['n_v3']} v3 / {m['n_v4']} v4\")
sys.exit(0 if m['n_v1'] > 0 else 1)" | tee -a "$LOG" || die "stage 1: no v1 directions"

# ── 2. THE GATE ───────────────────────────────────────────────────────────────────────
# alpha=0 makes the write hook a verified no-op, so this run must land on the banked
# attention-only number. Anything else means the patched runner is not the runner that
# produced those numbers, and no composed result from it would mean anything.
if [ -f "$OUT/gate_verdict.json" ]; then
  say "stage 2: gate already evaluated, skipping"
else
  G=$(free_gpu) || die "stage 2: no free GPU"
  say "stage 2: VALIDATION GATE on GPU $G (alpha=0, attention ON)"
  cd "$REPL/artifact"
  CUDA_VISIBLE_DEVICES="$G" "$CS_ENV/bin/python" obfuscation/main.py \
      --dataset humaneval --source-root "$REPL/data/obf/humaneval" \
      --techniques adversarial_rename --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
      --cache-dir "$HF_HOME/hub" --runs-per-snippet 3 --temperature 0.9 --top-p 0.95 \
      --max-new-tokens 512 --seed 1000 --record-layers off --visual-dump off \
      --steer --prior slice --beta-post 0.8 --steer-last-n-layers 8 \
      --head-subset-mode none --n-bins 8 \
      --nla-dir "$DIRS" --nla-condition v1 --nla-alpha 0.0 \
      --run-tag b5_gate_alpha0 \
      >> "$OUT/stage2_gate.log" 2>&1 || die "stage 2 run (see stage2_gate.log)"
  cd "$PROJ/nla"
  "$CS_ENV/bin/python" "$PROJ/nla/src/b5_gate.py" --out "$OUT/gate_verdict.json" \
      --tol "$GATE_TOL" >> "$LOG" 2>&1 || die "stage 2 scoring"
fi
PASS=$(python3 -c "import json;print(json.load(open('$OUT/gate_verdict.json'))['pass'])")
if [ "$PASS" != "True" ]; then
  say "GATE FAILED — stopping. Per the pre-registration B5 is dropped rather than debugged"
  say "into existence. See $OUT/gate_verdict.json"
  exit 2
fi
say "stage 2: GATE PASSED"

# ── 3. the six conditions ─────────────────────────────────────────────────────────────
# C_disjoint exists because attention steering occupies layers 20-27 and the NLA lives at
# layer 20, so a naive "both on" cell confounds composition with collision. Running the
# attention arm on 21-27 separates them; it is a one-flag change and it is the difference
# between a mechanism claim and a coincidence.
run_cell () {  # name  steer_flags  nla_cond  nla_alpha
  local name=$1 steer=$2 cond=$3 alpha=$4
  if [ -d "$REPL/artifact/obfuscation/result" ] && [ -f "$OUT/cell_${name}.done" ]; then
    say "stage 3: $name already done, skipping"; return 0; fi
  local G; G=$(free_gpu) || die "stage 3 ($name): no free GPU"
  say "stage 3: $name on GPU $G"
  cd "$REPL/artifact"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$G" "$CS_ENV/bin/python" obfuscation/main.py \
      --dataset humaneval --source-root "$REPL/data/obf/humaneval" \
      --techniques adversarial_rename --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
      --cache-dir "$HF_HOME/hub" --runs-per-snippet 3 --temperature 0.9 --top-p 0.95 \
      --max-new-tokens 512 --seed 1000 --record-layers off --visual-dump off \
      $steer --nla-dir "$DIRS" --nla-condition "$cond" --nla-alpha "$alpha" \
      --run-tag "b5_${name}" \
      >> "$OUT/stage3_${name}.log" 2>&1 || die "stage 3 cell $name"
  touch "$OUT/cell_${name}.done"; say "stage 3: $name complete"
}

ATTN="--steer --prior slice --beta-post 0.8 --steer-last-n-layers 8 --head-subset-mode none --n-bins 8"
ATTN_DISJOINT="--steer --prior slice --beta-post 0.8 --steer-last-n-layers 7 --head-subset-mode none --n-bins 8"

run_cell none      ""                 v1     0.0
run_cell attn      "$ATTN"            v1     0.0
run_cell nla       ""                 v1     "$ALPHA"
run_cell both      "$ATTN"            v1     "$ALPHA"
run_cell rand      "$ATTN"            random "$ALPHA"
run_cell disjoint  "$ATTN_DISJOINT"   v1     "$ALPHA"

# ── 4. score ──────────────────────────────────────────────────────────────────────────
say "stage 4: scoring"
cd "$REPL"
"$CS_ENV/bin/python" -m pipeline.analysis.aggregate --out results/tables/b5_grid.csv \
    >> "$LOG" 2>&1 || die "stage 4 aggregate"
say "done. cells in $OUT, table in $REPL/results/tables/b5_grid.csv"
