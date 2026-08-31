#!/usr/bin/env bash
# P0.3 — is layer 20 a live site for ANY intervention?
# Pre-registration: log/nla-harness/2026-08-27_p0-triage-prereg.md
#
#   tmux new-window -t p0 -n p03 'bash nla/scripts/p03_site.sh'
#
# Attention steering restricted to explicit layer bands, belief channel OFF (alpha=0 throughout,
# so this is purely the positional lever). A scope CURVE rather than a single point, because a
# lone null cannot distinguish a dead site from a lever below its operating point.
#
# TWO SEEDS BY CONSTRUCTION. E5b (2026-08-26) showed a five-transform family average flipping
# sign between draws on this exact pipeline, so a one-seed cell here would be uninterpretable.
# The unsteered baseline is re-run at BOTH seeds in this same batch rather than borrowed from
# B5/B0: the decision rule compares each band against the baseline's own seed-to-seed spread,
# which cannot be computed from someone else's run.
#
# Cells are idempotent (.done markers), so the script is restartable after an interruption.
set -uo pipefail
cd "$(dirname "$0")/../.."          # transcoders/

PROJ=/data/jvl210002/my_downloads/transcoders
REPL=/data/jvl210002/my_downloads/allocation_replication
CS_ENV=/data/jvl210002/conda_envs/codesteer
OUT=$PROJ/data/nla/p0/p03
LOG=$OUT/p03.log
mkdir -p "$OUT"

export HF_HOME=/data/jvl210002/my_downloads/.cache/huggingface
export TMPDIR=/data/jvl210002/tmp_pip
export JAVA_HOME=/data/jvl210002/my_downloads/tools/jdk-21.0.4+7
export PATH="$JAVA_HOME/bin:$PATH"
export OBF_JAVA_CLASSPATH=$REPL/artifact/PromptSteering/java_compat

say() { echo "[p03 $(date -Is)] $*" | tee -a "$LOG"; }

ATTN_BASE="--prior slice --beta-post 0.8 --head-subset-mode none --n-bins 8"

# name|steer-args|seed
CELLS=(
  "base_s1000||1000"
  "base_s2000||2000"
  "L20_s1000|--steer $ATTN_BASE --steer-layers 20:20|1000"
  "L20_s2000|--steer $ATTN_BASE --steer-layers 20:20|2000"
  "L2021_s1000|--steer $ATTN_BASE --steer-layers 20:21|1000"
  "L2021_s2000|--steer $ATTN_BASE --steer-layers 20:21|2000"
  "L2023_s1000|--steer $ATTN_BASE --steer-layers 20:23|1000"
  "L2023_s2000|--steer $ATTN_BASE --steer-layers 20:23|2000"
  "L2027_s1000|--steer $ATTN_BASE --steer-layers 20:27|1000"
  "L2027_s2000|--steer $ATTN_BASE --steer-layers 20:27|2000"
)

run_cell() {   # $1 name  $2 steer-args  $3 seed  $4 gpu
  local name="$1" steer="$2" seed="$3" gpu="$4"
  if [ -f "$OUT/cell_${name}.done" ]; then say "$name already done, skipping"; return 0; fi
  say "$name starting on GPU $gpu (seed $seed)"
  cd "$REPL/artifact"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" "$CS_ENV/bin/python" obfuscation/main.py \
      --dataset humaneval --source-root "$REPL/data/obf/humaneval" \
      --techniques adversarial_rename --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
      --cache-dir "$HF_HOME/hub" --runs-per-snippet 3 --temperature 0.9 --top-p 0.95 \
      --max-new-tokens 512 --seed "$seed" --record-layers off --visual-dump off \
      $steer --run-tag "p03_${name}" \
      >> "$OUT/${name}.log" 2>&1 \
    && touch "$OUT/cell_${name}.done" && say "$name complete" \
    || say "$name FAILED (see ${name}.log)"
  cd "$PROJ"
}

# Two workers, one per free GPU. GPU 1 holds an unrelated vLLM job and GPU 2 holds P0.2;
# neither is touched. Cells are split round-robin so the two seeds of a band land on
# different cards and a card-specific problem cannot masquerade as a seed effect.
worker() {
  local gpu="$1" offset="$2"
  for ((i=offset; i<${#CELLS[@]}; i+=2)); do
    IFS='|' read -r n s sd <<< "${CELLS[$i]}"
    run_cell "$n" "$s" "$sd" "$gpu"
  done
  say "worker on GPU $gpu done"
}

say "start — ${#CELLS[@]} cells across GPUs 0 and 3"
worker 0 0 &
worker 3 1 &
wait
say "all cells finished; scoring"
cd "$REPL" && "$CS_ENV/bin/python" -m pipeline.analysis.aggregate \
    --out results/tables/p03_grid.csv >> "$LOG" 2>&1 \
  && say "wrote results/tables/p03_grid.csv" || say "aggregate FAILED"
say "done"
