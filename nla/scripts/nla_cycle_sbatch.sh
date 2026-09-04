#!/bin/bash
#SBATCH --job-name=nla_cycle
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_cycle.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_cycle.out
# Experiment W, stage 0 — the cycle-consistency gate.
# Pre-registered: log/nla-harness/2026-09-03_nla-writeback-prereg.md
#
# THREE 12B models share one card: the subject (gemma-3-12b-it, ~24 GB bf16), the AV served by
# sglang, and the AR. h200 (141 GB) is requested for that reason and mem-fraction-static is held
# to 0.30 (~42 GB) so the two HF models have room. On an 80 GB card this does not fit and the AR
# must go to CPU (--ar-device cpu), which is slower but correct.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
PORT="${NLA_PORT:-30021}"
OUT="$PROJ/data/nla/p0/nla_cycle/$HOST"; mkdir -p "$OUT"
AV="$(python - <<'EOF'
import sys; sys.path.insert(0, "nla/src"); sys.path.insert(0, "nla/vendor/nla-repo")
from nla_cycle import AV_CHECKPOINTS; print(AV_CHECKPOINTS["gemma12b"])
EOF
)"

echo "# Experiment W stage 0 — cycle-consistency gate · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "# AV checkpoint: $AV"
sha256sum nla/src/nla_cycle.py nla/src/span_positions.py nla/src/extract.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

# FIRST USE OF THE GEMMA AV ANYWHERE IN THIS PROJECT. All 16 other AV call sites hardcode the
# Qwen checkpoint. injection_scale is 80000 here vs Qwen's 150; if that is mishandled the
# verbalizer emits CJK free-association instead of failing, which the script aborts on (exit 3).
setsid python -m sglang.launch_server \
  --model-path "$AV" --port "$PORT" \
  --disable-radix-cache --mem-fraction-static 0.30 \
  > "$OUT/av_server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -- -"$SERVER_PID" 2>/dev/null || true
  sleep 2
  # bracketed so the pattern cannot match this script's own command line
  pkill -9 -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 120); do
  curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "# AV up after ~$((i*5))s"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "# AV SERVER DIED — see $OUT/av_server.log"; tail -30 "$OUT/av_server.log"; exit 1; }
  sleep 5
done

# Smoke first (project rule): 3 items, with in-script assertions on CJK rate and matched cos.
# A broken injection or a dead round trip stops the run here rather than after 60 items.
echo; echo "=== SMOKE (3 items) ==="
python nla/src/nla_cycle.py --model "$HOST" --smoke \
  --out-dir "$OUT/smoke" --sglang-url "http://localhost:$PORT" || exit 1

echo; echo "=== FULL (60 items) ==="
python nla/src/nla_cycle.py --model "$HOST" \
  --out-dir "$OUT" --sglang-url "http://localhost:$PORT" --max-hours 4 || exit 1

echo; echo "=== GATE ==="
python - <<'EOF'
import json
s = json.load(open("data/nla/p0/nla_cycle/gemma12b/cycle_stats.json"))
print(f"verdict={s['verdict']}  W0a={s['H_W0a_cross_item_separation']}  "
      f"W0b={s['H_W0b_within_item_separation']}  licenses_stage1={s['licenses_stage1']}")
EOF
echo "# done $(date -u +%FT%TZ)"
