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
# No sglang. Job 376102 died in 40 s because sglang is installed in none of the three juno envs
# (it did not survive the cluster migration). The AV now runs in-process via nla/src/local_av.py,
# which reuses the vendored NLAClient._build_embeds injection path verbatim and only replaces the
# HTTP POST with model.generate(inputs_embeds=...). See that module for why (b) beat installing it.
#
# THREE 12B models on one card, all in this process: subject (gemma-3-12b-it), AV, AR — roughly
# 24 GB each in bf16, ~72 GB. h200 (141 GB) holds them with room to spare; on an 80 GB card the
# AR would have to go to CPU (--ar-device cpu).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
OUT="$PROJ/data/nla/p0/nla_cycle/$HOST"; mkdir -p "$OUT"

echo "# Experiment W stage 0 — cycle-consistency gate · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_cycle.py nla/src/local_av.py nla/src/span_positions.py nla/src/extract.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

# FIRST USE OF THE GEMMA AV ANYWHERE IN THIS PROJECT. All 16 other AV call sites hardcode the
# Qwen checkpoint; injection_scale is 80000 here vs Qwen's 150. If that is mishandled the
# verbalizer free-associates in CJK instead of failing, which the script aborts on (exit 3) —
# a broken instrument, recorded as such, never as a null.

# Smoke first (project rule): 3 items, in-script assertions on CJK rate and matched cos.
echo; echo "=== SMOKE (3 items) ==="
python nla/src/nla_cycle.py --model "$HOST" --smoke --out-dir "$OUT/smoke" || exit 1

echo; echo "=== FULL (60 items) ==="
python nla/src/nla_cycle.py --model "$HOST" --out-dir "$OUT" --max-hours 4 || exit 1

echo; echo "=== GATE ==="
python - <<'EOF'
import json
s = json.load(open("data/nla/p0/nla_cycle/gemma12b/cycle_stats.json"))
print(f"verdict={s['verdict']}  W0a={s['H_W0a_cross_item_separation']}  "
      f"W0b={s['H_W0b_within_item_separation']}  licenses_stage1={s['licenses_stage1']}")
EOF
echo "# done $(date -u +%FT%TZ)"
