#!/bin/bash
#SBATCH --job-name=site_opt
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_site_opt.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_site_opt.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
HOST=gemma12b
R="$PROJ/data/nla/p0/trace_llr/$HOST"
OUT="$PROJ/data/nla/p0/site_opt/$HOST"; mkdir -p "$OUT" "$OUT/smoke"
echo "# Experiment S — optimised-vector site test · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/site_optimise.py nla/src/steer.py nla/src/trace_llr.py

# Pre-registered precondition: S does not run on an R-UNINF readout. Exit 2 = principled refusal.
V=$(python -c "import json,sys;print(json.load(open('$R/llr_stats.json'))['verdict'])") || exit 1
echo "# R verdict: $V"
if [[ "$V" == "R-UNINF" ]]; then echo "REFUSED: R-UNINF — M is not a sensitive readout; S is void by the frozen rule"; exit 2; fi

echo; echo "=== SMOKE (12 items, 1 split, 4 steps) ==="
python nla/src/site_optimise.py --model "$HOST" --traces "$R/traces.jsonl" --llr-stats "$R/llr_stats.json" \
  --out-dir "$OUT/smoke" --smoke --max-hours 1 || exit 1
python - "$OUT/smoke/splits.json" <<'PY' || exit 1
import json, sys
s = json.load(open(sys.argv[1]))[0]
c = s["delta"]["curve"]
print("smoke curve", [round(x, 4) for x in c])
assert len(c) == 4 and all(x == x for x in c), "optimiser produced no finite steps"
print("SMOKE OK")
PY

echo; echo "=== FULL (5 splits, 30/30, 40 steps, greedy on held-out) ==="
python nla/src/site_optimise.py --model "$HOST" --traces "$R/traces.jsonl" --llr-stats "$R/llr_stats.json" \
  --out-dir "$OUT" --with-greedy --max-hours 6.5 || exit 1
echo "# finished $(date -u +%FT%TZ)"
