#!/bin/bash
#SBATCH --job-name=cov_run
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_cov_run.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_cov_run.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
HOST=gemma12b
CAL="$PROJ/data/nla/p0/steerv2/$HOST/coverage_match.json"
OUT="$PROJ/data/nla/p0/coverage/$HOST"; mkdir -p "$OUT"
CONDS=V1_gloss,V3_taskvec,R_random
echo "# Experiment C — token coverage · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/steer_run.py nla/src/span_positions.py nla/src/coverage_alpha.py

# Reference arm: one token, the site every banked result used.
echo; echo "=== last_prompt (1 token — the banked site) alpha 1.0 ==="
python nla/src/steer_run.py --model "$HOST" --with-v2 --only-conditions "$CONDS" \
  --positions last_prompt --frozen-alpha 1.0 --out-dir "$OUT" --save-replies --max-hours 3 || exit 1

# Arm 1, primary convention: equal DELIVERED perturbation at the answer position. May be
# unreachable (exit 2) — 45 identifier tokens reach that position only through attention.
set +e
A_DEL=$(python nla/src/coverage_alpha.py --file "$CAL" --coverage id_spans --convention delivered)
RC=$?
set -e
if [ $RC -eq 0 ]; then
  echo; echo "=== id_spans, DELIVERED-matched, alpha $A_DEL ==="
  python nla/src/steer_run.py --model "$HOST" --with-v2 --only-conditions "$CONDS" \
    --positions id_spans --frozen-alpha "$A_DEL" --out-dir "$OUT" --save-replies --max-hours 3 || exit 1
else
  echo; echo "=== id_spans DELIVERED-matched: UNREACHABLE (exit $RC) — reported, not silently skipped ==="
fi

# Arm 2, secondary convention: equal TOTAL injected energy. Always available — arithmetic.
A_EN=$(python nla/src/coverage_alpha.py --file "$CAL" --coverage id_spans --convention energy) || exit 1
echo; echo "=== id_spans, ENERGY-matched, alpha $A_EN ==="
python nla/src/steer_run.py --model "$HOST" --with-v2 --only-conditions "$CONDS" \
  --positions id_spans --frozen-alpha "$A_EN" --out-dir "$OUT" --save-replies --max-hours 3 || exit 1

echo; echo "=== rows by arm ==="
python - "$OUT/steer_results.jsonl" <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
agg = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r.get("skipped"): continue
    k = (r.get("positions"), r["condition"], r.get("alpha"))
    agg[k][0] += 1; agg[k][1] += bool(r.get("correct")); agg[k][2] += bool(r.get("parsed"))
print(f"{'positions':<14}{'condition':<14}{'alpha':>8}{'n':>5}{'acc':>8}{'parse':>8}")
for k in sorted(agg, key=lambda x: (str(x[0]), x[1], x[2] or 0)):
    n, c, p = agg[k]
    print(f"{str(k[0]):<14}{k[1]:<14}{str(k[2]):>8}{n:>5}{c/n:>8.3f}{p/n:>8.3f}")
EOF
echo "# done $(date -u +%FT%TZ)"
