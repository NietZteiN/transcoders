#!/bin/bash
#SBATCH --job-name=trace_llr
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_trace_llr.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_trace_llr.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
HOST=gemma12b
OUT="$PROJ/data/nla/p0/trace_llr/$HOST"; mkdir -p "$OUT" "$OUT/smoke"
echo "# Experiment R — clean-trace LLR re-score · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/trace_llr.py nla/src/steer.py nla/src/steer_multilayer.py nla/src/steer_vectors.py nla/src/steer_run.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ── smoke: 3 items, two arms, the no-op twice, the L0-prompt sanity. Pre-registered gate:
# the clean prompt must prefer its own clean trace (sanity.frac_positive == 1.0 on 3 items)
# and the two no-op replicates must agree to within the floor, or nothing downstream runs.
echo; echo "=== SMOKE ==="
python nla/src/trace_llr.py --model "$HOST" --smoke --out-dir "$OUT/smoke" --max-hours 0.5 || exit 1
python - "$OUT/smoke/llr_stats.json" <<'PY' || exit 1
import json, sys
s = json.load(open(sys.argv[1]))
san = s["sanity"]
print("smoke floor", s["floor"], "sanity", san)
assert san["n"] >= 3, "sanity rows missing"
assert san["frac_positive"] == 1.0, "clean prompt does not prefer its own clean trace -> M is broken"
assert s["floor"] < 0.05, f"bf16 floor {s['floor']} is not below the support threshold 0.05"
for k, v in s["per_arm"].items():
    assert v["n"] >= 3 and v["mean_dM"] is not None, f"arm {k} produced no rows"
print("SMOKE OK")
PY

# ── full: 60 items, every banked Gemma arm.
echo; echo "=== FULL ==="
python nla/src/trace_llr.py --model "$HOST" --out-dir "$OUT" --max-hours 5 || exit 1
echo; echo "=== VERDICT ==="
python - "$OUT/llr_stats.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"floor {s['floor']:.4f}  M(noop) {s['M_noop_mean']:.3f}  sanity {s['sanity']}")
print(f"{'arm':44}{'n':>4}{'dM':>9}{'ci_lo':>9}{'ci_hi':>9}{'dM_flip':>9}")
for k, v in sorted(s["per_arm"].items(), key=lambda kv: -(kv[1]["mean_dM"] or -9)):
    f = v["mean_dM_flippable"]
    print(f"{k:44}{v['n']:>4}{v['mean_dM'] or 0:>9.4f}{v['ci95'][0]:>9.4f}{v['ci95'][1]:>9.4f}{(f if f is not None else float('nan')):>9.4f}")
print("VERDICT", s["verdict"], "supporting", s["supporting_arms"], "random_moves", s["random_moves"])
PY
echo "# finished $(date -u +%FT%TZ)"
