#!/bin/bash
#SBATCH --job-name=kvbyp
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_kvbyp.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_kvbyp.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

HOST=gemma12b
OUT="$PROJ/data/nla/p0/steerv2/$HOST/run"
SMOKE="$PROJ/data/nla/p0/steerv2/$HOST/smoke"
mkdir -p "$OUT" "$SMOKE"
echo "# KV-bypass run · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/steer_run.py nla/src/steer_multilayer.py nla/src/multilayer_vectors.py

# The matched alpha comes from the frozen rule, read out of the calibration artifact. If the
# calibration did not bracket the single-layer displacement, matched_alpha.py exits non-zero and
# the whole job stops here rather than running an uncontrolled comparison.
MA=$(python nla/src/matched_alpha.py "$PROJ/data/nla/p0/steerv2/$HOST/energy_match.json") || { echo "FATAL: energy match unusable"; exit 1; }
echo "# matched multi-layer alpha (frozen rule): $MA"

# ── smoke: 3 items, one condition, both modes. Proves the multilayer path generates at all
#    before committing ~1.5 GPU-h to it. Project rule: micro-scale first.
echo; echo "=== SMOKE ==="
python nla/src/steer_run.py --model "$HOST" --multilayer --limit 3 --only-conditions V3_taskvec \
  --frozen-alpha "$MA" --out-dir "$SMOKE" --deterministic --save-replies || exit 1
grep -c . "$SMOKE/steer_results.jsonl" || true
python - "$SMOKE/steer_results.jsonl" <<'EOF' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert rows, "smoke produced no rows"
assert all(r.get("multilayer") for r in rows), "rows not tagged multilayer"
exp = rows[0]["layer"] + 1
assert all(r["n_layers_written"] == exp for r in rows), f"wrong layer count (want {exp})"
par = sum(r["parsed"] for r in rows) / len(rows)
print(f"[smoke] {len(rows)} rows · parse rate {par:.2f}")
# P0.2's failure mode was a parse-rate collapse to 0.100 under oversized perturbation. If the
# matched alpha reproduces that, the energy match failed and the main run is not worth starting.
assert par >= 0.5, f"parse rate {par:.2f} — energy match did not hold, stopping"
EOF

# ── arm 1: single-layer, the banked instrument reproduced IN THIS JOB.
#    Re-run rather than compared against banked numbers: greedy bf16 does not reproduce per item
#    across runs (0.85-0.90 agreement), so a cross-run comparison would confound the intervention
#    with the reproducibility floor.
echo; echo "=== SINGLE-LAYER (V3, V4) ==="
python nla/src/steer_run.py --model "$HOST" --only-conditions V3_taskvec,V4_oracle \
  --frozen-alpha 1.0 --out-dir "$OUT" --deterministic --save-replies --max-hours 3 || exit 1

# ── arm 2: multilayer at the matched alpha (primary) AND at alpha=1.0 (the "loud" arm that
#    shows what uncontrolled coverage does). R_random is the control the P0.2 rule requires.
echo; echo "=== MULTILAYER (V3, V4, R_random) at alpha in {$MA, 1.0} ==="
python nla/src/steer_run.py --model "$HOST" --multilayer --only-conditions V3_taskvec,V4_oracle,R_random \
  --alphas "$MA,1.0" --out-dir "$OUT" --deterministic --save-replies --max-hours 4 || exit 1

# The pre-registered rules are executed arithmetically, never typed by hand: kv_bypass_stats.py
# is their only executor, so the verdict cannot drift from the pre-registration text. Its exit
# code is non-zero only when an ARM IS MISSING (verdict INCOMPLETE) — a real null must not look
# like a crash, and a missing denominator must not look like a null.
# ── arm 3: the exact ceiling. Alpha does not apply to a state replacement, so it runs ONCE
#    rather than once per alpha — the same rows at two alphas would be duplicate work reported
#    as two conditions.
echo; echo "=== MULTILAYER V5_replace (exact state replacement, ceiling) ==="
python nla/src/steer_run.py --model "$HOST" --multilayer --only-conditions V5_replace \
  --frozen-alpha 1.0 --out-dir "$OUT" --deterministic --save-replies --max-hours 2 || exit 1

echo; echo "=== VERDICT (frozen rules, executed) ==="
python nla/src/kv_bypass_stats.py --host "$HOST" \
  --results "$OUT/steer_results.jsonl" --baseline "$OUT/baseline.jsonl" \
  --energy-match "$PROJ/data/nla/p0/steerv2/$HOST/energy_match.json" \
  --out "$PROJ/data/nla/p0/steerv2/$HOST/kv_bypass_stats.json"

echo; echo "=== rows by arm ==="
python - "$OUT/steer_results.jsonl" <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
agg = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    k = ("multi" if r.get("multilayer") else "single", r["condition"], r["alpha"])
    agg[k][0] += 1; agg[k][1] += bool(r["correct"]); agg[k][2] += bool(r["parsed"])
print(f"{'mode':<7}{'condition':<14}{'alpha':>8}{'n':>5}{'acc':>8}{'parse':>8}")
for k in sorted(agg):
    n, c, p = agg[k]
    print(f"{k[0]:<7}{k[1]:<14}{k[2]:>8}{n:>5}{c/n:>8.3f}{p/n:>8.3f}")
EOF
echo "# done $(date -u +%FT%TZ)"
