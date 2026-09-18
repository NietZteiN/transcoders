#!/bin/bash
#SBATCH --job-name=nla_dose
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=10:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_dose.out
# H-C4/H-C5/H-C6 — is the NLA fidelity gap closable with DATA? Pre-registered in
# log/nla-harness/2026-09-12_dose-response-prereg.md. Rules live in nla/src/dose_score.py.
#
# WHY DOWNWARD. H-C1 showed our recipe reaches causal fidelity 0.739 at 12B L32 vs the released
# pair's 0.98, so the gap is our recipe. Going UP in data means regenerating the explanation corpus
# (banked explain stage: 10 shards x ~3 h = ~30 GPU-h), so a 2x run is ~35 GPU-h -- NOT the "~5" the
# H-C1 entry estimated, which counted only training. This runs 25/50/100 % of the EXISTING rows at
# fixed layer L7 and decides whether that 35 GPU-h is worth spending.
#
# Each dose point gets its OWN root so no banked pair is ever overwritten; corpus/explain/acts are
# symlinked in, so the ~70 GB of activations is extracted ONCE and shared.
#
# EXIT 3 = the pre-flight failed: the re-extracted acts did not reproduce the banked L7 norms
# (injection_scale 5100, mean_av_train 5015.6255 +/- 0.5), so the dose points would not be
# comparable with the banked 100 % pair. No dose numbers are reported in that case.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1

T=nla/src/nla_train.py
GATE=nla/src/nla_ml_gate.py
GCFG=nla/configs/nla_ml_gate_dose.yaml
TRACES=$PROJ/data/nla/p0/trace_llr/gemma4b/traces.jsonl
REAL=$PROJ/data/nla/ml/gemma4b                 # the banked root (read-only here)
BASE=/scratch/juno/jvl210002/nla_ml_dose
K=7

echo "# H-C4 dose-response · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $T $GATE nla/src/dose_score.py $GCFG nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus). That is what
# H-C5, the 100 % re-train control, exists to measure.

echo; echo "=== unit: the frozen dose table + the --train-frac contract ==="
python -m pytest nla/tests/test_dose_score.py -q || exit 1
python -m pytest nla/tests/test_beta_sweep.py -q -k "subsample" || exit 1

echo; echo "=== stage 0: shared base root (corpus + explain symlinked from the banked root) ==="
mkdir -p "$BASE"
for d in corpus explain; do
  [ -e "$BASE/$d" ] || ln -s "$(readlink -f $REAL/$d)" "$BASE/$d"
done
ls -la "$BASE"

echo; echo "=== stage 1: re-extract activations (~70 GB, all 34 layers, ONE pass) ==="
# acts/L{K}.npy were deleted in the 2026-09-09 quota incident; only norms.json survived, which is
# what makes the pre-flight comparison possible.
if [ -f "$BASE/acts/norms.json" ]; then
  echo "[dose] acts already present, skipping extract"
else
  python $T --stage extract --root "$BASE" || exit 1
fi

echo; echo "=== PRE-FLIGHT: do the re-extracted acts reproduce the banked L7 norms? ==="
python - <<'PY' || exit 3
import json, sys
sys.path.insert(0, "nla/src")
from dose_score import check_extract
r = check_extract("/scratch/juno/jvl210002/nla_ml_dose/acts/norms.json")
print("[dose] pre-flight:", json.dumps(r))
sys.exit(0 if r["passes"] else 3)
PY

for F in 0.25 0.50 1.00; do
  TAG=$(echo "$F" | tr -d '.')
  R="$BASE/f$TAG"
  echo; echo "=== dose $F  (root $R) ==="
  mkdir -p "$R"
  for d in corpus explain acts; do
    [ -e "$R/$d" ] || ln -s "$BASE/$d" "$R/$d"
  done
  python $T --stage sft_av --layer $K --train-frac "$F" --root "$R" || exit 1
  python $T --stage sft_ar --layer $K --train-frac "$F" --root "$R" || exit 1
  python $T --stage check  --layer $K --root "$R"                   || exit 1
  # --root is MANDATORY on the gate: without it the gate reads the BANKED pair and would silently
  # score the 100 % run three times.
  # stage_vectors has no skip of its own (unlike the trainer's stages), so guard it: the vector stage
  # is ~35 min and job 391398 already produced f025's.
  if [ -f "$R/gate/vectors/L$K.npz" ]; then
    echo "[dose] vectors for L$K already present under $R, skipping"
  else
    python $GATE --stage vectors --layer $K --config "$GCFG" --root "$R" --traces "$TRACES" --max-hours 2 || exit 1
  fi
  # --ignore-liveness: DECLARED DEVIATION, log/nla-harness/2026-09-12_dose-liveness-deviation.md, filed
  # before any S_c3 was seen. A deliberately under-trained dose point can fall below the liveness floor
  # (25 % gave AR fve 0.1407 < 0.20 against the banked 0.3984) and the gate then refuses to score it,
  # which is right for an INSTRUMENT check and wrong for a dose-response, where an under-trained pair's
  # S_c3 is the measurement. Applied to all three points so they stay apples-to-apples. The resulting
  # gate_stats.json carries `reportable: false` BY DESIGN; H-C4's verdict comes from dose_score.py.
  python $GATE --stage score --config "$GCFG" --root "$R" --traces "$TRACES" --ignore-liveness --max-hours 1 || exit 1
done

echo; echo "=== score the dose response (frozen H-C4/H-C5/H-C6 rules) ==="
OUT=$PROJ/data/nla/ml/gemma4b/gate/dose
mkdir -p "$OUT"
python nla/src/dose_score.py \
  --rows-25  "$BASE/f025/gate/gate_rows.jsonl" \
  --rows-50  "$BASE/f050/gate/gate_rows.jsonl" \
  --rows-100 "$BASE/f100/gate/gate_rows.jsonl" \
  --rows-banked "$REAL/gate/gate_rows.jsonl" \
  --norms "$BASE/acts/norms.json" \
  --checks "$BASE/f025/L$K/check.json" "$BASE/f050/L$K/check.json" "$BASE/f100/L$K/check.json" \
  --out "$OUT/dose_stats.json"; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
