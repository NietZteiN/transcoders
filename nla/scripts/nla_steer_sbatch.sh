#!/bin/bash
#SBATCH --job-name=nla_steer
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_steer.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_steer.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
HOST=gemma12b
OUT="$PROJ/data/nla/p0/nla_steer/$HOST"; mkdir -p "$OUT"
echo "# Experiment N — NLA steering on a permitted host · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/steer_run.py nla/src/steer_vectors.py
# --deterministic stays OFF (2026-08-29: changes answers, forks the corpus).

# Primary: the full battery at the pre-registered alpha. The gate is V1 > V3, paired, inherited
# verbatim from B4 so the two hosts are comparable.
echo; echo "=== BATTERY at alpha 1.0 (primary) ==="
python nla/src/steer_run.py --model "$HOST" --with-v2 \
  --only-conditions V1_gloss,V2_wordedit,V3_taskvec,V4_oracle,R_random,F_foreign,A_antipodal \
  --frozen-alpha 1.0 --out-dir "$OUT" --save-replies --max-hours 5 || exit 1

# Secondary: V1 and V3 only, at the ends of the banked grid. Kept small deliberately -- the
# family is BH-corrected and a wide sweep buys little, since a 16x alpha range delivers only
# ~1.6x perturbation (2026-09-03_alpha-sweep-saturates.md).
echo; echo "=== SWEEP: V1, V3 at alpha 0.25 and 4.0 (secondary) ==="
python nla/src/steer_run.py --model "$HOST" --only-conditions V1_gloss,V3_taskvec \
  --alphas 0.25,4.0 --out-dir "$OUT" --save-replies --max-hours 3 || exit 1

# steer_stats.py's THREE path defaults all point at the banked B4 run; calling it without
# explicit paths would score the wrong experiment, and --out alone would OVERWRITE the banked
# n12/steer_stats.json. All three are passed explicitly, always.
echo; echo "=== GATE: V1 vs V3, paired ==="
python nla/src/steer_stats.py \
  --results "$OUT/steer_results.jsonl" \
  --baseline "$OUT/baseline.jsonl" \
  --out "$OUT/steer_stats.json" \
  --primary-alpha 1.0
echo "# done $(date -u +%FT%TZ)"
