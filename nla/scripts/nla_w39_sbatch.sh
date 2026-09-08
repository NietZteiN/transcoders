#!/bin/bash
#SBATCH --job-name=nla_w39
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_w39.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_w39.out
# H-W39 — is route similarity smooth in content similarity, or does it snap? Pre-registered
# 2026-09-08 (content-dose-prereg). Interpolates each span's own clean state toward THE SAME foreign
# vector N_foreign uses, at alpha 0.25/0.50/0.75; alpha 0 (P_patch) and alpha 1 (N_foreign) are
# already banked, so only the three middles are run.
#
# PRIMARY MEASURE IS NECESSITY, not sufficiency, chosen from banked data before these arms existed:
# across the five banked arms `nec` spreads the C3pure-vs-foreign contrast over 0.77 -> 0.40 -> -0.16
# where `suf` compresses it into 0.85 -> 0.74. Freezing on `suf` would repeat H-W36a's mistake of
# ruling on a measure with no dynamic range.
#
# The write is norm-matched, so alpha moves direction only, and a linear blend of near-orthogonal
# vectors is NOT uniform in angle -- so the achieved cos(v, own) is recorded per row and is the
# x-axis of every curve. Built from the banked vectors.npz: no AV, no AR, subject model only.
#
# EXIT 3 = an identity (SELF / ALL / ko_gap) failed; results are not reportable.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
SRC="$PROJ/data/nla/p0/heads/$HOST"                 # banked vectors + H-W31 rows
OUT="$PROJ/data/nla/p0/w39/$HOST"; mkdir -p "$OUT/smoke"
ARMS="D_25,D_50,D_75"

echo "# Experiment W39 content dose · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_heads.py nla/src/head_patch.py nla/src/steer.py nla/src/arm_guard.py \
  nla/configs/nla_heads.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

# The banked vectors ARE the input; refuse rather than silently recomputing them.
for f in vectors.npz spans.jsonl heads_rows.jsonl; do
  [ -s "$SRC/$f" ] || { echo "REFUSED: missing prerequisite $SRC/$f"; exit 2; }
done
cp "$SRC/vectors.npz" "$SRC/spans.jsonl" "$OUT/"
cp "$SRC/vectors.npz" "$SRC/spans.jsonl" "$OUT/smoke/"

echo; echo "=== unit: arm construction, hooks, guard ==="
python -m pytest nla/tests/test_null_arms.py nla/tests/test_head_patch.py \
  nla/tests/test_arm_guard.py -q || exit 1
python -m pytest nla/tests/test_steer.py -q -k "replacer or no_grad" || exit 1

echo; echo "=== SMOKE (3 items, singles 45-47, three null arms) ==="
python nla/src/nla_heads.py --model "$HOST" --stage sweep --smoke --arms "$ARMS" \
  --out-dir "$OUT/smoke" --max-hours 1 || exit 1

echo; echo "=== FULL (60 items, singles 33-47, three null arms) ==="
python nla/src/nla_heads.py --model "$HOST" --stage sweep --arms "$ARMS" \
  --out-dir "$OUT" --max-hours 8 || exit 1
python nla/src/nla_heads.py --model "$HOST" --stage joint --arms "$ARMS" \
  --out-dir "$OUT" --max-hours 2 || exit 1
echo "# done $(date -u +%FT%TZ)"
