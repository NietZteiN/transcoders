#!/bin/bash
#SBATCH --job-name=nla_w35
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_w35.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_w35.out
# H-W35 — is the eight-component path H-W31 found SPECIFIC to the clean state, or is it simply
# where anything written at identifier spans arrives? Pre-registered 2026-09-08 (specificity-prereg).
#
# Writes DIFFERENT content at the SAME 1,459 positions and compares the per-component sufficiency
# profiles against C3pure with H-W31c's machinery (Spearman + Jaccard of the top-16):
#   N_sibling  another span of the same item   (right item, wrong span; banked 87.4% of P_patch)
#   N_foreign  another item's clean span       (content, wrong item; 33.2%)
#   N_random   a random unit direction         (no content; banked -365.59, descriptive only)
#
# All three are built from the BANKED vectors.npz of job 382366 -- no AV, no AR, subject model only
# (~24 GB), which is why h100 is eligible alongside h200. C3pure/P_patch are NOT re-run; H-W31's
# rows are reused, so the identity gate here doubles as a cross-job reproducibility check.
#
# EXIT 3 = an identity (SELF / ALL / ko_gap) failed; results are not reportable.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
SRC="$PROJ/data/nla/p0/heads/$HOST"                 # banked vectors + H-W31 rows
OUT="$PROJ/data/nla/p0/w35/$HOST"; mkdir -p "$OUT/smoke"
ARMS="N_sibling,N_foreign,N_random"

echo "# Experiment W35 specificity · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
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
