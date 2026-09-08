#!/bin/bash
#SBATCH --job-name=nla_heads
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_heads.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_heads.out
# H-W31 — which downstream heads / MLPs carry the NLA-transported clean state (C3pure) from the
# L32 span positions to the reply tokens, against the raw clean state (P_patch) as comparator.
# Pre-registered 2026-09-07 (head-mediation-prereg); frozen rules in nla/configs/nla_heads.yaml.
#
# Three stages. `vectors` holds subject + AV + AR (~72 GB) — the reason this is h200-only; the
# `sweep` and `joint` stages hold the subject alone and never touch AV/AR. Activation patching at
# every o_proj input / MLP output of layers 33..47 between an unsteered and a steered forward of
# the same teacher-forced sequence: suf_c (c<-S into U), nec_c (c<-U into S), SELF_c and ALL
# identities, and --read-knockout (head blind to the span keys, against a mask-only reference).
#
# EXIT 3 = an identity (SELF / ALL / ko_gap) failed; results are NOT reportable. The smoke runs the
# full path on 3 items with the single-component list restricted to layers 45-47 (hooks and ALL
# still span 33-47) and must pass before the full 60-item run starts. ~98k forwards in the full
# run; the smoke prints ms/forward so the ETA can be read against --max-hours.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

HOST=gemma12b
OUT="$PROJ/data/nla/p0/heads/$HOST"; mkdir -p "$OUT/smoke"

echo "# Experiment W31 head localisation · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_heads.py nla/src/head_patch.py nla/src/steer.py nla/src/arm_guard.py \
  nla/src/repair_pairs.py nla/src/nla_writeback.py nla/src/local_av.py nla/src/extract.py \
  nla/configs/nla_heads.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

echo; echo "=== unit: hooks, masks, selection, arm guard ==="
# Explicit file lists, not a keyword filter (job 380178 died on a broadened -k).
python -m pytest nla/tests/test_head_patch.py nla/tests/test_arm_guard.py -q || exit 1
python -m pytest nla/tests/test_steer.py -q -k "replacer or no_grad" || exit 1

echo; echo "=== SMOKE (3 items; singles on layers 45-47, hooks/ALL 33-47, read knockout on) ==="
python nla/src/nla_heads.py --model "$HOST" --stage vectors --smoke --out-dir "$OUT/smoke" \
  --max-hours 1 || exit 1
python nla/src/nla_heads.py --model "$HOST" --stage sweep --smoke --read-knockout \
  --out-dir "$OUT/smoke" --max-hours 1 || exit 1
python nla/src/nla_heads.py --model "$HOST" --stage joint --smoke --out-dir "$OUT/smoke" \
  --max-hours 1 || exit 1

echo; echo "=== FULL (60 items; singles 33-47, read knockout on) ==="
python nla/src/nla_heads.py --model "$HOST" --stage vectors --out-dir "$OUT" --max-hours 2 || exit 1
python nla/src/nla_heads.py --model "$HOST" --stage sweep --read-knockout --out-dir "$OUT" \
  --max-hours 7 || exit 1
python nla/src/nla_heads.py --model "$HOST" --stage joint --out-dir "$OUT" --max-hours 1.5 || exit 1
echo "# done $(date -u +%FT%TZ)"
