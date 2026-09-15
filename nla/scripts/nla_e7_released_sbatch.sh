#!/bin/bash
#SBATCH --job-name=nla_e7_rel
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_e7_released.out
# H-E7 / H-C11 -- the RELEASED kitft/nla-gemma3-12b-L32-{av,ar} pair on the repaired anchoring (the 60
# items / 471 spans every run in this family uses), and the training-free erasure vector at 12B L32
# against it. Pre-registered in log/nla-harness/2026-09-13_e7-released-erasure-prereg.md.
#
# One job, three stages on ONE H200 nvl (host 23.5 + released AV ~24 + released AR ~24 GB bf16):
#   1. nla_ml_gate.py --stage vectors --layer 32  on a root whose L32/{av,ar} are symlinks to the HF
#      snapshots (both carry nla_meta.yaml; LocalAV/NLACritic loaded exactly these in the W work).
#   2. nla_ml_gate.py --stage score --ignore-liveness  (the released pair has no eval.json/check.json;
#      liveness is ignored BY CONSTRUCTION and the gate's `reportable` is False -- H-C11 is read from the
#      rows by nla_erasure.py, not from gate_stats.json).
#   3. nla_erasure.py --layer 32 --compare-root <our 12B root>: erase_{0.5,1,2}, erase_flip (the sign-flip
#      directional null; Gaussian random scores -359 at this layer), erase_rand, erase_own, released
#      edit/foreign/swap/self, and our pair's edit_ours/foreign_ours -- under gates G1 (anchoring
#      identity) and G2 (rows identity). Any gate failure -> exit 3, E-HARNESS-FAULT, nothing reportable.
# The 3-item smoke runs the same three stages into a separate root first; the full run does not start
# unless the smoke's gates pass.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
CFG=nla/configs/nla_ml_gate_12b.yaml
TRACES=$PROJ/data/nla/p0/trace_llr/gemma12b/traces.jsonl
OURS=/scratch/juno/jvl210002/nla_ml_gemma12b
REL=/scratch/juno/jvl210002/nla_ml_gemma12b_released
SMK=/scratch/juno/jvl210002/nla_ml_gemma12b_released_smoke
AV_SNAP=$(ls -d "$HF_HOME"/hub/models--kitft--nla-gemma3-12b-L32-av/snapshots/*/ | head -1)
AR_SNAP=$(ls -d "$HF_HOME"/hub/models--kitft--nla-gemma3-12b-L32-ar/snapshots/*/ | head -1)
echo "# H-E7/H-C11 released pair + erasure · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_erasure.py nla/src/nla_ml_gate.py nla/src/steer.py nla/src/nla_cycle.py "$CFG" nla/configs/nla_ml_12b.yaml
echo "AV snapshot $AV_SNAP"; echo "AR snapshot $AR_SNAP"
wc -l "$TRACES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_erasure.py nla/tests/test_steer.py -q -p no:faulthandler \
  -k "erasure or erase or loo or rules or identity or vector or replacer or no_grad or flip or anchoring" || exit 1

for R in "$SMK" "$REL"; do
  mkdir -p "$R/L32"
  [ -e "$R/L32/av" ] || ln -s "${AV_SNAP%/}" "$R/L32/av"
  [ -e "$R/L32/ar" ] || ln -s "${AR_SNAP%/}" "$R/L32/ar"
  ls -l "$R/L32"
done

echo; echo "=== SMOKE: 3 items through vectors -> score -> erasure (G1 subset-tolerant; G2 must pass) ==="
python nla/src/nla_ml_gate.py --stage vectors --layer 32 --config "$CFG" --root "$SMK" --traces "$TRACES" --smoke --max-hours 1 || exit 1
python nla/src/nla_ml_gate.py --stage score --config "$CFG" --root "$SMK" --traces "$TRACES" --smoke --ignore-liveness --max-hours 1 || exit 1
python nla/src/nla_erasure.py --config "$CFG" --root "$SMK" --traces "$TRACES" --smoke --layer 32 --compare-root "$OURS" \
  --out "$SMK/gate/erasure" --max-hours 1 || exit 1

echo; echo "=== FULL 1/3: released vectors L32 (host + released AV + AR) ==="
python nla/src/nla_ml_gate.py --stage vectors --layer 32 --config "$CFG" --root "$REL" --traces "$TRACES" --max-hours 2.5 || exit 1
echo; echo "=== FULL 2/3: released score L32 (--ignore-liveness; reportable=False by construction) ==="
python nla/src/nla_ml_gate.py --stage score --config "$CFG" --root "$REL" --traces "$TRACES" --ignore-liveness --max-hours 1
echo "(score rc=$? -- non-zero is expected: liveness ignored)"
echo; echo "=== FULL 3/3: erasure at L32 vs released edit + our edit_ours (gates G1/G2) ==="
python nla/src/nla_erasure.py --config "$CFG" --root "$REL" --traces "$TRACES" --layer 32 --compare-root "$OURS" --max-hours 1.5; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
