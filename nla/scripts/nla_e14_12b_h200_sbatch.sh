#!/bin/bash
#SBATCH --job-name=nla_e14_12b
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_e14_12b_h200.out
# H-E14, 12B L32 site ONLY, re-run PINNED TO h200 nvl.
#
# WHY THE PIN: job 393530 ran this site on an h100 80GB (g-04-02) and its identity gate REFUSED -- every
# banked-byte arm was off by ~1 nat (edit 1.069, foreign 1.002, swap 1.139) and, decisively, `self` came back
# at 1.0017 when it must be 0.000. `self` writes the state the model already has, and it is computed against an
# unsteered baseline *inside the same run*, so a ~1-nat deviation is not a stale-bytes problem: on this hardware
# the hook path itself perturbs the bf16 numerics of a 12B forward enough to move a ~700-token summed logp by
# a nat. The 12B banked rows (job 393219) were produced on h200 nvl (g-07-03) and the 4B sites reproduce fine
# on either, so the 0.05-nat identity tolerance is NOT portable across GPU architectures at 12B scale.
# This is why CLAUDE.md §4 requires the GPU id in provenance: for 12B, the banked rows are only reproducible
# on the architecture that produced them. Nothing from 393530's 12B site is quoted.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
# /scratch hit 100 % (18 MB free) on 2026-09-13 and job 393618 died writing a ~100 KB rows file.
# This output goes to /work (99 TB free) under nla/data/, which is gitignored.
OUT=/work/jvl210002/migration/transcoders/nla/data/e14_loonorm
REL=/scratch/juno/jvl210002/nla_ml_gemma12b_released
OURS=/scratch/juno/jvl210002/nla_ml_gemma12b
CFG12=nla/configs/nla_ml_gate_12b.yaml
TRACES12=$PROJ/data/nla/p0/trace_llr/gemma12b/traces.jsonl
echo "# H-E14 12B L32 on h200 · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_erasure.py nla/scripts/nla_e14_12b_h200_sbatch.sh
nvidia-smi --query-gpu=index,name --format=csv,noheader
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
  *H200*) echo "host arch OK (H200)" ;;
  *) echo "REFUSED: this site must run on H200 nvl to reproduce the banked rows; got $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; exit 2 ;;
esac
python nla/src/nla_erasure.py --config "$CFG12" --root "$REL" --traces "$TRACES12" --layer 32 \
  --compare-root "$OURS" --out "$OUT/gemma12b_L32_h200" --max-hours 1.0; rc=$?
echo "--- 12B L32 (h200) rc=$rc"
python - "$OUT/gemma12b_L32_h200/erasure_stats.json" <<'PY'
import json, sys
st = json.loads(open(sys.argv[1]).read())
print("identity:", json.dumps(st["identity"]["worst_abs_diff"]), "passes", st["identity"]["passes"])
if "rules" in st:
    for line in st["summary"]:
        print("  ", line)
PY
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
