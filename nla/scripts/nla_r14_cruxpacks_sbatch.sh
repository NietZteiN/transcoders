#!/bin/bash
#SBATCH --job-name=r14_cruxpacks
#SBATCH --partition=normal,h200,h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r14_cruxpacks.out
# H-R14 stage 2 (exploratory) -- build execution-validated T/F case packs for the CruxEval-X Java split
# (698 snippets) with the artifact's own build_case_pack. CPU ONLY (javac + java per case, no GPU).
# UNTESTED PATH: our pack builder has only ever run on HumanEval-X. `_extract_cruxeval_seeds` and the
# `cruxeval` -> counterfactual_tf profile switch exist in evaluation/java_counterfactual.py, but the YIELD
# is unknown -- the paper reports only 1.97 cases/snippet here vs 11.72 on HumanEval. A low yield is a
# RESULT (it bounds what CruxEval can ever answer), not a failure to work around.
# Pre-registered: log/nla-harness/2026-09-16_full-corpus-prereg.md
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
echo "# H-R14 CruxEval-X case packs · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_casepacks.py
java -version 2>&1 | head -1
python nla/src/ase_casepacks.py --dataset cruxeval --lang-dir Source/Cruxeval/java \
  --cache-dir $A/cache_cases_crux --out $A/packs_cruxeval_java.jsonl --max-hours 7.0; rc=$?
if [ -s "$A/packs_cruxeval_java.jsonl" ]; then
  python - <<'PY'
import json
rows=[json.loads(l) for l in open("/scratch/juno/jvl210002/ase2026/packs_cruxeval_java.jsonl") if l.strip()]
keep=[r for r in rows if "pack" in r]
n=sum(r["n_cases"] for r in keep)
print(f"[CRUX] packs built: {len(keep)} snippets with packs of {len(rows)} attempted · {n} cases · "
      f"{n/max(len(keep),1):.2f} cases/snippet (paper: 698 snippets / 1378 cases / 1.97 per snippet)")
PY
fi
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
