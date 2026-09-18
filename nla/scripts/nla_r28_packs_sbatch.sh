#!/bin/bash
#SBATCH --job-name=r28_packs
#SBATCH --partition=normal,h200,h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r28_packs.out
# H-R28a -- rebuild execution-validated case packs on the FLATTENED corpus, then PACKS-PAIRED.
# Pre-registered: log/nla-harness/2026-09-18_flatten-prereg.md
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
echo "# H-R28a packs · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_flatten_java.py
python nla/src/ase_casepacks.py --lang-dir $A/flat_humaneval_java \
  --cache-dir $A/cache_cases_flat --out $A/packs_flat_humaneval_java.jsonl --max-hours 5.0; rc=$?
python - <<'PY'
import json, collections
from pathlib import Path
A=Path("/scratch/juno/jvl210002/ase2026")
want=set(json.loads((A/"full/snippets_swap.json").read_text()))
orig={json.loads(l)["snippet"]: json.loads(l) for l in open(A/"full/packs_orig_swapsubset.jsonl")}
new={}
for l in open(A/"packs_flat_humaneval_java.jsonl"):
    r=json.loads(l)
    if "pack" in r: new[r["snippet"]]=r
ok, bad = [], collections.Counter()
for s in sorted(want):
    if s not in new: bad["no pack (skipped or compile/exec failure)"]+=1; continue
    a,b=new[s],orig[s]
    if a["n_cases"]!=b["n_cases"]: bad["case count differs"]+=1; continue
    la=[(c["case_id"],c["expected_bool"]) for c in a["pack"]["cases"]]
    lb=[(c["case_id"],c["expected_bool"]) for c in b["pack"]["cases"]]
    if la!=lb: bad["label sequence differs"]+=1; continue
    ok.append(s)
print(f"[GATE] PACKS-PAIRED survivors: {len(ok)} of {len(want)}")
print(f"[GATE] attrition: {dict(bad)}")
print(f"[GATE] H-R28a: {'CORPUS-OK' if len(ok)>=80 else 'CORPUS-TOO-SMALL'} (bar >= 80)")
(A/"full/snippets_flat.json").write_text(json.dumps(sorted(ok)))
with open(A/"full/packs_flat_paired.jsonl","w") as fh:
    for s in ok: fh.write(json.dumps(new[s])+"\n")
print(f"[GATE] {len(ok)} snippets / {sum(new[s]['n_cases'] for s in ok)} cases -> full/packs_flat_paired.jsonl")
PY
echo "# done rc=$rc $(date -u +%FT%TZ)"
