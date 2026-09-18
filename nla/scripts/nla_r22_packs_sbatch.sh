#!/bin/bash
#SBATCH --job-name=r22_packs
#SBATCH --partition=normal,h200,h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r22_packs.out
# H-R22a -- rebuild execution-validated T/F case packs on the DERANGED corpus with the artifact's own
# build_case_pack (javac + java per case), then apply the PACKS-PAIRED gate against the originals.
# CPU ONLY. Pre-registered: log/nla-harness/2026-09-18_adversarial-rename-prereg.md
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
echo "# H-R22a packs · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_rename_swap.py nla/src/ase_casepacks.py
java -version 2>&1 | head -1
python nla/src/ase_casepacks.py --lang-dir $A/swap_humaneval_java \
  --cache-dir $A/cache_cases_swap --out $A/packs_swap_humaneval_java.jsonl --max-hours 5.0; rc=$?
echo "=== PACKS-PAIRED gate: the deranged pack must reproduce the original's case count AND labels ==="
python - <<'PY'
import json, collections
from pathlib import Path
A=Path("/scratch/juno/jvl210002/ase2026")
want=set(json.loads((A/"full/snippets_148.json").read_text()))
orig={json.loads(l)["snippet"]: json.loads(l) for l in open(A/"full/packs_orig_full.jsonl")}
new={}
for l in open(A/"packs_swap_humaneval_java.jsonl"):
    r=json.loads(l)
    if "pack" in r: new[r["snippet"]]=r
ok, bad = [], collections.Counter()
for s in sorted(want):
    if s not in new: bad["no pack built"] += 1; continue
    a,b = new[s], orig[s]
    if a["n_cases"] != b["n_cases"]: bad["case count differs"] += 1; continue
    la=[(c["case_id"], c["expected_bool"]) for c in a["pack"]["cases"]]
    lb=[(c["case_id"], c["expected_bool"]) for c in b["pack"]["cases"]]
    if la != lb: bad["label sequence differs"] += 1; continue
    ok.append(s)
print(f"[GATE] PACKS-PAIRED survivors: {len(ok)} of {len(want)}")
print(f"[GATE] attrition: {dict(bad)}")
verdict = "CORPUS-OK" if len(ok) >= 100 else "CORPUS-TOO-SMALL"
print(f"[GATE] H-R22a would read: {verdict} (bar: >= 100)")
(A/"full/snippets_swap.json").write_text(json.dumps(sorted(ok)))
n=sum(new[s]["n_cases"] for s in ok)
print(f"[GATE] frozen swap set: {len(ok)} snippets / {n} cases -> full/snippets_swap.json")
# the paired pack files both conditions will be scored on
for name, src in (("packs_swap_paired.jsonl", new), ("packs_orig_swapsubset.jsonl", orig)):
    with open(A/"full"/name, "w") as fh:
        for s in ok: fh.write(json.dumps(src[s])+"\n")
print("[GATE] wrote full/packs_swap_paired.jsonl and full/packs_orig_swapsubset.jsonl")
PY
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
