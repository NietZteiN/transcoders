#!/bin/bash
#SBATCH --job-name=nla_r1_gate
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r1_gate.out
# H-R1a / H-R1b -- THE GATE. Does the ASE-2026 paper's identifier-renaming damage reproduce on a model
# we are allowed to run, under THEIR protocol (batched T/F case verification, their prompt, their
# parser, their Pass@k)? Their HumanEval-X figure is 76.49 -> 40.20 on Qwen2.5-7B; mine on Python/JS
# output prediction was +0.0117 [-0.037, +0.061]. Both cannot describe the same phenomenon.
# Pre-registered: log/nla-harness/2026-09-14_ase-replication-prereg.md
# Amended:        log/nla-harness/2026-09-14_ase-prereg-amendment.md  (packs rebuilt per variant)
#   H-R1a PROTOCOL-REPLICATED if original P@1 in [0.60,0.90]; PROTOCOL-BROKEN if < 0.55 (at/below the
#         50% chance floor set by the negation-paired cases => harness fault, not a model result)
#   H-R1b DAMAGE-REPLICATES if the drop >= 0.15 with the paired CI excluding 0; DAMAGE-ABSENT if < 0.05
# The whole 11-arm bake-off (H-R2) is gated on this.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
ORIG=$A/packs_humaneval_java.jsonl
REN=$A/packs_renamed_humaneval_java.jsonl
OUT=$A/gate
N=50
echo "# H-R1 gate · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_tf_run.py nla/src/ase_casepacks.py nla/src/ase_rename_java.py
nvidia-smi --query-gpu=index,name --format=csv,noheader
for f in "$ORIG" "$REN"; do [ -s "$f" ] || { echo "REFUSED: missing packs $f"; exit 2; }; done
wc -l "$ORIG" "$REN"

echo; echo "=== PACKS-PAIRED gate (amendment): same case count AND label sequence per snippet ==="
python - "$ORIG" "$REN" "$OUT/subset.json" "$N" <<'PY' || exit 3
import json, sys
from pathlib import Path
o={json.loads(l)["snippet"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
r={json.loads(l)["snippet"]: json.loads(l) for l in open(sys.argv[2]) if l.strip()}
dest=Path(sys.argv[3]); N=int(sys.argv[4]); dest.parent.mkdir(parents=True, exist_ok=True)
common=sorted(s for s in set(o)&set(r) if "pack" in o[s] and "pack" in r[s])
def labels(rec): return [bool(c["expected_bool"]) for c in rec["pack"].get("cases",[])]
paired=[s for s in common if labels(o[s])==labels(r[s]) and labels(o[s])]
frac=len(paired)/len(common) if common else 0.0
verdict="PACKS-PAIRED" if frac>=0.95 else "PACKS-UNPAIRED"
subset=paired[:N]
print(f"[PAIR] common {len(common)} · corresponding {len(paired)} ({frac:.3f}) -> {verdict}")
print(f"[PAIR] mismatching examples: {[s for s in common if s not in set(paired)][:6]}")
print(f"[PAIR] subset = first {len(subset)} by id: {subset[:5]} ...")
json.dump({"verdict":verdict,"frac":frac,"n_common":len(common),"n_paired":len(paired),
           "subset":subset}, open(dest,"w"), indent=1)
if len(subset) < 40:
    print(f"[PAIR] WARNING only {len(subset)} usable snippets (<40): underpowered, descriptive only")
PY

SUB=$(python -c "import json;print(len(json.load(open('$OUT/subset.json'))['subset']))")
echo "subset size: $SUB"
python - "$OUT/subset.json" "$ORIG" "$OUT/packs_orig_subset.jsonl" <<'PY'
import json,sys
sub=set(json.load(open(sys.argv[1]))["subset"])
with open(sys.argv[3],"w") as f:
    for l in open(sys.argv[2]):
        if l.strip() and json.loads(l)["snippet"] in sub: f.write(l)
PY
python - "$OUT/subset.json" "$REN" "$OUT/packs_ren_subset.jsonl" <<'PY'
import json,sys
sub=set(json.load(open(sys.argv[1]))["subset"])
with open(sys.argv[3],"w") as f:
    for l in open(sys.argv[2]):
        if l.strip() and json.loads(l)["snippet"] in sub: f.write(l)
PY

echo; echo "=== ARM 1/2: original (the ceiling) ==="
python nla/src/ase_tf_run.py --packs "$OUT/packs_orig_subset.jsonl" --condition original \
  --model llama8b --out "$OUT/original.jsonl" --max-hours 1.5 || exit 1

echo; echo "=== ARM 2/2: renamed, unsteered (the damage) ==="
python nla/src/ase_tf_run.py --packs "$OUT/packs_ren_subset.jsonl" --condition renamed_unsteered \
  --model llama8b --out "$OUT/renamed_unsteered.jsonl" --max-hours 1.5 || exit 1

echo; echo "=== H-R1a / H-R1b frozen rules ==="
python - "$OUT/original.jsonl" "$OUT/renamed_unsteered.jsonl" "$OUT/gate_stats.json" <<'PY'
import json, sys
import numpy as np
SEED, NB = 20260724, 10000
a={json.loads(l)["snippet"]:json.loads(l) for l in open(sys.argv[1]) if l.strip()}
b={json.loads(l)["snippet"]:json.loads(l) for l in open(sys.argv[2]) if l.strip()}
s=sorted(set(a)&set(b))
def cw(d,key):
    tc=sum(d[x]["n_cases"] for x in s)
    return sum(d[x][key]*d[x]["n_cases"] for x in s)/tc if tc else float("nan")
st={"n_snippets":len(s),"seed":SEED}
for k in (1,2,3):
    st[f"original_pass@{k}"]=cw(a,f"pass@{k}"); st[f"renamed_pass@{k}"]=cw(b,f"pass@{k}")
d=np.array([a[x]["pass@1"]-b[x]["pass@1"] for x in s])
w=np.array([a[x]["n_cases"] for x in s],dtype=float)
rng=np.random.default_rng(SEED)
idx=rng.integers(0,len(s),size=(NB,len(s)))
boot=np.sort([(d[i]*w[i]).sum()/w[i].sum() for i in idx])
drop=float((d*w).sum()/w.sum()); lo,hi=float(boot[int(.025*NB)]),float(boot[int(.975*NB)])
st["drop_pass@1"]=drop; st["drop_ci95"]=[lo,hi]
st["parsed_original"]=float(np.mean([a[x]["parsed_frac"] for x in s]))
st["parsed_renamed"]=float(np.mean([b[x]["parsed_frac"] for x in s]))
o1=st["original_pass@1"]
st["verdict_R1a"]=("PROTOCOL-BROKEN" if o1<0.55 else "PROTOCOL-REPLICATED" if 0.60<=o1<=0.90 else "PROTOCOL-ATYPICAL")
st["verdict_R1b"]=("DAMAGE-REPLICATES" if drop>=0.15 and lo>0 else
                   "DAMAGE-ABSENT" if drop<0.05 else "DAMAGE-PARTIAL")
print(json.dumps(st,indent=1))
print(f"\n[R1] original P@1 {o1:.4f} -> {st['verdict_R1a']}  (their Qwen2.5-7B: 0.7649; chance 0.50)")
print(f"[R1] renamed  P@1 {st['renamed_pass@1']:.4f}  (their renamed: 0.4020)")
print(f"[R1] DROP {drop:+.4f} [{lo:+.4f},{hi:+.4f}] -> {st['verdict_R1b']}  (theirs: -0.363)")
json.dump(st,open(sys.argv[3],"w"),indent=1)
PY
echo "# done $(date -u +%FT%TZ)"
