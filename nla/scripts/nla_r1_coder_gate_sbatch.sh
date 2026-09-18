#!/bin/bash
#SBATCH --job-name=nla_r1_coder
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r1_coder.out
# H-R1c -- is the ASE-2026 identifier-renaming damage CODER-MODEL-SPECIFIC?
#
# H-R1b returned DAMAGE-ABSENT on Llama-3.1-8B under their own protocol: original P@1 0.7488 (vs their
# Qwen2.5-7B's 0.7649 -- protocol replicated), renamed 0.7189, drop +0.0300 [-0.0553,+0.1119] against
# their -0.363. Their four evaluated models are Qwen2.5-7B/14B and DeepSeek-6.7B/V2-Lite: ALL coder
# models and all barred here. So the live hypothesis is that name-reliance is a property of
# code-specialised models, which also fits their own Table 4 (the same transform costs -36.3 on
# HumanEval-X but only -8.6 on CruxEval-X) and Papers 2-3 (coder models rho ~ 0 against human
# difficulty vs 0.30-0.47 for reasoning-tuned).
#
# Identical corpus, identical case packs, identical subset, identical decoding -- only the model
# changes. $1 = HF model id, $2 = short tag.
set -uo pipefail
MODEL_ID="${1:?usage: sbatch nla_r1_coder_gate_sbatch.sh <hf-model-id> <tag>}"
TAG="${2:?missing tag}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
OUT=$A/gate_$TAG
mkdir -p "$OUT"
echo "# H-R1c coder gate · $MODEL_ID ($TAG) · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_tf_run.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# the SAME subset files the Llama gate used, so every model sees identical items and cases
for f in "$A/gate/packs_orig_subset.jsonl" "$A/gate/packs_ren_subset.jsonl"; do
  [ -s "$f" ] || { echo "REFUSED: missing $f (run the Llama gate first)"; exit 2; }
done
wc -l "$A/gate/packs_orig_subset.jsonl" "$A/gate/packs_ren_subset.jsonl"

echo; echo "=== $TAG ARM 1/2: original ==="
python nla/src/ase_tf_run.py --packs "$A/gate/packs_orig_subset.jsonl" --condition original \
  --model-id "$MODEL_ID" --out "$OUT/original.jsonl" --max-hours 1.5 || exit 1
echo; echo "=== $TAG ARM 2/2: renamed, unsteered ==="
python nla/src/ase_tf_run.py --packs "$A/gate/packs_ren_subset.jsonl" --condition renamed_unsteered \
  --model-id "$MODEL_ID" --out "$OUT/renamed_unsteered.jsonl" --max-hours 1.5 || exit 1

echo; echo "=== $TAG H-R1a/H-R1b frozen rules (unchanged thresholds) ==="
python - "$OUT/original.jsonl" "$OUT/renamed_unsteered.jsonl" "$OUT/gate_stats.json" "$MODEL_ID" <<'PY'
import json, sys
import numpy as np
SEED, NB = 20260724, 10000
a={json.loads(l)["snippet"]:json.loads(l) for l in open(sys.argv[1]) if l.strip()}
b={json.loads(l)["snippet"]:json.loads(l) for l in open(sys.argv[2]) if l.strip()}
s=sorted(set(a)&set(b))
def cw(d,key):
    tc=sum(d[x]["n_cases"] for x in s)
    return sum(d[x][key]*d[x]["n_cases"] for x in s)/tc if tc else float("nan")
st={"model":sys.argv[4],"n_snippets":len(s),"seed":SEED}
for k in (1,2,3):
    st[f"original_pass@{k}"]=cw(a,f"pass@{k}"); st[f"renamed_pass@{k}"]=cw(b,f"pass@{k}")
d=np.array([a[x]["pass@1"]-b[x]["pass@1"] for x in s]); w=np.array([a[x]["n_cases"] for x in s],dtype=float)
rng=np.random.default_rng(SEED); idx=rng.integers(0,len(s),size=(NB,len(s)))
boot=np.sort([(d[i]*w[i]).sum()/w[i].sum() for i in idx])
drop=float((d*w).sum()/w.sum()); lo,hi=float(boot[int(.025*NB)]),float(boot[int(.975*NB)])
st.update({"drop_pass@1":drop,"drop_ci95":[lo,hi],
           "parsed_original":float(np.mean([a[x]["parsed_frac"] for x in s])),
           "parsed_renamed":float(np.mean([b[x]["parsed_frac"] for x in s]))})
o1=st["original_pass@1"]
st["verdict_R1a"]=("PROTOCOL-BROKEN" if o1<0.55 else "PROTOCOL-REPLICATED" if 0.60<=o1<=0.90 else "PROTOCOL-ATYPICAL")
st["verdict_R1b"]=("DAMAGE-REPLICATES" if drop>=0.15 and lo>0 else "DAMAGE-ABSENT" if drop<0.05 else "DAMAGE-PARTIAL")
print(json.dumps(st,indent=1))
print(f"\n[R1c] {st['model']}  original P@1 {o1:.4f} -> {st['verdict_R1a']}")
print(f"[R1c] renamed P@1 {st['renamed_pass@1']:.4f} · DROP {drop:+.4f} [{lo:+.4f},{hi:+.4f}] -> {st['verdict_R1b']}")
print(f"[R1c] reference: their Qwen2.5-7B 0.7649 -> 0.4020 (-0.363); our Llama-3.1-8B 0.7488 -> 0.7189 (+0.0300)")
json.dump(st,open(sys.argv[3],"w"),indent=1)
PY
echo "# done $(date -u +%FT%TZ)"
