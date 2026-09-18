#!/bin/bash
#SBATCH --job-name=r18_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=00:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r18_smoke.out
# H-R18a: is greedy decoding EXACT? Two greedy runs of the same arm on the same snippets must agree
# on every case. This is the premise the whole greedy pass rests on -- if it fails, H-R18b/c are not read.
# Also checks the --runs guard refuses the wasteful configuration.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/greedy_smoke; mkdir -p "$OUT"
echo "# H-R18a smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
echo "--- guard: --greedy with --runs 3 must be REFUSED (rc=2) ---"
python nla/src/ase_steer_run.py --packs $F/packs_ren_smoke.jsonl --arm unsteered --greedy --runs 3 \
  --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
  --vectors-config nla/configs/ase_vectors_full.yaml --out /dev/null --max-hours 0.1
echo "guard rc=$? (expect 2)"
for t in g1 g2; do
  echo "--- greedy run $t ---"
  python nla/src/ase_steer_run.py --packs $F/packs_ren_smoke.jsonl --arm unsteered --greedy --runs 1 \
    --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
    --vectors-config nla/configs/ase_vectors_full.yaml --out "$OUT/$t.jsonl" --max-hours 0.2 || true
done
python - <<'PY'
import json
from pathlib import Path
O=Path("/scratch/juno/jvl210002/ase2026/full/greedy_smoke")
def preds(t):
    d={}
    for l in open(O/f"{t}.jsonl"):
        r=json.loads(l)
        for i,run in enumerate(r["runs"]):
            for c,v in run["pred"].items(): d[(r["snippet"],i,c)]=v
    return d
def acc(t):
    rows=[json.loads(l) for l in open(O/f"{t}.jsonl")]
    n=sum(r["n_cases"] for r in rows)
    return sum(r["pass@1"]*r["n_cases"] for r in rows)/n
a,b=preds("g1"),preds("g2"); k=set(a)|set(b)
ag=sum(a.get(i)==b.get(i) for i in k)/len(k)
da=abs(acc("g1")-acc("g2"))
print(f"[SMOKE] greedy g1 vs g2: agreement {ag:.6f} over {len(k)} cases · |Dacc| {da:.9f}")
print(f"[SMOKE] H-R18a would read: {'GREEDY-EXACT' if ag==1.0 and da<=1e-9 else 'GREEDY-NOT-EXACT'}")
PY
echo "# done $(date -u +%FT%TZ)"
