#!/bin/bash
#SBATCH --job-name=r15_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r15_smoke.out
# H-R15 smoke: is --seed actually wired to the generations? Three 3-snippet runs of the SAME arm --
# two at the same seed, one at a different seed -- and a diff of the generated predictions. This is a
# wiring check, not a result: 3 snippets cannot measure the floor.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026; F=$A/full; OUT=$F/repro_smoke; mkdir -p "$OUT"
echo "# H-R15 smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
for tag in s724a s724b s917; do
  case $tag in s724a|s724b) SD=20260724;; s917) SD=20260917;; esac
  echo "--- $tag (seed $SD) ---"
  python nla/src/ase_steer_run.py --packs $F/packs_ren_smoke.jsonl --arm unsteered --seed $SD \
    --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template \
    --vectors-config nla/configs/ase_vectors_full.yaml --out "$OUT/$tag.jsonl" --max-hours 0.3 || true
done
python - <<'PY'
import json
from pathlib import Path
O=Path("/scratch/juno/jvl210002/ase2026/full/repro_smoke")
def preds(t):
    out={}
    for l in open(O/f"{t}.jsonl"):
        r=json.loads(l)
        for i,run in enumerate(r["runs"]):
            for c,v in run["pred"].items(): out[(r["snippet"],i,c)]=v
    return out
a,b,c=preds("s724a"),preds("s724b"),preds("s917")
def agree(x,y):
    k=set(x)&set(y); return (sum(x[i]==y[i] for i in k)/len(k), len(k)) if k else (float("nan"),0)
sa,n1=agree(a,b); sd,n2=agree(a,c)
print(f"[SMOKE] same-seed  s724a vs s724b : agreement {sa:.4f} over {n1} case-runs")
print(f"[SMOKE] diff-seed  s724a vs s917  : agreement {sd:.4f} over {n2} case-runs")
print("[SMOKE] --seed is wired iff the diff-seed agreement is materially below 1.0")
print("[SMOKE] NOTE: same-seed < 1.0 would already be the H-R15b answer in miniature.")
PY
echo "# done rc=$? $(date -u +%FT%TZ)"
