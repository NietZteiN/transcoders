#!/bin/bash
#SBATCH --job-name=nla_a13_churn
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_a13_churn.out
# H-A13 -- the churn control. Re-run the dataset_c screen a SECOND time under the identical command and
# count per-item label changes. Pre-registered in log/nla-harness/2026-09-14_churn-control-prereg.md.
#
# WHY: greedy decoding is not reproducible on this host (same prompt ids, same model, same job: 3 of 6
# generations diverged at tokens 32/73/201, one flipping l1b_correct), and the discordant cells of the
# decoy 2x2 are symmetric on BOTH corpora (banked 7 flippable / 5 reverse, McNemar p = 0.774;
# dataset_c 4/3, p = 1.000). The reverse cell cannot be a real effect -- misleading names cannot make
# the model better -- so it is a direct read on the noise floor. This measures that floor head-on.
#
# There is NO new code and NO intervention here: the same command, a different --out-dir. That is the
# entire design, and it is the control every accuracy measurement in this programme has lacked.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
RUN1=$PROJ/data/nla/p0/trace_llr/gemma4b_c/traces.jsonl
RUN2=/scratch/juno/jvl210002/nla_churn_run2
echo "# H-A13 churn control · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/host_traces.py nla/src/steer_run.py
nvidia-smi --query-gpu=index,name --format=csv,noheader
[ -f "$RUN1" ] || { echo "REFUSED: run-1 reference missing: $RUN1"; exit 2; }
wc -l "$RUN1"

echo; echo "=== RUN 2: identical command, 120 items, different out-dir ==="
python nla/src/host_traces.py --model gemma4b --datasets dataset_c --max-new-gen 2600 --fast-host \
  --limit 120 --out-dir "$RUN2" --max-hours 2.0; rc=$?
echo "--- run2 rc=$rc"

echo; echo "=== H-A13: the frozen churn rules ==="
python - "$RUN1" "$RUN2/traces.jsonl" <<'PY'
import json, sys
from math import comb
a={json.loads(l)['snippet_id']:json.loads(l) for l in map(str.strip,open(sys.argv[1])) if l}
b={json.loads(l)['snippet_id']:json.loads(l) for l in map(str.strip,open(sys.argv[2])) if l}
sids=sorted(set(a)&set(b))
def mc(x,y):
    m=x+y
    return 1.0 if not m else min(1.0,2*sum(comb(m,i) for i in range(min(x,y)+1))/2**m)
st={"experiment":"A13_churn_control","n_common":len(sids),"tiers":{}}
for tier in ("l0","l1b"):
    ch=[s for s in sids if bool(a[s][f"{tier}_correct"])!=bool(b[s][f"{tier}_correct"])]
    tok=[s for s in sids if list(a[s][f"{tier}_reply_ids"])!=list(b[s][f"{tier}_reply_ids"])]
    st["tiers"][tier]={"discordant_label":len(ch),"discordant_rate":len(ch)/len(sids) if sids else float('nan'),
                       "divergent_tokens":len(tok),"divergent_token_rate":len(tok)/len(sids) if sids else float('nan'),
                       "items":ch[:12]}
# the flippable rate in run 1, on the SAME items, is what the floor is judged against
fl=[s for s in sids if a[s]['l0_correct'] and not a[s]['l1b_correct']]
rv=[s for s in sids if a[s]['l1b_correct'] and not a[s]['l0_correct']]
st["run1_flippable"]=len(fl); st["run1_reverse"]=len(rv)
st["run1_flippable_rate"]=len(fl)/len(sids) if sids else float('nan')
st["run1_mcnemar_decoy_p"]=mc(len(rv),len(fl))
d=st["tiers"]["l1b"]["discordant_rate"]; f=st["run1_flippable_rate"]
st["verdict"]=("CHURN-DOMINATES" if d>=f else "CHURN-SUBDOMINANT" if d<0.5*f else "CHURN-INTERMEDIATE")
# doubly-confirmed flippable: flippable in BOTH runs (what a re-specified H-S14b would have to use)
dbl=[s for s in sids if a[s]['l0_correct'] and not a[s]['l1b_correct']
                     and b[s]['l0_correct'] and not b[s]['l1b_correct']]
st["doubly_confirmed_flippable"]=len(dbl); st["doubly_confirmed_items"]=dbl
print(json.dumps(st,indent=1)[:2200])
print(f"\n[A13] {st['verdict']}: L1b label discordance {d:.4f} vs run-1 flippable rate {f:.4f}")
print(f"[A13] token-level divergence: L0 {st['tiers']['l0']['divergent_token_rate']:.4f} · "
      f"L1b {st['tiers']['l1b']['divergent_token_rate']:.4f}")
print(f"[A13] flippable {st['run1_flippable']} · reverse {st['run1_reverse']} · decoy McNemar p="
      f"{st['run1_mcnemar_decoy_p']:.3f} · DOUBLY-CONFIRMED flippable {st['doubly_confirmed_flippable']}")
open("/scratch/juno/jvl210002/nla_churn_run2/a13_stats.json","w").write(json.dumps(st,indent=1))
PY
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
