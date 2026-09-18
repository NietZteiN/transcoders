#!/bin/bash
#SBATCH --job-name=nla_e14_loonorm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_e14_loonorm.out
# H-E14 -- is the "training-free" erasure vector DEPLOYABLE? Every erasure arm steps by dn = ||h0_s - h1b_s||,
# the true per-span displacement magnitude taken from the clean activation, which the trained AV/AR never sees.
# The direction is honestly leave-one-out; the length is borrowed. `erase_loonorm` borrows nothing: direction AND
# magnitude are estimated from the OTHER items' spans only.
# Pre-registered in log/nla-harness/2026-09-13_erasure-oracle-norm-prereg.md. Frozen there:
#   H-E14  erase_1.0 - erase_loonorm < +3.0 (or CI including 0) -> ORACLE-FREE ; >= +3.0 CI>0 -> ORACLE-SCALED
#   H-E14b erase_loonorm - erase_flip >= +3.0 CI>0 -> DEPLOYABLE-DIRECTIONAL else DEPLOYABLE-NOISE
#           (H-E14b is the number that belongs in the memo: a method with no oracle access could reproduce it)
# Sites: 4B L7 (the headline site, alpha PLATEAU) · 4B L13 (the selectivity site from H-E8, alpha TURNS OVER) ·
# 12B L32 (the refutation site, alpha MONOTONE). No AV/AR is loaded anywhere here and the 12B vectors banked by
# job 393219 are reused, so its 50-minute vectors stage does NOT repeat.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
OUT=/scratch/juno/jvl210002/nla_e14_loonorm
REL=/scratch/juno/jvl210002/nla_ml_gemma12b_released
OURS=/scratch/juno/jvl210002/nla_ml_gemma12b
CFG12=nla/configs/nla_ml_gate_12b.yaml
TRACES12=$PROJ/data/nla/p0/trace_llr/gemma12b/traces.jsonl
echo "# H-E14 oracle-free erasure · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_erasure.py nla/src/steer.py nla/scripts/nla_e14_loonorm_sbatch.sh
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_erasure.py nla/tests/test_steer.py -q -p no:faulthandler \
  -k "erasure or erase or loo or rules or identity or vector or replacer or no_grad or flip or anchoring or vacuous or loonorm or oracle" || exit 1

rc_all=0
for K in 7 13; do
  echo; echo "=== 4B L$K  (alpha $( [ $K = 7 ] && echo PLATEAU || echo TURNS-OVER )) ==="
  python nla/src/nla_erasure.py --layer $K --no-band --out "$OUT/gemma4b_L$K" --max-hours 0.5; rc=$?
  echo "--- 4B L$K rc=$rc"; [ "$rc" != "0" ] && rc_all=$rc
done

echo; echo "=== 12B L32 (the refutation site; released pair, compare-root = our SFT pair) ==="
python nla/src/nla_erasure.py --config "$CFG12" --root "$REL" --traces "$TRACES12" --layer 32 \
  --compare-root "$OURS" --out "$OUT/gemma12b_L32" --max-hours 0.75; rc=$?
echo "--- 12B L32 rc=$rc"; [ "$rc" != "0" ] && rc_all=$rc

echo; echo "=== H-E14 summary across sites ==="
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
base = Path(sys.argv[1])
sites = [("4B  L7 ", "gemma4b_L7", "L7"), ("4B  L13", "gemma4b_L13", "L13"),
         ("12B L32", "gemma12b_L32", "L32")]
print(f"{'site':>8} {'e_1.0':>8} {'loonorm':>8} {'flip':>8} {'edit':>8} {'E14 (oracle worth)':>26} "
      f"{'verdict':>14} {'E14b (deployable)':>24} {'verdict':>24}")
for tag, d, S in sites:
    f = base / d / "erasure_stats.json"
    if not f.exists():
        print(f"{tag:>8}  (missing)"); continue
    st = json.loads(f.read_text())
    if not st.get("reportable", True) or "rules" not in st:
        print(f"{tag:>8}  REFUSED by its identity gate ({st.get('verdict', 'gate failure')}) — nothing read. "
              f"worst: {st.get('identity', {}).get('worst_abs_diff')}")
        continue
    A = st["arms"]; R = st["rules"]
    m = lambda a: A.get(f"dG_{a}_{S}", {}).get("mean", float("nan"))
    ci = lambda k: (f"{R[k]['mean']:+.2f} [{R[k]['ci95'][0]:+.2f},{R[k]['ci95'][1]:+.2f}]" if k in R else "-")
    print(f"{tag:>8} {m('erase_1.0'):>8.2f} {m('erase_loonorm'):>8.2f} {m('erase_flip'):>8.2f} {m('edit'):>8.2f} "
          f"{ci('H_E14'):>26} {R.get('verdict_E14','?'):>14} {ci('H_E14b'):>24} {R.get('verdict_E14b','?'):>24}")
    print(f"{'':>8} identity {st['identity']['passes']} · share(loonorm) "
          f"{m('erase_loonorm')/m('swap'):.3f} vs share(e1.0) {m('erase_1.0')/m('swap'):.3f}")
PY
echo "# done rc=$rc_all $(date -u +%FT%TZ)"
exit $rc_all
