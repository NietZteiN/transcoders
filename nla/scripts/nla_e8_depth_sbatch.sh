#!/bin/bash
#SBATCH --job-name=nla_e8_depth
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_e8_depth.out
# H-E8 -- the WITHIN-MODEL depth mirror. Job 393219 found the training-free erasure vector non-directional at
# 12B L32 after it matched the trained edit at 4B L7, but it moved depth AND host size together. This runs the
# 4B erasure arms down a depth ladder on the already-banked 34 layers of vectors -- host only, no AV/AR, no
# training -- so depth and host separate inside one model.
# Pre-registered in log/nla-harness/2026-09-13_e8-depth-mirror-prereg.md. Frozen there:
#   ladder L3 L7 L13 L17 L23 L28 (readable: banked swap >= 10 nats; L29-33 excluded as inert -- 4B L32 swap
#   is +0.11 and CANNOT mirror 12B L32); per-layer gate erase_1.0 - erase_flip >= +3.0 CI>0 -> DIRECTIONAL@K;
#   DEPTH-EXPLAINS = NOISE at both L17 and L23 · HOST-EXPLAINS = DIRECTIONAL@L23 ·
#   CROSSOVER-INSIDE-4B = DIRECTIONAL@L17 + NOISE@L23 · MIRROR-UNINFORMATIVE = NOISE@L7 (control fails).
# L7 is the banked positive control and MUST reproduce (+35.75 erase_1.0 / +32.06 edit / share 0.489).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
OUT=/scratch/juno/jvl210002/nla_e8_depth
echo "# H-E8 depth mirror · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_erasure.py nla/src/steer.py nla/src/nla_ml_gate.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_erasure.py nla/tests/test_steer.py -q -p no:faulthandler \
  -k "erasure or erase or loo or rules or identity or vector or replacer or no_grad or flip or anchoring or vacuous" || exit 1

# L7 FIRST: it is the banked positive control. If it does not reproduce, the ladder is uninformative and the
# run stops before spending the other five layers.
rc_all=0
for K in 7 3 13 17 23 28; do
  echo; echo "=== L$K  (primary set only, --no-band) ==="
  python nla/src/nla_erasure.py --layer $K --no-band --out "$OUT/L$K" --max-hours 0.5; rc=$?
  echo "--- L$K rc=$rc"
  if [ "$K" = "7" ] && [ "$rc" != "0" ]; then
    echo "MIRROR-UNINFORMATIVE: the L7 positive control did not pass its gates; stopping."; exit 3
  fi
  [ "$rc" != "0" ] && rc_all=$rc
done

echo; echo "=== ladder summary (share_K = erase_1.0 / swap, against H-W23's 0.493) ==="
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
print(f"{'L':>3} {'swap':>8} {'e_0.5':>8} {'e_1.0':>8} {'e_2.0':>8} {'flip':>8} {'rand':>8} {'edit':>8} "
      f"{'frgn':>8} {'share':>6} {'gate(e1-flip)':>22} {'verdict':>22}")
rows = {}
for d in sorted(base.glob("L*/erasure_stats.json"), key=lambda p: int(p.parent.name[1:])):
    st = json.loads(d.read_text()); K = int(d.parent.name[1:]); S = f"L{K}"
    A = st["arms"]; R = st["rules"]; g = R.get("H_E7c") or {}
    m = lambda a: A.get(f"dG_{a}_{S}", {}).get("mean", float("nan"))
    sw = m("swap"); share = m("erase_1.0") / sw if sw else float("nan")
    gate = f"{g.get('mean', float('nan')):+.2f} [{g.get('ci95',[float('nan')]*2)[0]:+.2f},{g.get('ci95',[float('nan')]*2)[1]:+.2f}]"
    print(f"{K:>3} {sw:>8.2f} {m('erase_0.5'):>8.2f} {m('erase_1.0'):>8.2f} {m('erase_2.0'):>8.2f} "
          f"{m('erase_flip'):>8.2f} {m('erase_rand'):>8.2f} {m('edit'):>8.2f} {m('foreign'):>8.2f} "
          f"{share:>6.3f} {gate:>22} {R.get('verdict_E7c','?'):>22}")
    rows[K] = {"verdict": R.get("verdict_E7c"), "share": share, "identity": st["identity"]["passes"],
               "spec_banked": m("edit") - m("foreign")}
D = lambda K: rows.get(K, {}).get("verdict") == "ERASURE-DIRECTIONAL"
N = lambda K: rows.get(K, {}).get("verdict") == "ERASURE-NOISE"
if N(7):                      v = "MIRROR-UNINFORMATIVE"
elif D(23):                   v = "HOST-EXPLAINS"
elif N(17) and N(23):         v = "DEPTH-EXPLAINS"
elif D(17) and N(23):         v = "CROSSOVER-INSIDE-4B"
else:                         v = "MIRROR-INDETERMINATE"
cross = next((K for K in sorted(rows) if N(K)), None)
print(f"\nH-E8 {v} · crossover (shallowest NOISE layer) {cross} · "
      f"identity all pass {all(r['identity'] for r in rows.values())}")
print("shares: " + " · ".join(f"L{K} {rows[K]['share']:.3f}" for K in sorted(rows)))
(base / "e8_ladder.json").write_text(json.dumps({"verdict": v, "crossover": cross, "layers": rows}, indent=1))
PY
echo "# done rc=$rc_all $(date -u +%FT%TZ)"
exit $rc_all
