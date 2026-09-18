#!/bin/bash
#SBATCH --job-name=nla_fsdp_equiv
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=02:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_fsdp_equiv.out
# GATE before any 12B spend: is the 2-GPU FSDP path the SAME METHOD as the proven single-GPU one?
#
# FSDP was chosen over 8-bit Adam / bf16 weights precisely because it leaves the optimizer and
# precision recipe intact -- so the 12B pairs stay comparable with the 33 existing 4B pairs and
# with the released kitft/nla-gemma3-12b-L32 pair. That argument is worthless unless the
# multi-GPU path is verified, hence this test.
#
# Runs the SAME 4B AV stage twice at a fixed seed: (A) 1 GPU, (B) torchrun 4 GPUs --fsdp, then
# compares the per-step train_log.jsonl.
#
# Defaults: h200, world=2. Tried world=4 on the uncapped h100 partition first (job 389121) --
# g-04-02's 4 x H100 80 GB is the only node that fits a 4-way 12B run, but all 4 GPUs are held by
# another user's 22 h GRPO job and the node is RESERVED. g-06-01's 47 GB MIG slices can host the
# FSDP arm but NOT the single-GPU baseline: 4B fp32 weights+grads+Adam is 59.5 GB of irreducible
# state, which no micro-batch reduction touches.
#
# world=2 on h200 is not a compromise: 12B needs ~206 GB, i.e. ~103 GB/GPU at 2-way, which FITS a
# 143 GB H200 -- so this is a production route, not just a test rig. Override for the 4 x H100 node
# when it frees: PART=h100 GRES=gpu:nvidia_h100_80gb_hbm3:4 NPROC=4 MB=4 sbatch ...
# MB is the micro-batch for BOTH arms; identical on each side keeps the comparison exact.
#
# THE SHARPEST DIAGNOSTIC IS grad_norm AT STEP 1. Both likely bugs show up there, halved:
#   * the normalizer trap (each rank dividing by the GLOBAL token count while FSDP *averages*
#     gradients) scales every gradient by 1/world  -> grad_norm x0.5
#   * per-shard gradient clipping instead of the global norm -> grad_norm ~x1/sqrt(world)
# Step-1 LOSS, by contrast, is computed before any update, so it validates the data partition and
# accumulation only. Later steps then test that the trajectories stay together.
#
# Pre-declared tolerances (set before looking at any output):
#   step-1 loss      : relative difference < 1e-4   (identical weights; only fp summation order differs)
#   step-1 grad_norm : relative difference < 5e-3   (catches the x0.5 / x0.71 signatures above)
#   steps 2..N       : mean relative loss difference < 2e-2, final step < 3e-2
#                      (fused AdamW is unavailable for DTensor -> foreach under --fsdp: same AdamW,
#                       not bit-identical, so trajectories drift slightly)
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS=8

K=${K:-2}; STEPS=${STEPS:-20}; LIMIT=${LIMIT:-4000}; NPROC=${NPROC:-2}; MB=${MB:-8}
# --root is the trainer's INPUT root as well as its output root (corpus/, explain/, acts/ are read
# from it). Job 389065 failed because I pointed it at an empty dir. $SRC is a scratch root whose
# corpus/ and explain/ are symlinks into /work and whose acts/ were re-extracted by
# nla_fsdp_equiv_acts.sh; A and B get their own copies of it so the two runs cannot collide with
# each other or with the 34 trained pairs in /work.
SRC=/scratch/juno/jvl210002/fsdp_equiv_root
OUT=/scratch/juno/jvl210002/fsdp_equiv
A=$OUT/single; B=$OUT/fsdp2
for d in "$SRC/corpus" "$SRC/explain" "$SRC/acts/L$K.npy"; do
  [ -e "$d" ] || { echo "MISSING INPUT $d -- run nla/scripts/nla_fsdp_equiv_acts.sh first"; exit 1; }
done
rm -rf "$A" "$B"; mkdir -p "$A" "$B"
for d in "$A" "$B"; do
  ln -sfn "$SRC/corpus" "$d/corpus"; ln -sfn "$SRC/explain" "$d/explain"; ln -sfn "$SRC/acts" "$d/acts"
done

echo "# FSDP equivalence · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_train.py nla/src/fsdp_util.py nla/configs/nla_ml.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo; echo "=== unit: the helper contracts (partition cover + the normalizer correction) ==="
python -m pytest nla/tests/test_fsdp_util.py -q -p no:faulthandler || exit 1

echo; echo "=== A: single GPU (the proven path, --fsdp ABSENT) ==="
CUDA_VISIBLE_DEVICES=0 python nla/src/nla_train.py --stage sft_av --layer $K \
  --limit $LIMIT --max-steps $STEPS --micro-batch $MB --root "$A" || exit 1

echo; echo "=== B: $NPROC GPUs, FSDP2 ==="
torchrun --standalone --nproc_per_node=$NPROC nla/src/nla_train.py --stage sft_av --layer $K \
  --fsdp --limit $LIMIT --max-steps $STEPS --micro-batch $MB --root "$B" || exit 1

echo; echo "=== COMPARE ==="
python - "$A" "$B" "$K" "$NPROC" <<'PY'
import json, sys
from pathlib import Path
a, b, K, W = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def rows(r):
    return {d["step"]: d for d in map(json.loads, open(r / f"L{K}/av/train_log.jsonl"))}
A, B = rows(a), rows(b)
steps = sorted(set(A) & set(B))
assert steps, "no common steps logged"
rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-12)
s1 = steps[0]
l1, g1 = rel(A[s1]["loss"], B[s1]["loss"]), rel(A[s1]["grad_norm"], B[s1]["grad_norm"])
ratio = B[s1]["grad_norm"] / A[s1]["grad_norm"]
losses = [rel(A[s]["loss"], B[s]["loss"]) for s in steps]
mean_rel, last_rel = sum(losses) / len(losses), losses[-1]
print(f"  step {s1}: loss A={A[s1]['loss']:.6f} B={B[s1]['loss']:.6f} rel={l1:.2e}")
print(f"  step {s1}: grad_norm A={A[s1]['grad_norm']:.6f} B={B[s1]['grad_norm']:.6f} rel={g1:.2e} ratio={ratio:.4f}")
print(f"  steps {steps[0]}..{steps[-1]} (n={len(steps)}): mean rel loss diff {mean_rel:.2e}, final {last_rel:.2e}")
for s in steps:
    print(f"    step {s:>4}  A {A[s]['loss']:.6f}  B {B[s]['loss']:.6f}  rel {rel(A[s]['loss'], B[s]['loss']):.2e}")
ok = True
def chk(name, val, lim):
    global ok
    good = val < lim
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {val:.2e} < {lim:.0e}")
chk("step-1 loss rel", l1, 1e-4)
chk("step-1 grad_norm rel", g1, 5e-3)
chk("mean rel loss diff", mean_rel, 2e-2)
chk("final rel loss diff", last_rel, 3e-2)
if abs(ratio - 1.0 / W) < 0.06:
    print(f"  DIAGNOSIS: grad_norm ratio ~1/{W} -> the NORMALIZER TRAP (missing scale_for_world).")
elif abs(ratio - W ** -0.5) < 0.06:
    print(f"  DIAGNOSIS: grad_norm ratio ~1/sqrt({W}) -> clipping is seeing per-shard norms.")
print("EQUIVALENCE: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
rc=$?
echo; echo "=== peak memory ==="
for r in "$A" "$B"; do
  python - "$r" "$K" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / f"L{sys.argv[2]}/av/train_log.jsonl"
m = max(json.loads(l).get("mem_gb", 0) for l in open(p))
print(f"  {sys.argv[1]}: peak {m:.1f} GB/GPU")
PY
done
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
