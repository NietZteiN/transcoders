#!/bin/bash
#SBATCH --job-name=r14_crossfit
#SBATCH --partition=normal,h200,h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r14_crossfit.out
# H-R14 -- refit the ridge_map / role_proto vectors by NESTED GROUPED CROSS-FIT over all 148 aligned
# snippets. CPU ONLY (the pool capture is reused; no forward passes).
# Pre-registered: log/nla-harness/2026-09-16_full-corpus-prereg.md
#
# STAGE 1 IS A REGRESSION GATE, NOT A RESULT. ase_vectors.py was refactored (fit_nuisance / apply_to) to
# share the fitting code between the H-R7 single split and the new cross-fit. Before the cross-fit is
# trusted, the refactored SINGLE-SPLIT path must reproduce the banked H-R7 chat fit exactly -- lambda,
# rank, held-out cosine, margin and the CV-table crc. A mismatch means the refactor changed the fit, and
# every vector the bake-off already used is suspect. This costs ~25 min of CPU and no GPU.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
A=/scratch/juno/jvl210002/ase2026
echo "# H-R14 cross-fit · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_vectors.py nla/configs/ase_vectors_full.yaml nla/configs/ase_vectors_chat.yaml

echo "=== stage 1: REGRESSION GATE — refactored single split must reproduce the banked H-R7 fit ==="
python nla/src/ase_vectors.py --config nla/configs/ase_vectors_chat.yaml \
  --test-subset $A/gate/packs_orig_subset.jsonl \
  --out $A/full/regress_vectors_chat.pt --report $A/full/regress_vectors_chat.fit.json; rc=$?
[ $rc -ne 0 ] && { echo "# FATAL: regression fit exited $rc"; exit $rc; }
python - <<'PY'
import json, sys
banked = json.load(open("/scratch/juno/jvl210002/ase2026/vectors_codellama7b_L7_chat.fit.json"))
new    = json.load(open("/scratch/juno/jvl210002/ase2026/full/regress_vectors_chat.fit.json"))
b, n = banked["ridge_map"], new["ridge_map"]
checks = [
    ("lambda", b["best"]["lambda"], n["best"]["lambda"], 0.0),
    ("rank", b["best"]["rank"], n["best"]["rank"], 0.0),
    ("cv cos", b["best"]["cos"], n["best"]["cos"], 1e-9),
    ("mean_only_cos", b["mean_only_cos"], n["mean_only_cos"], 1e-9),
    ("margin", b["margin"], n["margin"], 1e-9),
    ("mean_delta_norm", b["mean_delta_norm"], n["mean_delta_norm"], 1e-9),
    ("erasure_cos_check", b["erasure_cos_check"], n["erasure_cos_check"], 1e-9),
    ("n_pool_spans", banked["n_pool_spans"], new["n_pool_spans"], 0.0),
    ("n_test_spans", banked["n_test_spans"], new["n_test_spans"], 0.0),
]
bad = [(k, x, y) for k, x, y, tol in checks if abs(float(x) - float(y)) > tol]
for k, x, y, tol in checks:
    print(f"  {k:18s} banked={x} new={y} {'OK' if abs(float(x)-float(y))<=tol else 'MISMATCH'}")
# The full CV table is compared NUMERICALLY, not by crc. The first run of this gate (job 408485) failed
# on crc alone while all ten statistics above matched to full double precision: 4 of 20 table cells
# differed by <= 5.551e-17, i.e. sub-ULP (double eps 2.22e-16), because BLAS summation order depends on
# the thread count and the banked fit ran at a different one. A crc is a bit-exactness test on a quantity
# that is NOT bit-reproducible across thread counts, so it was the wrong invariant. The invariants that
# actually matter -- every cell within 1e-12, and the argmax/top-5 ORDERING that selects (lambda, rank)
# -- are checked instead, and are strictly stronger than what the crc was standing in for. The frozen
# science threshold (gate_cos_margin >= 0.05) is untouched. Recorded: 2026-09-16_full-corpus-results.md.
tb, tn = banked["ridge_map"]["table"], new["ridge_map"]["table"]
if set(tb) != set(tn):
    bad.append(("cv table keys", sorted(tb), sorted(tn)))
else:
    worst = max(abs(tb[k] - tn[k]) for k in tb)
    ob = sorted(tb, key=lambda k: -tb[k]); on = sorted(tn, key=lambda k: -tn[k])
    print(f"  cv table           {len(tb)} cells · max|diff|={worst:.3e} (tol 1e-12) · "
          f"argmax {ob[0]} vs {on[0]} · top-5 order {'same' if ob[:5]==on[:5] else 'DIFFERENT'}")
    if worst > 1e-12: bad.append(("cv table max|diff|", worst, 1e-12))
    if ob[:5] != on[:5]: bad.append(("cv table top-5 ordering", ob[:5], on[:5]))
rp_b, rp_n = banked["role_proto"]["resolved_at"], new["role_proto"]["resolved_at"]
if rp_b != rp_n:
    bad.append(("role_proto.resolved_at", rp_b, rp_n))
print(f"  role_proto         banked={rp_b} new={rp_n} {'OK' if rp_b==rp_n else 'MISMATCH'}")
if bad:
    print(f"[GATE] FATAL: refactor changed the H-R7 fit: {bad}"); sys.exit(3)
print("[GATE] PASS: refactored single split reproduces the banked H-R7 chat fit exactly")
PY
rc=$?; [ $rc -ne 0 ] && { echo "# FATAL: regression gate failed rc=$rc"; exit $rc; }

echo "=== stage 2: nested grouped cross-fit over all 148 snippets ==="
python nla/src/ase_vectors.py --config nla/configs/ase_vectors_full.yaml --cross-fit 5; rc=$?
[ $rc -ne 0 ] && { echo "# FATAL: cross-fit exited $rc"; exit $rc; }
python - <<'PY'
import json, torch
r = json.load(open("/scratch/juno/jvl210002/ase2026/full/vectors_codellama7b_L7_full.fit.json"))
print(f"[XFIT] mode={r['mode']} snippets={r['n_snippets']} spans={r['n_spans']} verdict={r['verdict']}")
for f in r["cross_fit_folds"]:
    print(f"  fold {f['fold']}: lambda={f['best']['lambda']:g} rank={f['best']['rank']} "
          f"cos={f['best']['cos']:.4f} mean-only={f['mean_only_cos']:.4f} margin={f['margin']:+.4f} "
          f"{'PASS' if f['gate_pass'] else 'FAIL'}")
rm = r["ridge_map"]
print(f"[XFIT] GATE {'PASS' if rm['gate_pass'] else 'FAIL'} · margin min {rm['margin_min']:+.4f} "
      f"mean {rm['margin_mean']:+.4f} · {r['ridge_gate']}")
v = torch.load("/scratch/juno/jvl210002/ase2026/full/vectors_codellama7b_L7_full.pt", weights_only=False)
nsn = len(v["arms"]["ridge_map"]); nsp = sum(len(d) for d in v["arms"]["ridge_map"].values())
print(f"[XFIT] vectors: {nsp} spans over {nsn} snippets (pool had 4799 spans / 148 snippets)")
assert nsn == 148 and nsp == 4799, f"coverage gap: {nsn} snippets / {nsp} spans"
print("[XFIT] coverage OK — every aligned span got a cross-fitted vector")
PY
rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
