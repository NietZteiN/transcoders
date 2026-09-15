#!/bin/bash
#SBATCH --job-name=nla_a8_budget
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_a8_budget.out
# H-A8 -- is the ACCURACY INSTRUMENT budget-limited? At MAX_NEW_GEN = 1100 the banked accuracy run (job
# 391968) never got a parseable answer out of ~20 % of generations, on every arm, and an unparsed generation
# scores as WRONG -- so that mass was unreachable by any steering write. CLAUDE.md's anchor finding is a
# ~2048-token System-2 plateau, about twice the budget used. Pre-registered in
# log/nla-harness/2026-09-13_budget-confound-prereg.md. Frozen there:
#   H-A8a parsed(noop@2600) - parsed(noop@1100) >= +0.05, CI>0 -> BUDGET-BINDING (every 391968 accuracy null
#     is then labelled budget-confounded) else BUDGET-SATURATED
#   H-A8b acc(noop@2600) - acc(noop@1100), gated on BINDING; acc_given_parsed at both budgets
#   H-A8c c3_band@2600 - noop@2600 (the oracle arm, the only one leading on the 7 flippable) -- reported with
#     NO verdict word, under the unchanged bound: max detectable ~ +0.03 overall / ~ +0.10 flippable, n=60
#   Reproduction gate: noop@1100 on 12 items must reproduce the banked greedy rows EXACTLY (deterministic).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
BANKED=$PROJ/data/nla/ml/gemma4b/gate/accuracy/accuracy_rows.jsonl
OUT=/scratch/juno/jvl210002/nla_a8_budget
echo "# H-A8 budget confound · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_accuracy.py nla/src/steer.py nla/src/steer_run.py nla/scripts/nla_a8_budget_sbatch.sh
wc -l "$BANKED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/ -q -p no:faulthandler -k "accuracy or wilson or mcnemar or replacer or no_grad" || exit 1

echo; echo "=== STAGE A: reproduction gate — noop @ 1100, first 12 items, must match the banked rows exactly ==="
python nla/src/nla_accuracy.py --arms noop --max-new-gen 1100 --no-sampled --limit 12 \
  --out "$OUT/repro" --max-hours 0.6 || exit 1
python - "$OUT/repro/accuracy_rows.jsonl" "$BANKED" <<'PY' || exit 3
import json, sys
new = {r["snippet_id"]: r for r in map(json.loads, open(sys.argv[1]))}
old = {r["snippet_id"]: r for r in map(json.loads, open(sys.argv[2]))}
bad = []
for sid, r in new.items():
    o = old.get(sid)
    if o is None:
        bad.append((sid, "absent from banked")); continue
    for k in ("noop_greedy_correct", "noop_greedy_parsed"):
        if bool(r[k]) != bool(o[k]):
            bad.append((sid, f"{k}: {r[k]} vs banked {o[k]}"))
print(f"[A8] reproduction gate: {len(new)} items checked, {len(bad)} mismatches")
for b in bad:
    print("   ", b)
if bad:
    print("[A8] A8-HARNESS-FAULT: greedy is deterministic at fixed seed, so a mismatch means the budget flag "
          "changed something it must not have. Nothing is read."); raise SystemExit(3)
PY

echo; echo "=== STAGE B: noop + c3_band @ 2600, all 60 items, greedy only ==="
python nla/src/nla_accuracy.py --arms noop,c3_band --max-new-gen 2600 --no-sampled \
  --out "$OUT/b2600" --max-hours 4.5; rc=$?
echo "--- stage B rc=$rc"

echo; echo "=== STAGE C: the frozen H-A8 rules ==="
python - "$OUT/b2600/accuracy_rows.jsonl" "$BANKED" "$OUT/a8_stats.json" <<'PY'
import json, sys
import numpy as np
NEW, OLD, DEST = sys.argv[1], sys.argv[2], sys.argv[3]
SEED, N_BOOT, BAR = 20260724, 10_000, 0.05
new = {r["snippet_id"]: r for r in map(json.loads, open(NEW))}
old = {r["snippet_id"]: r for r in map(json.loads, open(OLD))}
sids = sorted(set(new) & set(old))

def boot(x, off=0):
    x = np.asarray(x, float)
    if not len(x): return {"mean": float("nan"), "ci95": [float("nan")]*2, "n": 0}
    rng = np.random.default_rng(SEED + off)
    b = np.sort(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1))
    return {"mean": float(x.mean()), "ci95": [float(b[int(.025*N_BOOT)]), float(b[int(.975*N_BOOT)])], "n": len(x)}

def mcnemar(b, c):
    from math import comb
    n = b + c
    if not n: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def pair(kn, ko, off):
    d = [float(bool(new[s][kn])) - float(bool(old[s][ko])) for s in sids]
    b = sum(1 for s in sids if bool(new[s][kn]) and not bool(old[s][ko]))
    c = sum(1 for s in sids if bool(old[s][ko]) and not bool(new[s][kn]))
    r = boot(d, off); r.update({"gained": b, "lost": c, "mcnemar_p": mcnemar(b, c)})
    return r

st = {"experiment": "A8_budget_confound", "n_items": len(sids), "seed": SEED, "bar": BAR,
      "budgets": {"new": 2600, "banked": 1100}, "summary": []}
st["H_A8a"] = pair("noop_greedy_parsed", "noop_greedy_parsed", 1)
st["verdict_A8a"] = ("BUDGET-BINDING" if st["H_A8a"]["mean"] >= BAR and st["H_A8a"]["ci95"][0] > 0
                     else "BUDGET-SATURATED")
st["H_A8b"] = pair("noop_greedy_correct", "noop_greedy_correct", 2)
for tag, src, kp, kc in (("new_2600", new, "noop_greedy_parsed", "noop_greedy_correct"),
                         ("banked_1100", old, "noop_greedy_parsed", "noop_greedy_correct")):
    p = [bool(src[s][kp]) for s in sids]
    st[f"acc_given_parsed_{tag}"] = {"parsed": float(np.mean(p)),
        "acc_given_parsed": float(np.mean([bool(src[s][kc]) for s in sids if src[s][kp]])) if any(p) else float("nan"),
        "acc": float(np.mean([bool(src[s][kc]) for s in sids]))}
# H-A8c: both arms at the SAME budget, so this one is internal to the new rows
c3 = [float(bool(new[s]["c3_band_greedy_correct"])) - float(bool(new[s]["noop_greedy_correct"])) for s in sids]
b = sum(1 for s in sids if new[s]["c3_band_greedy_correct"] and not new[s]["noop_greedy_correct"])
c = sum(1 for s in sids if new[s]["noop_greedy_correct"] and not new[s]["c3_band_greedy_correct"])
st["H_A8c"] = boot(c3, 3); st["H_A8c"].update({"fixed": b, "broke": c, "mcnemar_p": mcnemar(b, c)})
flip = [s for s in sids if old[s].get("flippable")]
if flip:
    st["H_A8c_flippable"] = {
        "n": len(flip),
        "c3_band": float(np.mean([bool(new[s]["c3_band_greedy_correct"]) for s in flip])),
        "noop": float(np.mean([bool(new[s]["noop_greedy_correct"]) for s in flip])),
        "parsed_c3_band": float(np.mean([bool(new[s]["c3_band_greedy_parsed"]) for s in flip]))}
st["H_A8c_note"] = ("NO verdict word by prereg: max detectable ~ +0.03 overall / ~ +0.10 flippable at n=60; "
                    "perfect rescue of all 7 flippable gives McNemar p = 0.092.")
a, bb = st["H_A8a"], st["H_A8b"]
st["summary"] = [
    f"H-A8a {st['verdict_A8a']}: parsed(2600) - parsed(1100) = {a['mean']:+.4f} {a['ci95']} "
    f"(bar +{BAR}) · gained {a['gained']} lost {a['lost']} McNemar p={a['mcnemar_p']:.4f}",
    f"H-A8b acc(2600) - acc(1100) = {bb['mean']:+.4f} {bb['ci95']} · fixed {bb['gained']} broke {bb['lost']} "
    f"p={bb['mcnemar_p']:.4f}",
    "parsed / acc|parsed / acc:  2600 " + " ".join(f"{st['acc_given_parsed_new_2600'][k]:.4f}" for k in
        ("parsed", "acc_given_parsed", "acc")) + "   |  1100 " + " ".join(
        f"{st['acc_given_parsed_banked_1100'][k]:.4f}" for k in ("parsed", "acc_given_parsed", "acc")),
    f"H-A8c (no verdict) c3_band - noop, both @2600 = {st['H_A8c']['mean']:+.4f} {st['H_A8c']['ci95']} · "
    f"fixed {st['H_A8c']['fixed']} broke {st['H_A8c']['broke']} p={st['H_A8c']['mcnemar_p']:.4f}"
    + (f" · flippable n={st['H_A8c_flippable']['n']} c3_band {st['H_A8c_flippable']['c3_band']:.3f} "
       f"vs noop {st['H_A8c_flippable']['noop']:.3f}" if flip else ""),
]
open(DEST, "w").write(json.dumps(st, indent=1))
print("\n".join(st["summary"]))
PY
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
