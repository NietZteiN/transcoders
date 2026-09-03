#!/bin/bash
#SBATCH --job-name=patch_map
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_patch_map.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_patch_map.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
HOST=gemma12b
R="$PROJ/data/nla/p0/trace_llr/$HOST"
OUT="$PROJ/data/nla/p0/patch_map/$HOST"; mkdir -p "$OUT" "$OUT/smoke"
echo "# Experiment T — causal patching map · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/patch_map.py nla/src/span_positions.py nla/src/trace_llr.py

V=$(python -c "import json;print(json.load(open('$R/llr_stats.json'))['verdict'])") || exit 1
echo "# R verdict: $V"
if [[ "$V" == "R-UNINF" ]]; then echo "REFUSED: R-UNINF — M is not a sensitive readout; T is void by the frozen rule"; exit 2; fi

# smoke: 3 items, layers 0 and 32. The pre-registered sanity cell (all @ L0) must recover
# >= 0.8 of the clean-corrupt gap or the alignment is broken and nothing else is read.
echo; echo "=== SMOKE ==="
python nla/src/patch_map.py --model "$HOST" --traces "$R/traces.jsonl" --llr-rows "$R/llr_rows.jsonl" \
  --out-dir "$OUT/smoke" --smoke --max-hours 0.5 || exit 1
python - "$OUT/smoke/patch_stats.json" <<'PY' || exit 1
import json, sys
s = json.load(open(sys.argv[1]))
print("smoke sanity all@L0 rec =", s["sanity_all_L0"], "grid:", {k: round(v["rec"], 3) for k, v in s["grid"].items() if v["rec"] is not None})
assert s["sanity_all_L0"] is not None and s["sanity_all_L0"] >= 0.8, "all@L0 does not recover the clean trace -> alignment broken"
print("SMOKE OK")
PY

echo; echo "=== FULL (60 items x 13 layers x 5 classes) ==="
python nla/src/patch_map.py --model "$HOST" --traces "$R/traces.jsonl" --llr-rows "$R/llr_rows.jsonl" \
  --out-dir "$OUT" --max-hours 5 || exit 1
python - "$OUT/patch_stats.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"sanity all@L0 {s['sanity_all_L0']:.3f} · VERDICT {s['verdict']} · argmax {s['argmax']} rec {s['argmax_rec']}")
cls = s["classes"]; print(f"{'layer':>6}" + "".join(f"{c:>10}" for c in cls))
for l in s["layers"]:
    print(f"{l:>6}" + "".join(f"{(s['grid'].get(f'L{l}|{c}') or {}).get('rec') or float('nan'):>10.3f}" for c in cls))
PY
echo "# finished $(date -u +%FT%TZ)"
