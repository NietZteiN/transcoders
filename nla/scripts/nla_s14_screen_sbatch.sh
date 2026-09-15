#!/bin/bash
#SBATCH --job-name=nla_s14_screen
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_s14_screen.out
# H-S14a -- screen the 257-snippet dataset_c for FLIPPABLE items (L0 correct, L1b wrong) at the
# corrected 2600-token budget. Pre-registered in log/nla-harness/2026-09-13_bigger-corpus-prereg.md.
#
# WHY 2600 AND NOT 1100: H-A8 (job 393511) showed the 1100-token budget leaves ~20 % of generations
# with no parseable answer, scored wrong. Screening at 1100 would drop those items from the flippable
# pool for running out of tokens rather than for comprehension -- the screen would silently select
# against exactly the hard items the experiment is about.
#
# --fast-host (added after job 394532 was cancelled at 15 min): that job used host_traces' default
# HF path (AutoModelForCausalLM on google/gemma-3-4b-it, i.e. the MULTIMODAL checkpoint) and measured
# 4.8 min for ONE item -- ~144 s per 2600-token generation, which is 20 h for 257 items rather than
# the 2.5 h this was costed at. --fast-host loads the banked TEXT-ONLY checkpoint through the same
# `load_gemma_text` path nla_accuracy.py uses (~17.5 s/generation). Verified before switching: the
# fast tokenizer reproduces all 120 banked prompt ids from traces.jsonl (built with AutoTokenizer)
# EXACTLY, so no token id can move -- and it is the more consistent choice anyway, since every
# accuracy number this screen feeds (H-A8 at 2600) was produced through that same loader.
#
# Prediction on record: 25-35 flippable from 257 (banked yield 7/60 = 11.7 %). At ~28 flippable a
# 3-in-7 rescue rate would give McNemar p ~ 5e-4, where the 7-item version cannot clear 0.09 even on
# a perfect rescue. JsonlSink is resumable, so re-submitting continues rather than restarting.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-S14a corpus screen · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/host_traces.py nla/src/build_dataset_c.py nla/src/steer_run.py \
          nla/src/gemma_text.py data/stimuli/dataset_c/dataset_c.jsonl
python -c "import json;m=json.load(open('data/stimuli/dataset_c/dataset_c_manifest.json'));print('[DSC] manifest:',{k:m[k] for k in ('n_snippets','n_rows','rename_gate','quarantined','skipped')})"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/ -q -p no:faulthandler -k "steer_run or task_bank or graded" 2>&1 | tail -2

echo; echo "=== SMOKE: 3 items, verify the new pool loads and grades ==="
python nla/src/host_traces.py --model gemma4b --datasets dataset_c --max-new-gen 2600 --limit 3 \
  --fast-host --out-dir /scratch/juno/jvl210002/nla_s14_smoke || exit 1

echo; echo "=== SCREEN: all 257 dataset_c snippets at 2600 tokens ==="
python nla/src/host_traces.py --model gemma4b --datasets dataset_c --max-new-gen 2600 --fast-host \
  --out-dir "$PROJ/data/nla/p0/trace_llr/gemma4b_c" --max-hours 5.0; rc=$?
echo "--- screen rc=$rc"
python -c "
import json
s=json.load(open('$PROJ/data/nla/p0/trace_llr/gemma4b_c/traces_summary.json'))
print('[S14] n',s['n'],'acc_l0',round(s['acc_l0'],4),'acc_l1b',round(s['acc_l1b'],4),
      'parsed_l0',round(s['parsed_l0'],4),'parsed_l1b',round(s['parsed_l1b'],4))
print('[S14] FLIPPABLE',s['n_flippable'],'of',s['n'],'=',round(s['n_flippable']/s['n'],4),
      '(predicted 25-35)')
" 2>/dev/null
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
