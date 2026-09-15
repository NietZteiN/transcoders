#!/bin/bash
#SBATCH --job-name=nla_ml_extract
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_extract.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_extract.out
# Phase B (ML) step 1: unit tests -> END-TO-END SMOKE of every stage on 40 docs (corpus, extract,
# explain, join, sft_av L5, sft_ar L5, check L5) in a throw-away root -> full corpus + extract.
# The smoke is the only place the whole chain (incl. LocalAV/NLACritic loading our checkpoints)
# runs before 34 layer jobs are queued behind it, so it fails loudly and the pipeline stops.
# h200 only (explicit gres): the h100 partition mixes in 47 GB MIG slices.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
CFG=nla/configs/nla_ml.yaml; T=nla/src/nla_train.py; SMOKE=data/nla/ml/gemma4b_smoke

echo "# ML extract · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum $T nla/src/gemma_text.py nla/src/local_av.py $CFG
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo; echo "=== unit tests ==="
python -m pytest nla/tests/test_nla_train.py -q || exit 1

echo; echo "=== SMOKE (40 docs, all stages, root $SMOKE) ==="
python $T --stage corpus  --root $SMOKE --limit 40 --force || exit 1
python $T --stage extract --root $SMOKE --force || exit 1
python $T --stage explain --root $SMOKE --shard 0 --n-shards 1 --batch-size 32 || exit 1
python $T --stage explain --root $SMOKE --join || exit 1
python $T --stage sft_av  --root $SMOKE --layer 5 --limit 64 --max-steps 2 --micro-batch 8 --force || exit 1
python $T --stage sft_ar  --root $SMOKE --layer 5 --limit 64 --max-steps 2 --force || exit 1
python $T --stage check   --root $SMOKE --layer 5 --limit 4 || exit 1

echo; echo "=== FULL corpus + extract ==="
python $T --stage corpus  || exit 1
python $T --stage extract || exit 1
cat data/nla/ml/gemma4b/corpus/corpus_summary.json | head -20
echo "# done $(date -u +%FT%TZ)"
