#!/bin/bash
#SBATCH --job-name=nla_ml_join
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_join.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_ml_join.out
# Phase B step 3: merge explanation shards (CPU). Refuses to proceed if any shard is incomplete,
# so the layer array behind it never trains on a partial corpus.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python - <<'PY' || exit 1
import json, glob, pyarrow.parquet as pq
n = pq.read_table("data/nla/ml/gemma4b/corpus/rows.parquet", columns=["row_id"]).num_rows
done = sum(sum(1 for l in open(f) if l.strip()) for f in glob.glob("data/nla/ml/gemma4b/explain/shard*.jsonl"))
print(f"rows {n}  generated {done}")
assert done == n, "explanations incomplete — resubmit nla_ml_explain.sh"
PY
python nla/src/nla_train.py --stage explain --join --device cpu || exit 1
