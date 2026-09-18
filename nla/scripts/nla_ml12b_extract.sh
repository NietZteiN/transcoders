#!/bin/bash
#SBATCH --job-name=nla12b_extract
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla12b_extract.out
# Phase C step 1: 12B activations for all 48 layers (stage_extract hardcodes range(n_layers)).
# 3.07 GB/layer at d=3840 -> 147 GB, on SCRATCH. Only {2,7,32} are needed for the pre-registered
# comparison, but extraction cost is dominated by the forward passes over 20k docs, which is the
# same whether we keep 3 layers or 48.
#
# corpus/ and explain/ are SYMLINKS to the 4B root: the two hosts share a tokenizer (㈜ = token
# 246566 in both), so the tokenised corpus and the 174,423 explanations are reusable unchanged --
# the explanations describe the text context, not activations. Saves ~20 GPU-h of re-explaining.
#
# h100 (QoS=N/A -> uncapped) rather than h200 (juno QOS, MaxJobsPU=4 shared with obtune).
# bf16 inference: 23.5 GB of weights + ~12 GB of per-layer hook buffers fits an 80 GB card.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
CFG=nla/configs/nla_ml_12b.yaml
echo "# 12B extract · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_train.py nla/src/gemma_text.py "$CFG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python nla/src/nla_train.py --stage extract --config "$CFG" --max-hours 5.5 || exit 1
python - <<'PY'
import json
from pathlib import Path
r = Path('/scratch/juno/jvl210002/nla_ml_gemma12b/acts')
n = json.loads((r / 'norms.json')
               .read_text())
print(f"rule={n['rule']} n_rows={n['n_rows']}")
for K in (2, 7, 32):
    L = n['layers'][str(K)]
    print(f"  L{K}: mean_norm {L['mean_all']:.1f} injection_scale {L['injection_scale']}")
print("files:", len(list(r.glob('L*.npy'))))
PY
echo "# done $(date -u +%FT%TZ)"
