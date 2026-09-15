#!/bin/bash
#SBATCH --job-name=equiv_acts
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_equiv_acts.out
# Re-extract 4B activations for the FSDP equivalence gate.
#
# Runs on h100 (QoS=N/A -> uncapped) rather than h200 (QOS juno, MaxJobsPU=4 saturated by obtune).
# 4B bf16 inference needs ~8 GB of weights, so an 80 GB H100 is ample.
# Why this is needed at all: all 34 acts/L*.npy were deleted on 2026-09-10 to free the 1.1 TB
# quota (approved; the gate stages never read acts, only the SFT stages do). The equivalence test
# trains a real AV layer, so it needs the real activations back. Cost ~50 min for 65 GB.
#
# The root is a SCRATCH dir that symlinks the existing corpus/ and explain/ out of /work, so
# nothing is regenerated except the activations, and the test's own L{K}/av output cannot collide
# with the 34 trained pairs.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
R=/scratch/juno/jvl210002/fsdp_equiv_root
echo "# equiv acts · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
ls -l "$R"
python nla/src/nla_train.py --stage extract --root "$R" || exit 1
python - <<'PY'
import json
from pathlib import Path
import numpy as np
r = Path('/scratch/juno/jvl210002/fsdp_equiv_root/acts')
n = json.loads((r / 'norms.json').read_text())
a = np.load(r / 'L2.npy', mmap_mode='r')
print(f"acts L2 shape={a.shape} dtype={a.dtype} finite_first1k={bool(np.isfinite(a[:1000]).all())}")
print(f"injection_scale L2={n['injection_scale']['2'] if 'injection_scale' in n else n.get('2')}")
PY
echo "# done $(date -u +%FT%TZ)"
