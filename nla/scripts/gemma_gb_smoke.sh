#!/bin/bash
# 2 items x 1 draw on every tier — proves build_call survives Gemma's tokenizer and that the
# generation path works, before committing five 10-hour array jobs. The project rule is to smoke
# at micro scale first; the failure this guards against is real (a malformed call once produced
# clean-code accuracy 1/16 while looking like a model result).
#SBATCH --job-name=gemma_smoke
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_gemma_smoke.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%j_gemma_smoke.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "# gemma G-B smoke · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
for T in L0 L1 L1b L2 L3; do
  echo "===== $T ====="
  python nla/src/p1b_ladder.py --host gemma12b --tier "$T" --draws 1 --limit 2 \
    --out-dir "$PROJ/data/nla/p0/p1b/ladder_gemma12b_smoke"
done
