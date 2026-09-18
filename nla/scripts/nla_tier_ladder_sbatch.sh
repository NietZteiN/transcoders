#!/bin/bash
#SBATCH --job-name=nla_tier
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_tier_ladder.out
# H-A1/H-A2/H-A3 -- the PROMPT-SPACE tier ladder on the host (L0/L1/L1b/L2/L3), greedy + n=8.
# Pre-registered in log/nla-harness/2026-09-12_tier-ladder-prereg.md. No steering, no autoencoder.
# L1 is perfect decoy erasure in token space, so rate_L1 - rate_L1b bounds every erasure lever's
# accuracy gain from above; L2/L3 size the flattening route on this host before Step 4 is designed.
# Identity gate (exit 3): rebuilt L0/L1b prompts must equal the banked trace ids bit-for-bit
# (passed 120/120 on the login node on 2026-09-12 before submission).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# prompt-space tier ladder · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_tier_accuracy.py nla/src/nla_accuracy.py nla/src/steer_run.py nla/src/task_bank.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m pytest nla/tests/test_tier_accuracy.py -q || exit 1

echo; echo "=== SMOKE: 2 items, greedy only ==="
python nla/src/nla_tier_accuracy.py --limit 2 --no-sampled --max-hours 0.5 \
  --out /scratch/juno/jvl210002/tier_smoke || exit 1

echo; echo "=== FULL: 60 items x 5 tiers, greedy + 8 samples ==="
python nla/src/nla_tier_accuracy.py --n-samples 8 --max-hours 7 || exit 1
echo "# done $(date -u +%FT%TZ)"
