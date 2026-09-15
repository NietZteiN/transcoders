#!/bin/bash
#SBATCH --job-name=nla_bsweep
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_bsweep.out
# H-S5..H-S9: can multi-layer NLA steering be made to work? Pre-registered in
# log/nla-harness/2026-09-12_beta-layerset-prereg.md; thresholds in nla/configs/nla_beta_sweep.yaml.
#
# Two things this measures that H-S1 could not:
#   (1) the SPECIFIC component of the edit (`edit - foreign`) at more than one layer -- the layer
#       sweep ran ARMS=(c3,swap,edit,random), with no `foreign` and no `rt`, so the content-specific
#       part of a multi-layer write has never been measured. Raw `edit` rewards non-specific decoy
#       destruction (a RANDOM vector scores +16.8/+18.9 at L2/L3), so raw nats are the wrong target.
#   (2) beta, the interpolation coefficient PositionReplacer deliberately lacked. Full replacement at
#       k layers overwrites the span position's trajectory k times: from k=1 to k=33 the CEILING arm
#       `swap` loses MORE (-14.50) than the NLA arm `c3` (-11.70). That is host damage, and beta is
#       what lets every layer nudge the position while the position keeps computing.
#
# Readout is G_sum, NOT accuracy: 2 of 60 items lost to obfuscation, 7 flippable. The accuracy stage
# is GATED on the nats result and is pre-declared unable to support a conclusion either way (H-S9).
#
# Host only -- no AV/AR is loaded -- so an 80 GB card is ample and h100 is eligible alongside h200
# (measured 2026-09-06: median queue wait 110 min vs median 8 min runtime, so widening partitions is
# the cheapest thing that touches wall time). --gres=gpu:1 is generic for exactly that reason.
#
# EXIT 3 = an identity gate failed (beta=1 drifted from the banked rows, beta=0 was not a value
# no-op, or SELF moved); results are NOT reportable in that case and the full run is not scored.
# ~27 k forwards at the measured 37 ms => ~17 min of compute, dominated by model load.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-S5..H-S9 beta x layer-set sweep · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/nla_beta_sweep.py nla/src/nla_ml_gate.py nla/src/steer.py \
          nla/configs/nla_beta_sweep.yaml nla/configs/nla_ml_gate.yaml
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# --deterministic stays OFF (2026-08-29: it changes answers and forks the corpus).

# Explicit file lists, never a keyword filter: job 380178 died because a broadened -k matched tests
# needing an LM fixture absent on this host. test_beta_sweep covers the frozen verdict table on
# synthetic rows; the replacer selector covers beta=1 bit-identity and beta=0 position counting.
echo; echo "=== unit: verdict table + the beta contract ==="
python -m pytest nla/tests/test_beta_sweep.py nla/tests/test_arm_guard.py -q || exit 1
python -m pytest nla/tests/test_steer.py -q -k "replacer or no_grad" || exit 1

echo; echo "=== SMOKE (6 items, 2 betas, 2 sets; identity gates must pass) ==="
python nla/src/nla_beta_sweep.py --smoke --limit 6 --max-hours 0.5 \
  --out /scratch/juno/jvl210002/bsweep_smoke || exit 1

echo; echo "=== FULL (60 items, 6 betas, full ladder + 7 named sets) ==="
python nla/src/nla_beta_sweep.py --max-hours 4 || exit 1
echo "# done $(date -u +%FT%TZ)"
