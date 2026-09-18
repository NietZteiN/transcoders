#!/bin/bash
#SBATCH --job-name=nla_r2_arm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G          # peak RSS measured: 15.4G (7B) / 26.7G (13B); 200G was ~7.5x over and did not fit the busy h200 nodes
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r2_%x.out
# H-R2 -- one bake-off arm on the SAME 50 programs, CodeLlama-7b-Instruct (the only model measured
# with damage worth recovering: original 0.7995 -> renamed 0.6866, +0.1129 DAMAGE-PARTIAL).
# All arms share their SteeredCausalLM runtime, so only the steering config differs.
# Pre-registered: log/nla-harness/2026-09-14_steering-bakeoff-prereg.md (+ arms amendment)
# $1 = arm name
set -uo pipefail
ARM="${1:?usage: sbatch nla_r2_arm_sbatch.sh <arm>}"
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
OUT=$A/bakeoff_codellama7b
mkdir -p "$OUT"
echo "# H-R2 arm=$ARM · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
nvidia-smi --query-gpu=index,name --format=csv,noheader
# every arm reads the RENAMED subset packs -- the condition being recovered
# residual arms also need the ORIGINAL packs (h0 capture) and the rename manifest (alignment)
EXTRA=""
case "$ARM" in swap_oracle|foreign|erasure|combined)
  EXTRA="--packs-orig $A/gate/packs_orig_subset.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0";;
# H-R6 vector arms: pre-fitted vectors from ase_vectors.py (nla/configs/ase_vectors.yaml); the runner
# refuses `ridge_map` unless the fit's pre-GPU gate passed
ridge_map|role_proto)
  sha256sum nla/src/ase_vectors.py nla/configs/ase_vectors.yaml; ls -la $A/vectors_codellama7b_L7.pt
  EXTRA="--packs-orig $A/gate/packs_orig_subset.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0 --vectors $A/vectors_codellama7b_L7.pt";;
prompt_types)
  sha256sum nla/src/ase_roles.py nla/configs/ase_vectors.yaml;;
esac
python nla/src/ase_steer_run.py --packs "$A/gate/packs_ren_subset.jsonl" --arm "$ARM" $EXTRA \
  --model-id codellama/CodeLlama-7b-Instruct-hf --out "$OUT/$ARM.jsonl" --max-hours 2.5; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
