#!/bin/bash
#SBATCH --job-name=r2_original
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r2_%x.out
# H-R2c denominator: the ORIGINAL (un-renamed) subset packs through THEIR runtime, unsteered.
# gate/original.jsonl (H-R1) was produced by our chat-templated runner (ase_tf_run.py) and is not
# comparable to the bake-off arms (raw prompt, no chat template): see
# log/nla-harness/2026-09-14_bakeoff-runtime-amendment.md
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
OUT=$A/bakeoff_codellama7b
echo "# H-R2 original-packs unsteered · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
python nla/src/ase_steer_run.py --packs "$A/gate/packs_orig_subset.jsonl" --arm unsteered \
  --model-id codellama/CodeLlama-7b-Instruct-hf --out "$OUT/original_unsteered.jsonl" --max-hours 2.5; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
