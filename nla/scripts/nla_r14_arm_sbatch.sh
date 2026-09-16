#!/bin/bash
#SBATCH --job-name=nla_r14_arm
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r14_%x.out
# H-R14 -- one bake-off arm on the FULL aligned HumanEval-X corpus: 148 snippets / 1,742 cases (4.0x the
# 50/434 of H-R7), CodeLlama-7b-Instruct, THEIR SteeredCausalLM runtime with the chat template applied.
# Pre-registered: log/nla-harness/2026-09-16_full-corpus-prereg.md
#
# $1 = arm name   $2 = optional shard index 0..3
#
# SHARDING IS RESTRICTED TO THE CODESTEER ARMS ON PURPOSE. They record attention and ran 2h20 on 50
# snippets (~2.8 min/snippet), so 148 would overrun the wall; they carry no cross-snippet state, so
# disjoint snippet shards concatenate exactly. `erasure` must NEVER be sharded: its vector is a
# leave-one-ITEM-out mean over the other snippets PASSED TO THE RUNNER, so a shard would silently
# redefine the arm from LOO-over-147 to LOO-over-36. The script refuses that combination.
set -uo pipefail
ARM="${1:?usage: sbatch nla_r14_arm_sbatch.sh <arm> [shard]}"
SHARD="${2:-}"
case "$ARM" in
  erasure|swap_oracle|ridge_map|role_proto|foreign|combined)
    [ -n "$SHARD" ] && { echo "REFUSED: $ARM depends on the snippet set passed in (LOO / guard); do not shard it"; exit 2; };;
esac
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
A=/scratch/juno/jvl210002/ase2026
F=$A/full
OUT=$F/bakeoff_full
mkdir -p "$OUT"
SUF=""; [ -n "$SHARD" ] && SUF=".shard$SHARD"
echo "# H-R14 (full corpus) arm=$ARM shard=${SHARD:-none} · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_steer_run.py
sha256sum $F/snippets_148.json
nvidia-smi --query-gpu=index,name --format=csv,noheader

# every arm reads the RENAMED packs (the condition being recovered); `original_unsteered` reads the
# ORIGINAL packs through the same unsteered code path -- it is the damage denominator for H-R14a/e.
if [ -n "$SHARD" ]; then PACKS="$F/packs_ren_full.shard$SHARD.jsonl"; else PACKS="$F/packs_ren_full.jsonl"; fi
RUNARM="$ARM"
if [ "$ARM" = original_unsteered ]; then
  RUNARM=unsteered
  if [ -n "$SHARD" ]; then PACKS="$F/packs_orig_full.shard$SHARD.jsonl"; else PACKS="$F/packs_orig_full.jsonl"; fi
fi

# residual arms need the ORIGINAL packs (h0 capture) and the rename manifest (alignment). packs-orig is
# always the FULL file: prepare_residual indexes it by snippet id, so a shard would only hide snippets.
EXTRA=""
case "$ARM" in
  swap_oracle|foreign|erasure|combined)
    EXTRA="--packs-orig $F/packs_orig_full.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0";;
  ridge_map|role_proto)
    # H-R14 vectors: NESTED GROUPED CROSS-FIT (nla/configs/ase_vectors_full.yaml). The runner refuses
    # `ridge_map` unless the fit's pre-GPU gate passed -- here that gate is conjunctive across folds.
    sha256sum nla/src/ase_vectors.py nla/configs/ase_vectors_full.yaml
    ls -la $F/vectors_codellama7b_L7_full.pt
    python -c "
import json,sys; r=json.load(open('$F/vectors_codellama7b_L7_full.fit.json'))
print('[FIT]', r['mode'], r['verdict'], r['ridge_gate'], 'margin_min', round(r['ridge_map']['margin_min'],4))
sys.exit(0 if r['ridge_map']['gate_pass'] and r['verdict']=='OK' else 3)" || { echo "# FATAL: cross-fit gate did not pass"; exit 3; }
    EXTRA="--packs-orig $F/packs_orig_full.jsonl --manifest $A/rename_manifest.jsonl --layer 7 --beta 1.0 --vectors $F/vectors_codellama7b_L7_full.pt";;
esac
VCFG=nla/configs/ase_vectors_full.yaml

python nla/src/ase_steer_run.py --packs "$PACKS" --arm "$RUNARM" $EXTRA \
  --model-id codellama/CodeLlama-7b-Instruct-hf --chat-template --vectors-config "$VCFG" \
  --out "$OUT/$ARM$SUF.jsonl" --max-hours 7.0; rc=$?
echo "# rows written: $(wc -l < "$OUT/$ARM$SUF.jsonl" 2>/dev/null || echo 0)"
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
