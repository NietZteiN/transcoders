#!/bin/bash
# The Instrument-3 dense-probe baseline on Llama-3.1-8B, same ladder, same stimuli.
#
# Queued to fill GPUs the Gemma run is not using. Independent of it: different host, different
# output tree, no NLA involved. Llama-3.1-8B is the only panel model with both pretrained SAEs
# and transcoders, so this is the baseline E1/E2 have to clear — and it independently tests
# whether "the read site ties a token count" is a Qwen fact or a general one.
#
# h200 had 3 free GPUs of 52 while h100 had 8 of 9, and the original array was pinned to h200 —
# so three of five tiers sat queued behind a full partition. Gemma-3-12B is ~24 GB in bf16 and
# fits an 80 GB H100 with room to spare, so listing both partitions roughly triples the GPUs
# this work can land on. Each tier is additionally split into two disjoint draw-shards, so the
# three remaining tiers occupy six GPUs instead of three.
#
# Shards write draws_dN.jsonl; nla/scripts/merge_ladder_shards.sh folds them into draws.jsonl.
#SBATCH --job-name=llama_gb_s
# Llama-3.1-8B is ~16 GB in bf16, so unlike Gemma-12B (~24 GB) it also fits an A30 24 GB card.
# Listing a30 as well adds three more GPUs this work can land on.
#SBATCH --partition=h200,h100,a30
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_llama_gb_s.out
#SBATCH --error=/work/jvl210002/migration/transcoders/log/slurm/%A_%a_llama_gb_s.out
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda
activate_env "$NLA_ENV"
cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TIERS=(L0 L0 L1 L1 L1b L1b L2 L2 L3 L3)
STARTS=(0  5   0  5   0   5    0  5   0  5)
T=${TIERS[$SLURM_ARRAY_TASK_ID]}
S=${STARTS[$SLURM_ARRAY_TASK_ID]}

echo "# gemma G-B shard · tier $T draws $S..$((S+4)) · job $SLURM_JOB_ID on $SLURMD_NODENAME ($SLURM_JOB_PARTITION)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python nla/src/p1b_ladder.py --host llama8b --tier "$T" --draws 5 --draw-start "$S"
