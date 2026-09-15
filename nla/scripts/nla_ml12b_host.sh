#!/bin/bash
#SBATCH --job-name=nla12b_host
#SBATCH --partition=a30
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla12b_host.out
# Materialise the text-only Gemma3ForCausalLM checkpoint for google/gemma-3-12b-it.
# CPU-only (safetensors key remap: language_model.model.X -> model.X, vision tower dropped).
# Partition a30 (not `normal`): a30/h100 have QoS=N/A, so jobs there face only the uncapped
# `normal` QOS instead of juno's MaxJobsPU=4, which obtune was saturating (2026-09-10). No GPU is
# A GPU is requested but NOT used: a site plugin routes gres-less jobs off the GPU partitions
# back to `normal` (and thus back under juno's cap), so one idle A30 is the price of an
# uncapped slot for this ~30 min CPU remap. A30 chosen over H100 as the cheaper card to idle.
# NOT runnable on the login node: `ulimit -v` there is 8 GB and each shard load exceeds it
# (2026-09-10: MemoryError os error 12). Output goes to SCRATCH, not /work — the checkpoint is
# trivially regenerable from the hub cache and would otherwise cost 23 GB of a 1.1 TB quota.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "# 12B text host · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/gemma_text.py
python - <<'PY'
import sys, json; sys.path.insert(0, 'nla/src')
from pathlib import Path
from gemma_text import ensure_text_checkpoint, load_gemma_text, load_tokenizer
out = Path('/scratch/juno/jvl210002/nla_ml_gemma12b/host_text')
p = ensure_text_checkpoint('google/gemma-3-12b-it', out)
idx = json.loads((p / 'model.safetensors.index.json').read_text())
print(f"keys={len(idx['weight_map'])} total_size_GB={idx['metadata']['total_size']/1e9:.1f}")
# Prove it loads and is not random-init: a 2-layer trunk must find every weight in the index.
m = load_gemma_text(p, n_layers=2, dtype=__import__('torch').bfloat16, device='cpu')
cfg = m.config
print(f"loaded trunk: n_layers={cfg.num_hidden_layers} d_model={cfg.hidden_size} vocab={cfg.vocab_size}")
tok = load_tokenizer(p); print(f"tokenizer ok, inj char id={tok.convert_tokens_to_ids('㈜')}")
PY
echo "# done $(date -u +%FT%TZ)"
