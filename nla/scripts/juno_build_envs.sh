#!/usr/bin/env bash
# Rebuild the two conda environments on juno-l-01.
#
# THEY ARE NOT INTERCHANGEABLE AND MUST NOT BE MERGED. The attention-steering backend imports
# transformers-4.x decoder internals (Qwen2Attention, apply_rotary_pos_emb, repeat_kv) that do not
# exist in 5.x, while the NLA side needs 5.12.1 for the G0 replication to reproduce byte-for-byte.
# This is why B5 moved the NLA write hook INTO the codesteer env rather than the other way around,
# and why nla/src/steer.py imports only torch and stdlib. Keep it that way.
#
# Versions are pinned to what actually ran on csr-94608 (see nla/continuation/02_ENVIRONMENT.md),
# NOT to artifact/PromptSteering/requirements.txt, which still lists torch 2.1.0 / transformers
# 4.51.3 — an older pairing than the one the P0.3 cells were produced under. Reproducing the
# banked cells matters more than matching a stale requirements file.
#
# Hardware note: juno's GPUs are H200 (sm_90), not the A6000s (sm_86) these pins were chosen on.
# Both torch builds below ship sm_90 kernels, so the pins carry over; CUDA wheels are cu12x.
set -euo pipefail
cd "$(dirname "$0")/../.."
source nla/scripts/juno_env.sh
load_conda

log() { echo "[env $(date -Is)] $*"; }

# ── codesteer — attention steering, all of P0.3 ────────────────────────────────────────
if [ ! -x "$CS_ENV/bin/python" ]; then
  log "creating codesteer at $CS_ENV"
  conda create -y -p "$CS_ENV" python=3.11 pip
  "$CS_ENV/bin/pip" install --no-input torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
  "$CS_ENV/bin/pip" install --no-input \
      "transformers==4.57.1" "accelerate==1.13.0" "huggingface-hub==0.36.2" \
      "javalang==0.13.0" "numpy==1.26.4" "safetensors" "pyyaml" "tqdm" "pytest"
  log "codesteer done"
else
  log "codesteer already present — skipping"
fi

# ── nla-mi — everything NLA: the AV/AR checkpoints, steer_run, layer_rotation ───────────
if [ ! -x "$NLA_ENV/bin/python" ]; then
  log "creating nla-mi at $NLA_ENV"
  conda create -y -p "$NLA_ENV" python=3.11 pip
  "$NLA_ENV/bin/pip" install --no-input torch --index-url https://download.pytorch.org/whl/cu128
  "$NLA_ENV/bin/pip" install --no-input \
      "transformers==5.12.1" "safetensors" "accelerate" "numpy" "scipy" "pandas" "pyarrow" \
      "pyyaml" "scikit-learn" "matplotlib" "tabulate" "httpx" "orjson" "datasets" "pytest" "ninja"
  log "nla-mi done"
  # ninja is here deliberately: the SGLang server path needs it for the flashinfer JIT. It was
  # installed ad hoc on the old box and never made it into nla/environment.yml — the known gap
  # recorded in nla/continuation/02_ENVIRONMENT.md. sglang itself is NOT installed: nothing in the
  # Phase-0 path uses the rollout server, and it pins its own torch build that would fight this one.
else
  log "nla-mi already present — skipping"
fi

log "verifying"
"$CS_ENV/bin/python"  -c "import torch,transformers;print('codesteer torch',torch.__version__,'tf',transformers.__version__)" || true
"$NLA_ENV/bin/python" -c "import torch,transformers;print('nla-mi    torch',torch.__version__,'tf',transformers.__version__)" || true
log "ALL DONE"
