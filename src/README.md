# src/ — pipeline code (Phase-0 skeleton)

Import-discipline: **nothing imports torch at module scope** — pin the GPU first
(`src/gpu.py`), then heavy imports happen inside functions ([`../CLAUDE.md`](../CLAUDE.md) §1).

| Module | Role |
|---|---|
| `seedutil.py` | `set_seed()` — Python/NumPy/torch/CUDA + deterministic kernels; returns the seed for the manifest. |
| `gpu.py` | `pick_free_gpus(n)` / `pin(ids)` — idle-GPU selection from `nvidia-smi`; refuses to guess when busy. Thresholds are passed in by entrypoints from `configs/compute.yaml` (gpu.py itself stays config-free so it can run first). `python -m src.gpu` prints status. |
| `provenance.py` | `RunManifest` — seed, command, config, script sha256 (root-anchored; missing script = error), GPU ids, dictionary identity, requested + **resolved** model revision, library versions → `run_manifest.json` per run ([`../CLAUDE.md`](../CLAUDE.md) §4). |
| `configs.py` | `PROJECT_ROOT` anchor; YAML loader: `extends` deep-merge + `model`/`dictionary` registry resolution; `resolve_layers()` — **0-indexed block indices, resid_post convention** (`hidden_states[l+1]`), out-of-range raises. |
| `data.py` | `Snippet` schema — identifier/dispatcher **character spans**, not token indices (tokenizer-portable); JSONL loaders for Dataset A/B (empty result raises); `toy_stimuli()` with real spans on the `fibfib`/`smoothArea` L0/L1b pair. |
| `extract_activations.py` | The shared front-end for E1/E2/E6/E7: residual-stream capture → `PROJECT_ROOT/data/activations/<run_id>/` (CWD-independent, collision-proof run dirs). Spans resolved to tokens per model via fast-tokenizer offsets, verified by decode-overlap, **hard-failing** below `positions.min_resolution_rate`; keys are `snippet__language__tier` with a duplicate guard; manifests written incrementally (crash still leaves provenance); safetensors only, no silent fallback. `--smoke` exercises the span path on a tiny model; `--gpu auto` picks + pins an idle A6000 before torch loads. `apply_dictionary()` is **stubbed** until the SAE repo ids in `configs/dictionaries.yaml` are confirmed. |

Quick start:
```bash
bash scripts/smoke.sh                       # plumbing check (tiny model, CPU)
python -m src.gpu                           # GPU status
python -m src.extract_activations --config configs/experiments/e1_semantic_capture.yaml --gpu auto
```
