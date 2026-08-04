# data/ — large artifacts (NOT version-controlled)

Everything heavy lives here, never in `$HOME` and never mixed into `docs/`/`configs/`/`papers/`
(see [`../CLAUDE.md`](../CLAUDE.md) §2). Contents are regenerable-but-expensive (GPU-hours) —
treat deletions with the same care as results.

```
data/
  DATA_SOURCES.md         # provenance of every symlink below — read this first
  stimuli/
    dataset_a_source.json # symlink → canonical Dataset A (100 rows = 20 × 5 tiers) + human labels csv
    dataset_b_source.json # symlink → canonical Dataset B (250 rows = 50 × 5 tiers)
    alignment/            # symlinks → L1b_mapping_var parquets (orig→adversarial identifier maps)
    dataset_a/ dataset_b/ # (converter output, pending) JSONL in the src/data.py Snippet schema
  behavioral/             # symlinks → Papers 2–3 join tables (trials/adv/cf + Paper-3 human/model)
  activations/            # extraction runs: <run_id>/{*.safetensors, manifest.jsonl, run_manifest.json}
  dictionaries/           # downloaded pretrained SAEs/CLTs (if not left in HF_HOME)
  trained/                # any SAEs/CLTs we train ourselves (+ their training configs)
```

Every run directory must contain its `run_manifest.json` (seed, command, script sha256,
GPU, dictionary identity) — written automatically by `src/provenance.py`.
