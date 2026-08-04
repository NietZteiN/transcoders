# transcoders — SAE + Transcoder Feature & Circuit Analysis

*Last updated: 2026-08-04*

**Instrument 3** of the mechanistic follow-up study *"Opening the Black Box of
Obfuscated-Code Comprehension."* Papers 1–3 established *behaviorally* how code
obfuscation breaks program comprehension in humans and LLMs; this instrument uses
sparse autoencoders (SAEs), transcoders, and attribution graphs to explain the
*internal mechanism* behind the two documented failure routes:

- **Atom-level interference** — adversarial identifier renaming (tier **L1b**), a
  Stroop-like trap that injects plausible-but-wrong semantics.
- **Relational overload** — control-flow flattening (tiers **L2/L3**), dispatcher
  indirection that forces hidden-state simulation.

Sibling instruments: attention reallocation (Instrument 1) and Natural Language
Autoencoders (Instrument 2, gear in [`nla/`](nla/)). All three map onto Schulte's
Block Model. The experiment menu (E1–E8) lives in
[`docs/experiment_menu.md`](docs/experiment_menu.md); the hypotheses ledger and task
tracker in [`docs/CHECKLIST.md`](docs/CHECKLIST.md).

## Layout

| Path | Contents |
|---|---|
| `configs/` | Version-controlled experiment/pipeline configs |
| `data/` | Large artifacts (stimuli symlinks, activation caches, dictionaries) — **not committed**; see [`data/README.md`](data/README.md) and [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) |
| `docs/` | Experiment menu, checklist, proposal deck |
| `log/` | Research ledger, one folder per thread ([`log/README.md`](log/README.md) is the index) |
| `nla/` | Instrument-2 Natural Language Autoencoder gear (see below) |
| `papers/` | Foundational references — [`REFERENCES.md`](papers/REFERENCES.md) + [`references.bib`](papers/references.bib) (PDFs not committed) |
| `scripts/` | Entry-point scripts |
| `src/` | Library code |

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
conda env create -f environment.yml   # exact pins in environment.lock.txt
```

## What is not in the repo

- **`nla/data/checkpoints/`** (~25 GB): the released verbalizer/reconstructor pair
  [`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar) and
  [`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) —
  download from Hugging Face into `nla/data/checkpoints/{ar,av}/`.
- **`data/` contents**: activation caches and dictionaries are regenerable but
  expensive; provenance for every source is in [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).
- **`nla/vendor/nla-repo/`** is a git submodule of the upstream
  [natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
  repo — run `git submodule update --init` if you cloned without `--recurse-submodules`.

## Changelog
- **2026-08-04** — Initial public release of the repo scaffold.
