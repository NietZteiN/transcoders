# transcoders — SAE + Transcoder Feature & Circuit Analysis

*Last updated: 2026-09-07*

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

## Results

- **[`RESULTS.md`](RESULTS.md) — master index over every result in the ledger.** Charter status,
  the readout programme, Phase-0 licensing, the retracted positives and the one survivor,
  Experiment W in full, coverage gaps, standing constants, and the open questions. Every number is
  quoted from the dated entry that produced it.
- **[The Edit Bottleneck](https://claude.ai/code/artifact/e4c53ff9-4e05-4119-bec7-c0f7b95658c7)** —
  published summary of the flagship finding: turning an activation into English and back is
  **98.3 %** causally faithful, while *editing* the English does nothing a random vector does not.
- [`log/README.md`](log/README.md) — the full append-only ledger, 120 dated entries by thread.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Version-controlled experiment/pipeline configs |
| `data/` | Large artifacts (stimuli symlinks, activation caches, dictionaries) — **not committed**; see [`data/README.md`](data/README.md) and [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) |
| `docs/` | Experiment menu, checklist, proposal deck |
| `log/` | Research ledger, one folder per thread ([`log/README.md`](log/README.md) is the index) |
| `nla/` | Instrument-2 Natural Language Autoencoder gear (see below) |
| `RESULTS.md` | Master index over all results (see above) |
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
- **2026-09-07** — Added [`RESULTS.md`](RESULTS.md) and linked the published summary artifact.
- **2026-08-04** — Initial public release of the repo scaffold.
