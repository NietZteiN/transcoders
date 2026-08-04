# setup — project scaffolding, framing, foundational literature

*Last updated: 2026-07-24*
**Status:** done (ongoing)

Meta-thread for getting the `transcoders/` sub-project ready: registering the foundational papers, organizing the goal artifacts, and standing up the charter + research-ops scaffold.

## Hypotheses — open
- (none — this thread is organizational, not experimental)

## Hypotheses — resolved
- (none)

## What worked
- Three foundational papers registered and made citable by BibTeX key (`papers/references.bib`, `papers/REFERENCES.md`).
- Charter (`../../CLAUDE.md`) stood up from the `translation/` template: real compute layer (no SLURM; 4× A6000), a project-specific §3 (Instrument 3 = SAE + transcoder analysis), and the full `log/` protocol.
- Goal artifacts organized into `../../docs/` (`experiment_menu.md` + proposal deck).
- Hypotheses/experiments/tasks formalized into `../../docs/CHECKLIST.md` (HT1–HT8 ↔ E1–E8; phased to-do).

## What didn't
- (nothing blocking yet)

## Open ideas
- Phase 0: stand up `src/` / `configs/` / `data/`, pin a conda env (`sae_lens` / `circuit-tracer` / `transformer_lens`), pull stimuli + pretrained dictionaries.
- Cheapest high-value starts: **E7** (pure analysis) and **E4** (no pretrained dictionary needed); then **E1/E2** on Llama-3.1-8B (Llama Scope).

## Entries
- `2026-07-24_hypotheses-checklist.md` — formalized HT1–HT8 + phased task list into `docs/CHECKLIST.md`.
- `2026-07-24_charter-and-scaffold.md` — wrote CLAUDE.md, built log/, organized docs/, seeded scratchpad.
- `2026-07-24_register-foundational-papers.md` — registered Papers 1–3 (moved PDFs, wrote references.bib + REFERENCES.md).

## Doc / results links
- [`../../docs/CHECKLIST.md`](../../docs/CHECKLIST.md) — hypotheses ledger (HT1–HT8), experiment tracker, phased task list.
- [`../../docs/experiment_menu.md`](../../docs/experiment_menu.md) — the SAE/transcoder experiment menu (E1–E8).
- [`../../docs/`](../../docs/) — the two-instrument proposal deck.
- [`../../papers/REFERENCES.md`](../../papers/REFERENCES.md) — foundational papers.
