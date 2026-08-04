# Experiment log — master index

*Last updated: 2026-08-04*

The continuous research ledger for the `transcoders/` sub-project (protocol in [`../CLAUDE.md`](../CLAUDE.md) §6). Entries are organized **by thread and by date**: `log/<thread>/YYYY-MM-DD_<slug>.md`. Copy new entries from [`TEMPLATE.md`](TEMPLATE.md). **Append-only** — never edit a past entry; corrections go in a new dated entry that references the old one.

On every new entry, update **both** tables below and the thread's own `README.md`.

## By thread

| Thread | Status | Entries | Purpose |
|--------|--------|---------|---------|
| [`setup/`](setup/) | done (ongoing) | 3 | Foundational-paper registration, goal-artifact organization, charter + scaffold, hypotheses/task checklist. |
| [`infra/`](infra/) | active | 3 | Pipeline code, conda env, configs, seed/GPU/provenance discipline. **E1 path live end-to-end 2026-08-04 (converter + SAE loader + real-model smoke); Q-norm open.** |

<!-- Planned threads (create the folder + seed its README when the first entry lands):
     sae-features (E1/E6), steering (E2), attribution-graphs (E3), state-binding (E4), triangulation (E7) -->

## Timeline

| Date | Thread | Entry | Headline |
|------|--------|-------|----------|
| 2026-07-24 | setup | [`2026-07-24_register-foundational-papers`](setup/2026-07-24_register-foundational-papers.md) | Registered Papers 1–3 (moved PDFs → `papers/`, wrote `references.bib` + `REFERENCES.md`). |
| 2026-07-24 | setup | [`2026-07-24_charter-and-scaffold`](setup/2026-07-24_charter-and-scaffold.md) | Wrote `CLAUDE.md` (Instrument 3 = SAE + transcoder analysis); built `log/`; organized `docs/`; seeded scratchpad. |
| 2026-07-24 | setup | [`2026-07-24_hypotheses-checklist`](setup/2026-07-24_hypotheses-checklist.md) | Formalized HT1–HT8 ↔ E1–E8 + phased task list into `docs/CHECKLIST.md`. |
| 2026-08-03 | infra | [`2026-08-03_pipeline-scaffold-smoke`](infra/2026-08-03_pipeline-scaffold-smoke.md) | Phase-0 scaffold built (`configs/`+`src/`+`environment.yml`); smoke + CWD-independence pass; 34-agent adversarial review → 28 findings fixed (incl. a flagship-corrupting token-alignment bug → char-span schema). |
| 2026-08-04 | infra | [`2026-08-04_env-dictionaries-data`](infra/2026-08-04_env-dictionaries-data.md) | Env `transcoders-mi` built + frozen (circuit-tracer downgrades documented); **all 14 dictionaries + 12 model ids pinned** with HfApi-verified shas; Dataset A/B + Papers 2–3 behavioral tables located + symlinked (`data/DATA_SOURCES.md`). |
| 2026-08-04 | infra | [`2026-08-04_e1-path-live`](infra/2026-08-04_e1-path-live.md) | **E1 path live end-to-end:** stimuli converter (cross-tier spans; decoy↔true pairing 100% on A-L1b; resolution 1.0000), SAE loaders, real Llama-3.1-8B smoke on GPU. First transfer numbers **FVU 0.506 / cos 0.773 / L0 28** → **Q-norm** opened (scale mismatch vs real transfer gap). |

## Changelog
- **2026-08-04b** — Added the `e1-path-live` entry; infra now 3 entries. Everything between raw stimuli and SAE features now runs on the real model; the one open question before the full E1 sweep is Q-norm. Discovered mid-run that the alignment parquets are from a different L1b generation round (documented in `DATA_SOURCES.md`); `/`-in-snippet-id filename bug fixed.
- **2026-08-04** — Added the `env-dictionaries-data` entry; infra now 2 entries. Phase-0 blockers cleared: env READY, registries fully pinned (zero `repo: null` left), data linked. Remaining before E1: stimuli JSONL converter, `apply_dictionary`, 3 gated-license acceptances.
- **2026-08-03** — **infra thread activated** (1 entry): pipeline scaffold + smoke + adversarial review. Span-resolution smoke green (mean 1.000); dictionary repo ids + conda env creation remain open.
- **2026-07-24** — Log initialized with the `setup` thread. Migrated the flat `LOG.md` into `setup/2026-07-24_register-foundational-papers.md`; added `setup/2026-07-24_charter-and-scaffold.md` (charter + scaffold) and `setup/2026-07-24_hypotheses-checklist.md` (hypotheses/task checklist). setup now 3 entries.
