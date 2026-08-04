# DATA_SOURCES.md — provenance of the linked stimuli & behavioral tables

*Last updated: 2026-08-04.* Everything under `stimuli/` and `behavioral/` is a **symlink**
into the canonical Papers 2–3 artifacts elsewhere in the monorepo (located + verified by a
6-agent hunt, 2026-08-04; see `log/infra/`). Canonical hub for stimuli:
`model_understanding/base/data/corpus/` — `model_understanding_dualsystem/data/*` symlinks
into it and `dataset/` holds byte-identical older copies (md5-verified by the hunt); the
`* copy` dirs are duplicates. Do not edit link targets — treat them as read-only inputs.

## stimuli/
| Link | What it is |
|---|---|
| `dataset_a_source.json` | **Dataset A**: 100 rows = 20 HumanEval-X snippets (10 JS + 10 Py) × 5 tiers (L0/L1/L1b/L2/L3), incl. the `fibfib`→`smoothArea` flagship trap (task JavaScript/63). |
| `dataset_a_human_labels.csv` | Dataset A human grades: 600 graded responses over 98 snippet-tier cells (`id,netid,question_tag,response,answer_key,manual_status`). |
| `dataset_b_source.json` | **Dataset B**: 250 rows = 50 base problems × 5 tiers (10 each from humaneval-x-{py,js}, cruxeval-x-{py,js}, leetcode-2025-05). Supersets exist (`tasks_unified_500.json` = 2,500 rows) if more items are ever needed. |
| `alignment/humaneval_x_{python,js}_L1b_mapping.parquet` | Per-problem `L1b_mapping_var` dicts (original → adversarial identifier). ⚠️ **Verified 2026-08-04: these mappings are from a DIFFERENT L1b generation round than the study stimuli** (A: `fibfib→smoothArea`; parquet: `fibfib→rns` — ~0/10 match). The converter (`src/convert_stimuli.py`) therefore derives mappings **cross-tier from the stimuli themselves** (parallel identifier-token walk L0↔L1b etc.); these parquets are kept for reference only. NB: **SFR embeddings are NOT on this host** (see `model_understanding_dualsystem/stats/outputs/MISSING_DATA.md`); distance measures exist only as proxies in `behavioral/paper2_adv_features.parquet`. |

## behavioral/
| Link | What it is |
|---|---|
| `paper2_trials.parquet` | **Paper 2 primary join target**: per-run trial table, 31,711 rows × 37 cols; filter `is_core == 1` → the 29,546 analysed runs of the 28,750-run design. Keys: `snippet_id`, `model`, `tier`, `correct`, `dispatcher_present`, … |
| `paper2_adv_features.parquet` | Per-snippet adversarial features (71 rows): `ISF`, `spike_regime`, semantic-displacement proxies (`D_L0_L1b_proxy`, `delta_misframe_proxy`), `distance_band`. |
| `paper2_dispatcher_cf.csv` | Per-(snippet, tier) control-flow keyword count `cf_count` (250 rows) — the dispatcher-complexity proxy behind r = −0.196. |
| `paper3_human_graded.csv` | **Paper 3 human join target**: consensus-graded per-response rows (426; n=73 study, 71 retained) — `condition`, `system` (S1/S2), `level` (L0/L1b/L2), `correct`. |
| `paper3_model_results.xlsx` | Paper 3 LLM companion: 1,584 rows = 8 models × conditions (`instructed_mode` System1/System2, token-limit arms) × tiers. |

## Gotchas (verified by the hunt — do not rediscover these the hard way)
- **HCI is not a stored column.** Derive as `hci = (correct == 0) & (confidence_z > 0)` with
  `confidence_z` z-scored within model × prompt_condition — the reference implementation is
  `model_understanding_dualsystem/stats/R/06_rq45_adversarial.R`.
- Raw 51-col source of `paper2_trials.parquet`: `model_understanding_dualsystem/logprob/unified_metrics.csv`
  (has `identifier_spike_fraction`, `atom_spike_rate` per run). Clean release mirrors:
  `model_understanding_release/results/`.
- Obfuscation pipeline (to regenerate/extend tiers): `allocation_replication/pipeline/obfuscation/`
  (`rename.py`, `flatten.py`, …).

## Converted outputs (2026-08-04 — `src/convert_stimuli.py`)
`stimuli/dataset_a/dataset_a.jsonl` (100 rows) and `stimuli/dataset_b/dataset_b.jsonl`
(250 rows) in the `src/data.py` Snippet schema: `identifier_spans` classified per span
(orig / adversarial / fn_* / l1_neutral / self_derived) via cross-tier derivation,
`dispatcher_spans` (Python `while <var>` state machines + JS object-dispatch tables; 1/350
undetected), and `meta.rename_map` = the decoy↔true pairing (100% positional pairing on all
Dataset A L1b rows; fallbacks only on some L1/L3). Span→token resolution validated at
**1.0000** on both Llama-3.1-8B-Instruct and Qwen3-0.6B tokenizers. Full stats:
`stimuli/conversion_report.json`.
