### Target Date: 2026-09-07 (H-W28 — re-derive the identifier pairing for the WHOLE stimulus set, and route L3 through L2. Frozen before running.)

**Thread:** nla-harness · **Job:** none (CPU) · **Raised by:**
[`2026-09-07_rename-map-audit.md`](2026-09-07_rename-map-audit.md) (H-W28) ·
**Enabled by:** [`2026-09-07_l3-is-l1-not-l1b.md`](2026-09-07_l3-is-l1-not-l1b.md) ·
**Method from:** [`2026-09-06_repaired-rerun-prereg.md`](2026-09-06_repaired-rerun-prereg.md) /
`nla/src/repair_pairs.py` (H-W22, validated precision 1.000 / recall 1.000 on L1 and L1b).

- **Why.** `rename_map` pairs an original identifier with its obfuscated form by **equal-length
  positional matching** and emits `?unpairedN` keys when that fails. Measured now, over
  `data/stimuli/dataset_{a,b}/*.jsonl` (350 rows):

  | dataset | tier | language | sentinel keys |
  |---|---|---|---|
  | A | L1 | javascript | **57/62 = 91.9 %** |
  | A | L1 | python | 5/38 = 13.2 % |
  | A | L1b | both | **0/98 = 0.0 %** |
  | A | L3 | javascript | **92/92 = 100.0 %** |
  | A | L3 | python | 6/48 = 12.5 % |
  | B | L1 | javascript | **123/123 = 100.0 %** |
  | B | L1 | python | 91/159 = 57.2 % |
  | B | L1b | javascript / python | 0/93 / **98/186 = 52.7 %** |
  | B | L3 | javascript / python | **175/199 = 87.9 %** / 102/190 = 53.7 % |

  The W corpus was repaired for its own 60 items only. Everything else that pairs identifiers across
  tiers — **Instrument 1's identifier-level attention measures included** — still inherits this hole.

- **The one design decision, and it comes from today's L3 result.** `recover()` aligns two token
  sequences with identifiers masked, so it bridges a **rename** well and a **restructuring** badly
  (H-W22 measured precision 1.000 on L1/L1b anchored on L0, but **0.833 on JavaScript L3** anchored
  on L0, which is why L3 was left out then). L3 = **L1 ∘ L2**: it differs from L0 by *both* a rename
  and a flattening, but from **L2** by a rename **only** — L2 carries the true names with the
  flattened structure. So:

  | tier | anchor | what the alignment must bridge |
  |---|---|---|
  | L1 | L0 | rename only |
  | L1b | L0 | rename only |
  | **L3** | **L2** | **rename only** (was: rename + flattening) |

  L2 keeps L0's identifiers, so an L2→L3 pair is already an L0→L3 pair under the same key. This is
  also what `src/convert_stimuli.py:121` does — it derives L3's map by pairing `("L2","L3")` — so
  the change aligns the recovery with the converter rather than departing from it.

- **Setup.** `nla/src/repair_all.py` (new), CPU, deterministic, no seed needed —
  `repair_pairs.recover(anchor_code, tier_code, language, min_votes=1, min_align_frac=0.5)` at its
  H-W22-validated defaults. Inputs `data/stimuli/dataset_{a,b}/dataset_{a,b}.jsonl`. Output is a
  **new sibling file**, never an in-place edit: `data/stimuli/pairing_recovered.jsonl`, one record
  per (dataset, snippet_id, tier) with `pairs {original: renamed}` and a per-key
  `source ∈ {pipeline, recovered}`; the pipeline's real pairs always take precedence and are never
  overwritten — recovery fills sentinel slots only.

- **Hypotheses, with the rules frozen now.**
  - **H-W28a (accuracy).** Score recovery against the pairs the pipeline *did* record, per
    (dataset, tier, language) cell: `precision` = agreements / pairs comparable in both,
    `recall` = comparable / recorded-real. **A cell's recovered pairs are persisted only if
    precision ≥ 0.95 on ≥ 20 comparable pairs.** Below that the cell is **refused** — reported, not
    written. A cell with < 20 comparable pairs is `UNVALIDATED` and also refused, however good it
    looks: L1b-javascript has 0 sentinels and needs no repair, and L3-javascript-A has *no* recorded
    real pairs at all, so it cannot be validated on its own cell and must borrow the verdict of the
    same tier in the other dataset or stay refused.
  - **H-W28b (the L3 routing claim).** Anchoring L3 on **L2** beats anchoring it on **L0**, measured
    as precision on the same comparable pairs, in the same run. **Predicted: L2-anchored ≥ 0.95 and
    strictly greater than L0-anchored on JavaScript L3.** Both are computed; if L0 wins, the frozen
    routing above is wrong and L3 stays anchored on L0.
  - **H-W28c (coverage).** Report the sentinel share per cell before and after. No threshold —
    coverage is the deliverable, accuracy is the gate. Recording it as descriptive so a low-coverage
    but high-precision result cannot be quietly retold as a failure.

- **Rejected paths.** (1) Re-implementing positional matching — H-W22 established it recovers
  nothing, because it is very likely what the generator already tried. (2) Editing the stimulus
  files in place: they are the shared input to three instruments and a silent change to them is the
  worst possible failure mode here; a sibling file with an explicit `source` per key lets a consumer
  opt in. (3) Regenerating the tiers from the obfuscation pipeline — it did not travel to this host.

- **Bounds, stated before the numbers.** Validation can only be done where the pipeline succeeded,
  and the cells where it succeeded are plausibly the *easy* ones — so a precision measured at 1.000
  on comparable pairs is an upper bound on precision over the sentinel slots this actually fills.
  That asymmetry is not fixable from inside this corpus and must be carried into any consumer.

- **Results / verdict:** not yet run.
