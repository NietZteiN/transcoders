### Target Date: 2026-09-07 (H-W28 — the identifier pairing recovers 3.05× at precision 1.000, and routing L3 through L2 is what makes JavaScript L3 recoverable at all)

**Thread:** nla-harness · **Job:** none (CPU, ~40 s on the login node) ·
**Resolves:** [`2026-09-07_pairing-recovery-prereg.md`](2026-09-07_pairing-recovery-prereg.md) ·
**Code:** `nla/src/repair_all.py` (new), on `repair_pairs.recover()` at its H-W22 defaults
(`min_votes=1`, `min_align_frac=0.5`) · **Artifacts:** `data/stimuli/pairing_recovered.jsonl`
(210 rows), `data/stimuli/pairing_recovery_report.json`.

- **Hypotheses:** H-W28a (recovery is accurate enough to trust, per cell), H-W28b (anchoring L3 on
  **L2** beats anchoring it on L0), H-W28c (coverage, descriptive).

- **Setup.** Deterministic, no seed, no model. For each of the 350 stimulus rows, each renamed tier
  is aligned against its frozen anchor — L1 and L1b on **L0**, L3 on **L2** — and scored against the
  pairs the pipeline itself recorded. Gate, frozen in the prereg: a (dataset, tier, language) cell is
  written only at **precision ≥ 0.95 on ≥ 20 comparable pairs**; a cell that cannot be scored may
  take the verdict of the same (tier, language) in the other dataset, marked `ACCEPTED-BORROWED`.
  The stimulus files are **not touched** — output is a sibling file carrying a per-key
  `source ∈ {pipeline, recovered}`, and the pipeline's real pairs always win.

- **Results.**

  | cell | gate | precision | comparable | sentinel/keys | usable pairs before → after |
  |---|---|---|---|---|---|
  | a·L1·python | ACCEPTED | **1.000** | 33 | 5/38 | 33 → 58 |
  | a·L1b·javascript | ACCEPTED | **1.000** | 49 | 0/49 | 49 → 56 |
  | a·L1b·python | ACCEPTED | **1.000** | 49 | 0/49 | 49 → 59 |
  | a·L3·python | ACCEPTED | **1.000** | 42 | 6/48 | 42 → 68 |
  | a·L3·javascript | **ACCEPTED-BORROWED** | — (0 comparable) | 0 | **92/92** | **0 → 119** |
  | b·L1·python | ACCEPTED | **1.000** | 68 | 91/159 | 68 → 278 |
  | b·L1b·javascript | ACCEPTED | **1.000** | 93 | 0/93 | 93 → 132 |
  | b·L1b·python | ACCEPTED | **1.000** | 88 | 98/186 | 88 → 278 |
  | b·L3·javascript | ACCEPTED | **1.000** | 24 | **175/199** | **24 → 282** |
  | b·L3·python | ACCEPTED | **1.000** | 88 | 102/190 | 88 → 307 |
  | a·L1·javascript | **REFUSED-UNVALIDATED** | 1.000 | **5** (< 20) | 57/62 | 5 → 5 |
  | b·L1·javascript | **REFUSED-UNVALIDATED** | — (0 comparable) | 0 | **123/123** | 0 → 0 |

  **Corpus totals: 539 pipeline pairs → 1,642 usable, a 3.05× increase**, 1,103 recovered pairs
  written across 210 rows. Precision is **1.000 in every cell that could be scored** — 539 of 539
  comparable pairs agree with the pipeline. 29 of 210 rows still hold no usable pair.

  **H-W28b — the anchor comparison, on the same comparable pairs:**

  | cell | L2-anchored | L0-anchored |
  |---|---|---|
  | a·L3·python | 1.000 (n = 42) | 1.000 (n = 30) |
  | b·L3·python | **1.000 (n = 88)** | **0.968 (n = 63)** |
  | b·L3·javascript | **1.000 (n = 24)** | **no comparable pairs recovered at all** |

- **Hypothesis verdicts.**
  - **H-W28a ✓ SUPPORTED**, and at the ceiling: 10 of 12 cells licensed, precision 1.000 wherever
    measurable. Two cells refused — both **L1·javascript**, and refused for lack of ground truth
    rather than for being wrong (dataset A's 5 comparable pairs all agree, but 5 < 20).
  - **H-W28b ✓ SUPPORTED, as predicted, and more decisively on JavaScript than the prediction
    allowed for.** On Python L3 the L2 anchor is equal-or-better (1.000 vs 1.000; **1.000 vs 0.968**).
    On JavaScript L3 it is not a precision comparison at all: anchored on L0, recovery produces
    **no checkable pair whatsoever**; anchored on L2 it produces 24 at 1.000 and unlocks **401 pairs
    across the two datasets** (119 + 282). The prediction was "L2 ≥ 0.95 and strictly greater than
    L0"; what happened is that the L0 route does not reach JavaScript L3 at all.
  - **H-W28c** — descriptive, above. The corpus-wide sentinel rate was **749/1,288 = 58.2 %** of
    `rename_map` keys; the two refused cells account for **180** of those sentinels.

- **Observations.**
  1. **This is today's L3 finding paying for itself within hours.** L3 = L1 ∘ L2 was recorded this
     morning as a correction to the charter; used as an engineering fact, it says L3 differs from L2
     by a rename only, and that single change is what makes JavaScript L3 — the worst cell in the
     corpus, 100 % and 87.9 % sentinels — recoverable. H-W22 had left L3 out precisely because the
     L0 route scored 0.833 on JavaScript.
  2. **The refused cells are the honest kind of failure.** L1·javascript is the one place the
     pipeline recorded almost nothing in *either* dataset, so there is nothing to validate against
     and the borrowing clause has nothing to borrow from. The recovery does produce 181 pairs there;
     they are computed, reported and **not written**. Writing them would have been indistinguishable,
     to a downstream consumer, from the pairs that were validated.
  3. **The bound stated in the prereg still binds and should be quoted with the 1.000.** Validation
     is only possible where the pipeline succeeded, and those are plausibly the easy cases, so
     precision 1.000 on comparable pairs is an **upper bound** on precision over the sentinel slots
     actually filled. The number to trust is "1.000 on 539 checkable pairs", not "1.000 on 1,103
     recovered ones".
  4. **Instrument 1 can now be told something actionable** rather than only warned. Its
     identifier-level attention measures inherit the 42–100 % hole; for every tier except
     L1·javascript there is now a validated sibling mapping to join against.

- **New questions / hypotheses.**
  - **H-W34:** L1·javascript is unrepairable from inside this corpus. The one remaining route is the
    obfuscation generator itself (`allocation_replication/pipeline/obfuscation/`), which did not
    travel to this host — recovering it is a data-transfer question, not a method question.
  - **H-W33** (raised this morning) is now cheap and should follow: the converter's L3 span classes
    are assigned by name *shape*, mislabelling 1,327 JavaScript L3 spans `adversarial`. With a
    validated L2→L3 mapping in hand, a span can be classified by **what the tier did to it**.
  - Re-run the W-family repaired anchoring against this file rather than the ad-hoc per-run
    `recover()` call, so every consumer shares one audited mapping.

- **Bounds.** Precision is measured on 539 pairs, all from cells where the pipeline succeeded (see
  observation 3). `ACCEPTED-BORROWED` transfers a verdict across datasets within the same tier and
  language; it is marked in the output so a consumer can exclude it, and it covers exactly one cell
  (a·L3·javascript, 119 pairs). No behavioural or GPU measurement is involved anywhere in this entry.
