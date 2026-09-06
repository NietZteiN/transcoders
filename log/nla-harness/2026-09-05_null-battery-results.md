### Target Date: 2026-09-05 (Null battery — the effect is item-specific but NOT span-specific: representation and causation come apart)

**Thread:** nla-harness · **Job:** 378019 (h200, **1:06:52**, COMPLETED, 0 errors, 49/49 items) ·
**Host:** `gemma12b` L32/48 · **Resolves:** [`2026-09-05_null-battery-prereg.md`](2026-09-05_null-battery-prereg.md).

- **Hypotheses:** H-W9 (item specificity), H-W12 (span specificity), H-W6 (operator null), each as
  `P_patch − null ≥ +12.11` nats with a CI excluding 0.

- **Setup:** four arms at the **375 locatable spans**, 49 items, identical positions,
  `PositionReplacer` (direction-only, no α). No AV/AR loaded. Seeded foreign/shuffled pairings and
  random vectors from `crc32(snippet_id#span)`. Paired bootstrap, 10,000, seed 20260724.
  Artifacts `data/nla/p0/nulls/gemma12b/nulls_{rows.jsonl,stats.json}`.

- **Results.**

  | arm | G_sum | 95 % CI | % of `P_patch` | acc | veto |
  |---|---|---|---|---|---|
  | `P_patch` — this span's clean state | **+45.71** | [+40.89, +50.78] | 100 % | 0.714 | pass |
  | `N_shuffled_clean` — another span, **same item** | **+39.94** | [+35.56, +44.72] | **87.4 %** | 0.735 | pass |
  | `N_foreign_clean` — another **item's** clean span | **+15.18** | [+12.41, +18.05] | 33.2 % | 0.673 | pass |
  | `N_random` — random unit vector | **−365.59** | [−510.93, −237.13] | −800 % | 0.306 | **TRIPPED** |

  | contrast | gap | 95 % CI | clears +12.11 | items |
  |---|---|---|---|---|
  | **H-W9** `P_patch − N_foreign_clean` | **+30.53** | [+25.75, +35.79] | **✓** | **49/49** |
  | **H-W12** `P_patch − N_shuffled_clean` | **+5.77** | [+3.37, +8.43] | **✗** | 36/49 |
  | **H-W6** `P_patch − N_random` | **+411.30** | [+281.92, +556.38] | ✓ | 49/49 |

  Baseline acc 0.714 / parse 0.857. **Verdict `W9-ITEM-SPECIFIC`.**

- **What worked / hypothesis verdict.**
  - **H-W9 ✓ SUPPORTED, decisively.** A foreign item's clean activation retains only **33.2 %** of
    the matched effect, and the matched arm wins on **49 of 49 items**. **H-W7's causal-fidelity
    result may now be described as item-specific** — the caveat filed with it is discharged.
  - **H-W6 ✓ SUPPORTED, and trivially so.** Replacement with random directions is **catastrophic**:
    −365.59 nats, accuracy 0.714 → **0.306**, parse 0.857 → 0.490, the only veto trip in the family,
    positive on just 5/49 items and as low as −1,847 on one. The operator alone destroys; every
    positive effect in this family therefore comes from content.
  - **H-W12 ✗ NOT SUPPORTED at the frozen bar.** Writing a *different span's* clean state from the
    **same item** costs only **+5.77 nats** — the CI excludes 0, so the effect is real, but it is
    less than half the +12.11 bar and it wins on only 36 of 49 items. All 49 items had more than one
    locatable span, so this is not a denominator artifact.

- **Observations.**
  - **The finding: representation is span-specific, causation is only item-specific.** Stage 0
    measured the NLA round trip as sharply span-specific *representationally* — matched +0.6807
    against **+0.0525** for another span in the same snippet
    ([`2026-09-04_cycle-gate-results.md`](2026-09-04_cycle-gate-results.md)). Causally, that
    specificity almost vanishes: **87.4 %** of the effect survives swapping in a sibling span's
    clean state. The residual at each identifier position *encodes which identifier it is* — and the
    causal work those positions do for the answer is carried by information **shared across the
    item's spans**, not by the span-specific part. Two instruments, same positions, opposite
    granularity, and the disagreement is the result rather than a problem (E7's premise, arriving
    inside one instrument).
  - **This reframes what the clean-state patch is doing.** It is not restoring "what this identifier
    means". It is restoring something item-level — plausibly *"this program is the un-obfuscated
    one"* — that is present at every identifier span and mostly interchangeable between them.
  - **My stage-1 worry was based on a mismatched comparison, and this corrects it.** I flagged that
    W1's +19.47 sat on the banked `R_random @ id_spans` benchmark of +19.60 and treated that as
    evidence of generic perturbation. But that banked arm is **additive** steering at α = 0.149 — a
    small nudge — whereas the replacement operator with random content gives **−365.59**. The two
    are not comparable, and the "matches random" framing in
    [`2026-09-04_writeback-results.md`](2026-09-04_writeback-results.md) should be read with that
    caveat: it matches a *weak additive* random arm, not a like-for-like random one.
  - **Reproducibility, incidental but worth banking:** `P_patch` returns **+45.71** here and
    **+45.71** in H-W7 — a different job on a different node, same value. Teacher-forced scoring is
    reproducible to the reported precision, unlike the 0.85–0.90 greedy-generation floor this thread
    has documented since 2026-08-29. Every `G_sum` result in the W family inherits that.
  - Accuracy again barely moves for any non-destructive arm (0.673–0.735 against a 0.714 baseline)
    with only 3 of 49 items flippable. Unchanged conclusion: `G_sum` is the readout that works here.

- **New questions / new hypotheses.**
  - **H-W13:** if the causal content is item-level and interchangeable across spans, does writing
    **one** span's clean state (instead of all ~7.7) recover most of the effect? A dose curve over
    the number of patched spans would say whether this is a shared signal or a redundant one — and
    it is cheap, since it reuses the banked pool.
  - **H-W14:** is the item-level signal *"this is the clean program"* or *"this program specifically"*?
    Contrast a foreign **L0** activation (this battery, +15.18) against a foreign **L1b** one. If
    they are equal, what is being restored is program identity; if the L0 foreign beats the L1b
    foreign, there is a generic clean-code component too.
  - **H-W12 is not closed, it is bounded:** the span-specific component is real (CI excludes 0) and
    small (+5.77, 12.6 % of the effect). A design powered for a 6-nat effect could pin it.

- **Next Steps:** H-W13 first — cheapest, uses the banked pool, and directly tests the shared-vs-
  redundant reading of today's result. Pre-register before running.
