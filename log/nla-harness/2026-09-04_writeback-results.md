### Target Date: 2026-09-04 (W stage 1 — the NLA's protocol matches a random vector; its channel's ceiling is twice what the edit reaches; H-W2′ measured nothing)

**Thread:** nla-harness · **Job:** 377399 (h200, g-08-13, **1:43:09**, COMPLETED, 60/60 rows, 0 errors) ·
**Host:** `gemma12b` L32/48 · **Resolves:** [`2026-09-03_nla-writeback-prereg.md`](2026-09-03_nla-writeback-prereg.md)
stage 1, under [`2026-09-04_stage2-vacuous-correction.md`](2026-09-04_stage2-vacuous-correction.md)
and [`2026-09-04_support-floor-too-low.md`](2026-09-04_support-floor-too-low.md).

- **Hypotheses:** H-W1 (the protocol steers), H-W2′ (the write reaches the answer site in the edit's
  direction), with C1 (round-trip null), C2 (foreign edit) and C3 (NLA clean-state ceiling).

- **Setup:** `nla/src/nla_writeback.py`, `steer.PositionReplacer` (norm-matched replacement, **no α**),
  four arms at **identical** positions — 1,480 token positions over 476 spans, **198 spans editable**
  (41.6 %). Banked stage-0 reads; banked clean traces; `MAX_NEW_GEN` 1100; seed 20260724;
  `--deterministic` OFF. Artifacts `data/nla/p0/writeback/gemma12b/writeback_{rows.jsonl,stats.json}`.

- **Results.** `G_sum` in nats, n = 60, percentile bootstrap. Frozen support **+12.11**; the unit
  (whole prompt swap) is **+121.13**.

  | arm | G_sum | 95 % CI | % of prompt swap | support | veto |
  |---|---|---|---|---|---|
  | **W1_edit** — the protocol | **+19.47** | [+16.51, +22.51] | **16.1 %** | ✓ | pass |
  | C1_roundtrip — unedited | +8.84 | [+7.21, +10.51] | 7.3 % | ✗ | pass |
  | C2_foreign — wrong content | +10.68 | [+9.04, +12.35] | 8.8 % | ✗ | pass |
  | **C3_ceiling** — NLA round trip of the **clean** state | **+37.91** | [+32.49, +43.47] | **31.3 %** | ✓ | pass |
  | *banked* `R_random @ id_spans` | *+19.60* | — | *16.2 %* | — | — |
  | *banked* `P_prompt` (prompting) | *+27.53* | — | *22.7 %* | — | — |

  Paired: **W1 − C1 = +10.63** [+8.07, +13.34], positive on 49/60. **W1 − C2 = +8.79**
  [+6.35, +11.41], 47/60. **C3 − W1 = +18.44** [+14.17, +23.00], 53/60.
  Dose: ρ(W1 − C1, editable spans) = **+0.402**.
  Behaviour: baseline acc 0.633 / parse 0.800; W1 acc 0.600, C1 0.617, C2 0.583, C3 0.617 — no arm
  trips the 0.05 veto. **Frozen verdict: `W-GENERIC-PERTURBATION`.**

- **What worked / hypothesis verdict.**
  - **H-W1 — clears the frozen rule and fails the pre-filed floor.** W1 = **+19.47** against the
    banked random vector at these very positions, **+19.60**. The protocol lands on the generic
    benchmark to within 0.13 nats, and below prompting (+27.53), which nothing in this programme has
    ever beaten. Per [`2026-09-04_support-floor-too-low.md`](2026-09-04_support-floor-too-low.md),
    filed at 11/60 before any control mean existed, that is written up as **generic perturbation, not
    steering**.
  - **C1 ✓ the experiment is interpretable.** The bare round trip moves +8.84, below support, so W1 is
    not a round-trip artifact and the family is not `W-UNINFORMATIVE-ROUNDTRIP`.
  - **A real content-specific component exists, and it is sub-threshold.** W1 beats the *foreign*
    edit — same operator, same positions, same edit shape, wrong content — by **+8.79** with a CI
    excluding zero, and the W1 − C1 gap **scales with the number of editable spans** (ρ = +0.402).
    So the true-term edit does carry content. It is just worth ~9 nats against a 121-nat unit.
  - **H-W2′ ✗ VOID — it measured nothing. See below.** The verdict's `registers = False` is a
    **null measurement, not evidence of non-delivery**, and the reported verdict should be read as
    resting on H-W1 and the controls alone.

- **Observations.**
  - **The headline is C3, not W1.** The NLA round trip of the *clean* state delivers **+37.91 —
    31.3 % of a full prompt swap, nearly double the edited protocol and above prompting.** Every
    prior result in this programme was consistent with "this channel cannot deliver anything";
    Phase 0 row 4 and R2 both leaned on that. **It can.** Feed the AV the clean activation, verbalize
    it, reconstruct it, write it back norm-matched at the identifier spans, and a third of the
    obfuscation cost comes back. **The channel is not the bottleneck. The edit is.**
  - So the protocol's failure is now located precisely: not the site (stage 0 showed span
    specificity), not the operator (C3 uses the identical operator), not delivery (C3 = +37.91), but
    **the edited text**. The AV describes these positions in surface terms — "a function
    signature/docstring is being presented" — and reaches the identifier mainly by quoting it, so
    substituting one identifier for another changes little of what the AR reconstructs.
  - **`c3_missing` = 101 of 476 spans (21 %)** fell back to C1's vector because the true term could
    not be located in the L0 code by word-boundary search. C3's true ceiling is therefore
    **understated** — it is a mixture of 79 % clean-state round trips and 21 % unedited L1b round
    trips, and the pure arm would be higher than +37.91.
  - Accuracy moved nowhere (0.633 → 0.600/0.617/0.583/0.617), exactly as the 6/60 flippable gate
    predicts. `G_sum` was the right readout; accuracy on this corpus remains uninformative.

- **H-W2′ is VOID, and the reason is structural (defect #6).**
  All 60 items returned `null` for every H-W2′ cosine. `‖Δh‖ < 1e-9` — **exactly zero**.
  **A forward hook on layer 32's output cannot change layer 32's output at any other position in the
  same forward pass.** It receives the complete tensor and overwrites selected rows; every other row
  is unchanged by definition. The write propagates to layers **≥ 33**, so reading `last_prompt` at
  **layer 32** is blind to it by construction.
  The 09-04 correction replaced a statistic that was **trivially 1** with one that is **trivially 0**
  — the same defect class, in the opposite direction, in the same hypothesis, and it survived a
  written correction because I reasoned about *which position* to read and never about *which layer*.
  **Delivery itself is not in doubt** — W1 moves G_sum by +19.47, which is only possible if the write
  reached the output — so what is lost is the *direction* test, not the *delivery* test.
  **Any repair must cross layers**, and then `Δedit` (an L32-space vector) is no longer commensurate
  with `Δh` at layer 47, so a cosine is the wrong statistic. **No replacement rule is proposed here**;
  H-W2′ is retired as unmeasurable in this form and a successor must be pre-registered from scratch.

- **New questions / new hypotheses.**
  - **H-W6 (already pre-registered):** random vector through `PositionReplacer` at the same spans.
    Now clearly the decisive missing arm — W1 matching *additive* random is suggestive; matching
    *replacement* random would settle it.
  - **H-W7:** the pure C3 arm, on the 79 % of spans with a locatable L0 term. If a clean-state round
    trip beats prompting outright, the NLA channel becomes the strongest intervention this programme
    has found, and the paper's framing changes from "belief-steering fails" to **"the channel works;
    natural-language edits are what it cannot carry."**
  - **H-W8:** why is C2 (+10.68) *above* C1 (+8.84)? A wrong-content edit beats no edit. If that
    survives replication it says the gain is partly in *disturbing* the decoy, independent of what
    replaces it — the same shape as R2's random-noise result.

- **Next Steps:** H-W7 first (cheap, uses banked reads, no new stimuli), then H-W6. Both must be
  pre-registered before running.
