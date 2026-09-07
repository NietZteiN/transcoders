### Target Date: 2026-09-07 (The `arm_guard` refused a contrast and was right: H-W15's residual is a position-count artifact — the verdict survives, its caveat does not)

**No GPU.** Follows [`2026-09-06_tier-source-results.md`](2026-09-06_tier-source-results.md) and
[`2026-09-06_arm-guard-and-audit.md`](2026-09-06_arm-guard-and-audit.md). **Append-only.**

- **How it surfaced.** Job **380557** failed at 8:29 with
  `ArmNotWritten: R_2-P_2: no item where both arms wrote at matching positions` — the guard built
  yesterday for bugs #8/#9, refusing a contrast I had already run twice and reported.

  **It was right.** The forward and reverse dose ladders select **different spans** at each k, spans
  differ in token length, so the two arms write **different numbers of token positions**:

  | k | forward positions | reverse positions | equal on |
  |---|---|---|---|
  | 1 | 3.24 | **3.55** | 27/49 items |
  | 2 | 6.12 | **6.63** | 16/49 |
  | 4 | 12.22 | **13.37** | 8/49 |

  The reverse ladder writes **more** at every k. H-W15's contrast was therefore confounded with
  *amount of intervention* — the exact confound the dose family's `F_k` controls exist to remove,
  and which nothing checked for the ladder.

- **What the confound explains.** [`2026-09-06_tier-source-results.md`](2026-09-06_tier-source-results.md)
  reported R−P = **+1.10 / +0.98 / +2.40** and flagged, as a curiosity, that *"reverse runs
  consistently ~1–2 nats higher at all three k … the sign is consistent and is flagged, not buried."*
  It is now explained. The dose curve's own rate is **~1.86 nats per written position**
  (45.7 nats over 24.6 positions), and the position asymmetry predicts:

  | k | extra positions | predicted | observed | explained |
  |---|---|---|---|---|
  | 1 | +0.31 | +0.58 | +1.10 | 53 % |
  | 2 | +0.51 | +0.95 | +0.98 | **97 %** |
  | 4 | +1.14 | +2.12 | +2.40 | **88 %** |

  Per-item Spearman of the residual against the position difference: ρ = +0.144 (p = 0.32),
  **+0.279 (p = 0.053)**, **+0.283 (p = 0.049)** at k = 1, 2, 4.

- **Verdict: H-W15 ✓ STANDS, and is strengthened.** "Saturation is about *quantity*, not *which*
  spans" survives — indeed the one thing that looked like a residual position-identity effect turns
  out to be **another quantity effect**. What does not survive is the caveat: the consistent
  positive sign is not an unexplained hint of an ordering effect, it is the reverse ladder writing
  more tokens. **That sentence in the 2026-09-06 entry should be read with this one.**

- **Observations.**
  - **A guard written for one failure caught a different, older one.** `arm_guard` was built for
    bugs #8/#9 — arms that wrote *nothing*. It fired on arms that wrote *unequal amounts*, in an
    experiment that had already been run twice and written up. That is the first time in this family
    that a check has caught a defect in a **banked** result rather than in the run it was guarding.
  - **The fix was to remove the arms, not to weaken the guard.** H-W15 is settled and its residual is
    now explained offline, so the repaired run drops the ladder (`--no-ladder`) instead of relaxing
    `paired()` to admit a known-confounded contrast. Weakening a check to make a run proceed is how
    the check stops meaning anything.
  - **Ten defects now.** The split holds: the measurements have been sound every time; what fails is
    the layer between a measurement and a verdict. This one differs in a useful way — it was caught
    by an automated invariant rather than by me noticing an odd-looking number, which is the first
    evidence that the guard programme is doing what it was meant to.

- **New questions / new hypotheses.**
  - **H-W24:** every contrast in this family should be re-checked for position-count equality, not
    just the ladder. The dose family's `P_k − F_k` contrasts are matched by construction (same spans,
    different vectors) and are safe; **`T_L0_all − T_L2_all` and `T_L0_sub − T_L1_sub` need
    verifying**, since a tier anchor can map to a term of different token length than its L0
    counterpart. Cheap, no GPU, and it bears on H-W16a and H-W17a directly.

- **Next Steps:** H-W24 before quoting H-W16a/H-W17a again. Job **380566** is the repaired re-run
  with the ladder dropped.
