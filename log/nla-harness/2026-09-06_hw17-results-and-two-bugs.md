### Target Date: 2026-09-06 (H-W17 — meaning carries **64 %** of the item component on Python; plus two bugs, one of which invented a verdict contradicting a result I already had)

**Thread:** nla-harness · **Job:** 379908 (h200, **45:23**, COMPLETED, 0 errors) ·
**Resolves:** [`2026-09-06_meaning-clean-prereg.md`](2026-09-06_meaning-clean-prereg.md), under
[`2026-09-06_hw17-language-confound.md`](2026-09-06_hw17-language-confound.md).

- **The result, on the 20 items that actually received a write** (19/20 Python — see the language
  addendum; this is a **Python** result and must always be cited as one):

  | arm | G_sum | 95 % CI |
  |---|---|---|
  | `T_L0_ren` — true identifiers | **+34.19** | [+28.44, +40.07] |
  | `T_L1_ren` — **nonsense** identifiers | **+12.37** | [+9.36, +15.41] |
  | **gap** | **+21.82** | **[+15.52, +28.26]** |

  **H-W17a ✓ SUPPORTED** — the gap clears the +12.11 bar with the CI well clear of it, positive on
  **19 of 20** items. **Nonsense identifiers recover only 0.362 of what true identifiers recover.**
  SELF identity **−0.0096** → PASS (fourth consecutive).

- **What this settles.** **Identifier *meaning* carries ~64 % of the item component, and mere
  *removal* of the decoy carries ~36 %.** Both are large and both are real. The
  "removal suffices" reading that H-W16b's artifact produced — **and that I had pre-registered as my
  expectation** — is **refuted** on the clean contrast. My prior was wrong, and the confound had
  been agreeing with it.

- **⚠️ Bug #8: `--strict-anchor` silently nulled the L2 arm, and the frozen rule reported a verdict
  from it.**

  The flag was documented as restricting the **L1** contrast. But `tier_anchor()` is called for L1
  **and L2**, and **L2 has no `rename_map` at all** (0/49 items — measured on 2026-09-06 before the
  previous job). L2 anchors *only* through the fallback path, which is exactly what strict mode
  removes. So `T_L2_all` wrote at **zero positions** in every item: `L2_ok: 0`, `dG = +0.00` with a
  CI of [+0.00, +0.00].

  The scorer then computed `H_W16a_structure = T_L0_all − T_L2_all = 45.71 − 0.00 = +45.71`,
  `clears=True`, and returned **`W16-STRUCTURE-CARRIES-IT`** — from an arm that was not an arm.
  **That verdict is void.** It also directly contradicts the same contrast measured properly in job
  379819, where L2 recovers **98.6 %** and structure is worth **+0.66 nats**. The genuine result
  stands; this one is discarded.

  A CI of exactly [0.00, 0.00] should be impossible for a real intervention and is the signature to
  watch for. Nothing in the harness treats "this arm wrote nothing" as different from "this arm did
  nothing", which is the same class as B5's silently-shrinking denominator and P0.3's missing
  `cells_scored.json`.

- **⚠️ Bug #9: `score()` averaged 29 structural zeros into both sub-arms.**

  With strict anchoring, 29 of 49 items have no qualifying span, so both `T_L0_sub` and `T_L1_sub`
  wrote nothing and scored **exactly 0.0000** there (verified: max |ΔG| over those 29 rows = 0.0000).
  The scorer averaged over all 49 rows regardless, reporting `T_L0_sub` **+13.95** and a gap of
  **+8.91** that *failed* the bar — against **+34.19** and **+21.82** on the items that actually
  received a write. **The printed verdict line for H-W16b is therefore also wrong**, in the
  direction of a false negative.

  Both numbers come from the same rows; **no GPU was needed to correct this**, only filtering on
  `n_positions_sub > 0`, which the rows already carried. That is the payoff from persisting
  per-item rows — the discipline the H-W4 correction argued for two days ago.

- **Observations.**
  - **Three of my last four verdict lines have been artifacts** (H-W16b's fallback dilution, this
    run's nulled L2 arm, this run's zero-row averaging), while the *underlying measurements* were
    sound each time. The pattern is consistent and worth stating plainly: **the frozen decision
    rules keep firing on arms whose denominators changed underneath them.** A rule that reads a mean
    without asserting how many positions produced it is not a decision rule, it is a formatting step.
  - **Concrete fix for anything downstream:** every arm must record `n_positions_written` and the
    scorer must **refuse** (not average) any arm whose positions are 0 on an item, and flag any arm
    whose written-position count differs from its comparator's. That is a harness change, not a new
    experiment, and it would have caught bugs #8 and #9 automatically.
  - Updated decomposition of the item component, Python only, provisional: **meaning ≈ 64 %,
    decoy-removal ≈ 36 %, control-flow structure ≈ 1 %** (the last from job 379819's valid L2 arm).

- **New questions / new hypotheses.**
  - **H-W20:** add written-position accounting to the scorer and re-score every banked W arm. No GPU.
    Until then, no W arm's mean is trustworthy without checking its position count by hand.
  - **H-W19** (from the language addendum) is unchanged and now more pressing: the L1 tier may not be
    comparable across languages anywhere in the study.

- **Next Steps:** H-W20 first — it is free, and it audits every number this thread has produced.
