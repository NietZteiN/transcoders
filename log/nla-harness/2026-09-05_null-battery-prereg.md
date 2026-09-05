### Target Date: 2026-09-05 (Null battery — is the clean-state effect item-specific, span-specific, or just the operator? Frozen before running.)

Raised by [`2026-09-05_nla-fidelity-results.md`](2026-09-05_nla-fidelity-results.md), which recorded
that **the decisive null is missing**: `C3pure`/`P_patch` reach 40 % of a prompt swap and beat
prompting, but nothing in that design rules out that **any** clean-code direction — or any direction
at all — does the same at these positions. Until this runs, "the NLA preserves *item-specific*
causal content" outruns the evidence. **Committed before the run.**

Three nulls, one job, because they share every position and differ only in what is written.

- **Hypotheses (all against `P_patch`, re-run in the same job as the internal anchor so every
  contrast is within-run).**
  - **H-W9 — item specificity.** `P_patch − N_foreign_clean` ≥ **+12.11** nats, CI excluding 0,
    where `N_foreign_clean` is the clean activation of a **different item's** matched span.
    CONFIRM → the effect carries item-specific content. **REFUTE → it is "any clean-code
    direction", and H-W7's fidelity result collapses to a statement about the operator.**
  - **H-W6 — operator null.** `P_patch − N_random` ≥ **+12.11**, CI excluding 0, where `N_random`
    is a seeded random unit vector. REFUTE → norm-matched replacement of decoy identifier
    activations helps regardless of content, which would be a result about the **trap** (its damage
    lives in those activations being *specifically wrong*) rather than about the NLA.
  - **H-W12 — span specificity.** `P_patch − N_shuffled_clean` ≥ **+12.11**, CI excluding 0, where
    `N_shuffled_clean` is the clean activation of a **different span within the same item**. This is
    the causal analogue of stage 0's within-item cosine test, which passed on representation
    (+0.6807 vs +0.0525); H-W12 asks whether that specificity survives into *behaviour*.

- **Setup (frozen).** Identical to H-W7: the **375 locatable spans**, 49 items, every arm writing at
  exactly the same positions with `PositionReplacer` (norm-matched, direction-only, **no α**).
  Arms: `P_patch` (anchor), `N_foreign_clean`, `N_shuffled_clean`, `N_random`, plus the unsteered
  noop. Foreign pairing and the random vectors are drawn from a **seeded** function of
  `(snippet_id, span index)` so the arms are reproducible and cannot drift between runs.
  Readout `G_sum`; percentile bootstrap 10,000; seed 20260724; BH-FDR across the three contrasts;
  R2's behavioural veto (0.05) on every arm; `--deterministic` OFF. Host `gemma12b` L32/48.

- **Verdicts, fixed in advance.**
  | H-W9 | H-W6 | reading |
  |---|---|---|
  | clears | clears | **`W9-ITEM-SPECIFIC`** — H-W7's fidelity claim stands as an item-specific one |
  | fails | clears | **`W9-GENERIC-CLEAN`** — any clean-code direction works; the fidelity number describes the *class*, not the item, and the H-W7 entry's wording must be amended |
  | — | fails | **`W9-OPERATOR-ONLY`** — replacement at these spans helps regardless of content; **retract the causal-fidelity framing entirely** and rewrite it as a finding about the trap |

- **Stated in advance, because it is the likely outcome and must not be spun.** Residual streams are
  anisotropic — stage 0 measured every raw pairwise cosine near 0.97 — so a *foreign* clean
  activation is not an obviously different direction from the matched one. **A partial or complete
  failure of H-W9 is a live possibility, and it would be the correct and publishable result**: it
  would say the identifier positions accept "a clean-code-shaped direction" and that the trap's
  damage is at that coarse a granularity. The H-W7 entry already flags that its fidelity figure and
  stage 0's raw cosine are one fact, not two; this battery is what decides how that fact should be
  described.

- **Power / bounds.** n = 49 items, the same easier subset (baseline acc 0.714, swap unit +111.57).
  H-W7's contrasts had CIs ±5 nats, so a +12.11 gap is comfortably resolvable. One host, one layer,
  one tier, 375 identifier-span positions of obfuscated code.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
- **Next Steps:** implement as `nla/src/nla_nulls.py`; smoke, then 49 items.
