### Target Date: 2026-09-06 (H-W15 saturation source + H-W16 what the 56 % item component IS. Frozen before running.)

Raised by [`2026-09-06_dose-and-tier-results.md`](2026-09-06_dose-and-tier-results.md), which
decomposed the clean-state effect and left **56.2 % (+25.67 nats) unexplained as "it comes from this
item"**. **Committed before the run.** One job; every arm writes at subsets of the same 375
locatable positions.

#### The stimulus ladder makes a much sharper test available than I expected

All five tiers (**L0 · L1 · L1b · L2 · L3**) exist for **all 49** dose items. Patching an L1b span
from a *different tier of the same program* asks directly what the item component is made of:

| tier | identifiers | control flow | what patching from it isolates |
|---|---|---|---|
| **L0** | true | original | the anchor (+45.71, measured 3×) |
| **L1** | **nonsense** (`c`, `d`, `e`) | original | **uninformative names, no trap** |
| **L2** | **true** | **flattened** | true names, wrong structure |

**L1 vs L1b is the distinction Papers 2–3 are built on** — L1 is merely *uninformative* while L1b is
a *trap* — so H-W16b asks whether the causal repair is **deleting the decoy** or **installing the
truth**. Nothing in this programme has been able to ask that until now.

**Coverage, measured before designing (and the reason the two arms differ in scope):**

| anchor | spans covered | note |
|---|---|---|
| **L2** | **375 / 375 (100 %)** | L2 preserves identifiers, so the true term is present verbatim |
| **L1** | **190 / 375 (50.7 %)** | 131 via `meta.rename_map`, **59 unrenamed** (the true term survives into L1 and would have been missed by a rename_map-only lookup); 185 genuinely absent |

All 49 items retain ≥ 1 L1-anchored span. **Because L1 covers only half the positions, its arm runs
on the 190-position subset with a matched `T_L0_sub` arm at exactly those positions** — the
comparison is paired and within-run, never against the 375-position anchor. A shrinking denominator
compared against a full one is precisely the P0.3 failure mode.

- **Hypotheses.**
  - **H-W16a (structure vs identifiers).** `T_L0_all − T_L2_all`. **CONFIRM "structure matters"** if
    the gap ≥ **+12.11** with a CI excluding 0. **REFUTE** → the item component is carried by
    identifier content, which L2 preserves, and control-flow structure contributes little.
  - **H-W16b (delete the trap, or install the truth?) — the sharp one.**
    `T_L0_sub − T_L1_sub` on the matched 190 positions. **If the gap is < +12.11**, **nonsense names
    repair the damage as well as true names do**, and the mechanism is *removal of the decoy* rather
    than *restoration of meaning* — which would make the whole W programme a result about
    interference rather than about belief. **If ≥ +12.11**, true identifier semantics carry it.
  - **H-W15 (is saturation about quantity or position?).** Re-run the dose ladder in **reverse**
    seeded order (`R_1, R_2, R_4`) alongside the **forward** ladder (`P_1, P_2, P_4`) in the same
    job. The two select different span subsets at each k. **CONFIRM "quantity"** if every
    |R_k − P_k| < **+12.11** with the forward/reverse difference CI containing 0 — the yield depends
    only on *how many* spans are patched. **REFUTE** → some spans carry more than others, and the
    dose curve's saturation is a property of *which* positions, not of the amount.

- **Setup (frozen).** Arms: `T_L0_all`, `T_L2_all` (375 positions); `T_L0_sub`, `T_L1_sub` (190);
  `P_1/P_2/P_4` and `R_1/R_2/R_4` (forward and reverse ladders, same seeded order as job 379075 so
  the forward arms replicate it); plus the unsteered noop. `PositionReplacer`, direction-only, no α.
  Cross-tier anchor = the matched term's **first occurrence, last token**, the same rule as the L0
  anchor. Readout `G_sum`; paired percentile bootstrap 10,000; seed 20260724; BH-FDR across the
  three gated contrasts. **Generations only for noop, `T_L2_all` and `T_L1_sub`** — the two new
  claims; every other arm is reported with `veto: null` and cannot carry a verdict.
  **SELF is re-asserted** (`|ΔG| ≤ 1.0`) and the run **exits 3** rather than reporting if it fails.
  `--deterministic` OFF. Host `gemma12b` L32/48.

- **Stated in advance.** I expect **H-W16b to fail its bar** — i.e. nonsense to work nearly as well
  as truth — because the decomposition already shows only **9.1 %** of the effect attaches to a
  vector being *clean* rather than *obfuscated*. **That outcome is the more interesting one and must
  not be written up as a disappointment:** it would relocate the whole finding from *"restoring
  belief"* to *"removing interference"*, and it is the mechanism Papers 2–3's Stroop framing
  predicts. Recording the prediction now so the write-up cannot drift toward whichever result
  arrives.
- **Power / bounds.** n = 49 for the 375-position arms; the same 49 items contribute ≥ 1 span to the
  190-position subset. Prior contrasts in this family had CIs of ±5 nats, so a 12.11-nat bar is
  resolvable and a null is informative rather than underpowered.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
- **Next Steps:** implement as `nla/src/nla_tiers.py`; smoke, then 49 items.
