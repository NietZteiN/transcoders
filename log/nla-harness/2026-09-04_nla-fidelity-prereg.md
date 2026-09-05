### Target Date: 2026-09-04 (H-W7 — how much of an activation's CAUSAL power survives the round trip through language? Frozen before running.)

Raised by [`2026-09-04_writeback-results.md`](2026-09-04_writeback-results.md), where the arm nobody
was watching turned out to be the finding: `C3_ceiling` = **+37.91 nats (31.3 % of a full prompt
swap)**, nearly double the edited protocol and above prompting — and **understated**, because 101 of
476 spans fell back to the unedited-L1b vector. **Committed before the run.**

#### First, why those 101 spans fell back — it is the stimuli, not the code

Diagnosed on CPU over all 476 annotated L1b identifier spans:

| | spans | share |
|---|---|---|
| **locatable** — decoy maps to a true term that exists in the L0 code | **375** | **78.8 %** |
| true term absent from L0 — `rename_map` gives `?unpaired0`, `?unpaired1`, … | 88 | 18.5 % |
| decoy not in `rename_map` at all | 13 | 2.7 % |

**18.5 % of annotated identifier spans have no original counterpart.** Their `rename_map` entries are
`?unpairedN` sentinels: adversarial renaming did not only *rename* identifiers, it **introduced new
ones**. The remaining 2.7 % are language builtins and one-character loop variables (`istitle`,
`lower`, `casefold`, `ch`, `d`) that were annotated as identifier spans but never renamed.

**This bears on the whole study, not just this experiment.** Any measure that assumes a
decoy↔true correspondence at annotated spans — E1's semantic-capture diff included — is undefined on
roughly a fifth of them. It is the same *class* of stimulus-level obstacle as T's alignment failure
([`2026-09-03_patch-alignment-impossible.md`](2026-09-03_patch-alignment-impossible.md)), found the
same way: by asking why a denominator shrank instead of accepting the number it produced.

- **Hypotheses / what we're testing.** The question stage 1 made unavoidable: the NLA channel
  demonstrably delivers, so **how much of an activation's causal power survives being turned into
  English and back?** No one in this project has measured that, and it is the most NLA-native
  question available — the released pair's own MSE/cos metric scores *reconstruction*, never
  *causal* fidelity.

  - **H-W7a:** the pure clean-state round trip beats **prompting**. CONFIRM if `C3pure`'s 95 % CI
    lower bound exceeds **+27.53** (banked `P_prompt`, same 60 items). REFUTE otherwise.
  - **H-W7b — the headline quantity.** **NLA causal fidelity = C3pure / P_patch**, where `P_patch`
    writes the clean activation's own direction at the same positions. CONFIRM "high fidelity" if
    the ratio's CI lower bound ≥ **0.50**; REFUTE if the CI upper bound < 0.50.
  - **H-W7c:** `C3pure` exceeds the round-trip null on the same positions by ≥ **+12.11** nats,
    CI excluding 0. This is the C1 clause carried over — without it a large `C3pure` could still be
    the operator rather than the content.

- **Setup (frozen).**
  - **Positions:** the **375 locatable spans only**, and **every arm writes at exactly those
    positions**. The `?unpaired` spans are excluded by construction, not by choice — no clean-state
    vector exists for them.
  - **Arms:** `C3pure` = AR(AV(h_L0)); `P_patch` = h_L0 itself; `C1r` = AR(AV(h_L1b)) — the
    round-trip null re-run on this position set so all three are commensurable; plus the unsteered
    noop.
  - **Both ceilings are direction-only, and that is what makes the ratio meaningful.**
    `PositionReplacer` writes `‖h‖ · unit(v)`, so `P_patch` supplies the clean activation's
    *direction* at the L1b position's own magnitude. Since the AR is trained on `MSE = 2(1−cos)` and
    its output norm carries no information, a direction-only ceiling is the only fair comparison —
    a full-magnitude patch would beat the AR on something the AR cannot even represent, and the
    ratio would measure the handicap rather than the fidelity.
  - **Readout `G_sum`**, n = 60 items, percentile bootstrap 10,000, seed 20260724, BH-FDR across the
    family. R2's behavioural veto (0.05 on accuracy or parse) applies to every arm.
    `--deterministic` OFF. Host `gemma12b` L32/48; AV/AR `kitft/nla-gemma3-12b-L32-{av,ar}`.
  - **Ratio CI** by the same item-level bootstrap, resampling items and recomputing both means —
    not by dividing two independently bootstrapped means, which would ignore their pairing.

- **What each outcome means, fixed in advance.**
  | C3pure vs prompting | fidelity ratio | reading |
  |---|---|---|
  | beats | ≥ 0.50 | **the NLA channel is the strongest intervention this programme has found**, and the paper's frame changes from "belief-steering fails" to *"the channel works; natural-language **edits** are what it cannot carry"* |
  | beats | < 0.50 | the channel is strong but language is a lossy bottleneck — quantified for the first time |
  | does not beat | ≥ 0.50 | the round trip is faithful but activations at these positions are simply not very causal |
  | does not beat | < 0.50 | stage 1's C3 was mostly the 21 % fallback contamination; retract the "channel delivers" reading |

- **Power / bounds, stated in advance.** n = 60 items on `G_sum`; stage 1's C3 CI was
  [+32.49, +43.47], so the arm is far from the noise floor. One host, one layer, one tier, and the
  same 60-item corpus every W result rests on. **A fidelity ratio measured at 375 identifier-span
  positions of obfuscated code does not license a general claim about NLA fidelity** — it is one
  regime, and the OOD one at that.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
- **Next Steps:** implement as arms in `nla/src/nla_writeback.py` behind `--arms/--locatable-only`;
  smoke, then 60 items. H-W6 (replacement-random null) stays queued behind this.
