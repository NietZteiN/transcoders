### Target Date: 2026-09-04 (Correction filed BEFORE the run: H-W2 as pre-registered is vacuous)

Amends [`2026-09-03_nla-writeback-prereg.md`](2026-09-03_nla-writeback-prereg.md). **Append-only —
that entry keeps its text.** Written while stage 1 is still being implemented and **no stage-1 or
stage-2 number exists**, so this cannot be an after-the-fact reinterpretation.

- **Hypotheses / what we're testing:** H-W2 is re-operationalised. H-W1 and the decision table are
  untouched.

#### The defect

The prereg defines H-W2 as:

> **H-W2 (the write registers).** Re-verbalizing the written position shows the read moved toward
> the edit. CONFIRM if `cos(AR(AV(h_after)), AR(text_edited)) > cos(AR(AV(h_after)), AR(text_read))`…

**`h_after` at the written position IS the written vector.** The stage-1 operator is a replacement
at layer 32, and the AV/AR pair reads layer 32 — so re-verbalizing that position returns what was
just written, and the comparison is true by construction at cos ≈ 1. It measures the arithmetic of
the assignment, not the model.

This is the fifth measurement-definition defect in this family's short life (R's ratio denominator,
S's ungated veto, T's unreachable threshold, G's normalisation, now this). It is the first one
**caught before the run rather than by a control afterwards**, which is the only reason it costs
nothing.

#### The replacement, frozen now

The informative question is not whether the written position holds what we wrote, but whether the
write **reaches the position where the answer is produced**. That is `last_prompt` — the site every
banked steering arm targeted, and where R2 measured `V5_replace` at +0.0014 of G.

- **H-W2′ (frozen).** With the stage-1 W1 write applied at the spans, read layer 32 at
  **`last_prompt`** and ask whether it moved **in the direction of the edit**:
  - `Δh = h_last(steered) − h_last(unsteered)`
  - `Δedit = mean over the item's spans of [ AR(edited read) − AR(unedited read) ]`
  - **CONFIRM** if `cos(Δh, Δedit) > 0` on **> 50 %** of items with a bootstrap CI excluding 0.50.
  - **Controls, same statistic:** `Δedit` from a **foreign** item (C2's edit content) must not
    register; and the **C1** arm's `Δh` — a write that carries no edit — must not register against
    `Δedit`.
  - Both sides are **differences**, so they are centred by construction and the anisotropy that
    made raw cosine near-degenerate in stage 0 ([`2026-09-04_cycle-gate-results.md`](2026-09-04_cycle-gate-results.md))
    cannot inflate this statistic. That is deliberate, and it is why the statistic is a direction
    agreement rather than a raw similarity.

- **Descriptive companion, no gate:** AV-verbalize `h_last` steered vs unsteered and report the
  shift in H-W3's decoy/true mention rates. "What the model says it represents at the answer site,
  before and after the edit" is the Instrument-2-native version of the same question.

- **The decision table is unchanged**, and H-W2′ slots into it exactly where H-W2 did — including
  the row that matters most: **null H-W1 × registering H-W2′ = readout ≠ mechanism**, now with a
  meaning it did not have before, since a vacuous H-W2 would have made that row unreachable by
  guaranteeing "registers" in every case.

- **Setup:** unchanged. **Results / verdict:** *pending; nothing has been run.*
- **Next Steps:** implement stage 1 + H-W2′ in `nla/src/nla_writeback.py`.
