# 2026-09-03 — KV bypass: pre-registration (steering v2, axis C)

**Thread:** nla-harness · **Status:** frozen before the run · **Depends on:** `docs/PLAN_steering_v2.md` (axis C)

---

## 1. The defect this tests

Every steering result this project has banked — B4 (refuted), B5 (null), P0.2 (channel) — was
produced by `steer.ActivationSteerer`, which hooks **one** transformer block. A write at
`layers[20]`, position *p*, propagates to layers 21–27 **for that position only**. The K/V entries
at layers 0–20 for *p* were computed before the hook fired and are left unedited.

**Consequence:** every subsequently generated token attends to the *un-edited* decoy state through
the bottom 21 of 28 layers. The intervention never changes what the model subsequently *reads*; it
changes only what one position contributes upward, in 7 of 28 layers.

This is a sufficient mechanical explanation for B4's null that is **independent of whether an
item-level belief exists**. It has never been tested. Until it is, "there is no belief to edit" and
"the channel cannot deliver" remain observationally equivalent for the write-side results.

**Measured, not assumed** (`data/nla/p0/steerv2/kv_bypass_gate.json`, job 372668, PASS):

| at the edited position, layers 0–20 | max |Δ| |
|---|---|
| single-layer steering (the banked instrument) | **0.000** |
| multi-layer prefill steering | 682.698 |

Exactly zero. The bypass is not a hypothesis about the code; it is a property of it.

## 2. Instrument

`nla/src/steer_multilayer.py` — writes `α·‖h_ℓ‖·Δ̂_ℓ` at every layer ℓ ∈ [0, 20], at the target
position, **during prefill only**. Prefill-only is deliberate: writing during decode as well would
confound closing the bypass (axis C) with steering the generation (axis A).

Validated by `nla/src/kv_bypass_gate.py`, all four checks PASS:
1. `alpha0_identity` — α=0 is **byte-identical** to unsteered (max |Δ| = 0.0), 0 positions written.
2. `writes_localised` — 21 writes for 21 hooked layers; max change at **any other position** = 0.0.
3. prefill-only — no decode-step writes.
4. `bypass_demonstrated` — the table above.

## 3. Directions — and what is deliberately excluded

**V3** (held-out mean contrastive direction, leave-one-out) and **V4** (item oracle) only. Both are
plain activation differences and are therefore **defined at every layer**; each layer gets its own
Δ_ℓ derived at that layer, extracted through `layer_rotation.all_layer_acts` (same read position,
same gated code path as P0.1).

**V1/V2 are excluded and this is not an oversight.** The AR is trained at layer 20 only, so no
licensed layer-7 NLA direction exists. P0.1 returned **HARD** (cos(Δ₂₀, Δ_ℓ) = 0.277 over 21–27),
which forbids re-using the L20 vector elsewhere — it does not forbid steering elsewhere with the
correct per-layer vector. **This run therefore tests the CHANNEL, not the NLA.** It cannot and will
not be reported as evidence for or against the NLA's causal role.

## 4. Energy control — the confound, and the frozen rule

Editing 21 layers at α each is not 21× one edit; it compounds, because a change at ℓ alters the
input to ℓ+1, which is then edited again. The gate measured the result: α=1.0 → 682 units of
displacement. **P0.2 already showed where oversized perturbation leads** — widening coverage at
unchanged α drove V1's parse rate to **0.100**, destroying generation rather than under-delivering.
An uncontrolled comparison here would repeat that error and be uninterpretable.

**FROZEN RULE.** Let r(α) = ‖h₂₇^steered − h₂₇^unsteered‖ / ‖h₂₇^unsteered‖ at the steered
position, median over the first 12 pairs. The matched multi-layer α is the grid point minimising
|log r_multi(α) − log r_single(1.0)|. Grid: {0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}.

This is a calibration on **displacement**, not on any outcome — no accuracy, parse rate, or reply
text is consulted in selecting it. Computed by `nla/src/multilayer_vectors.py` (job 372695) and
recorded in `data/nla/p0/steerv2/energy_match.json` **before** any steering run.

The analytic `1/√n` in `MultiLayerSpec.energy_matched` holds the sum of squared *coefficients*
constant, which is **not** the same as holding delivered perturbation constant once edits compound.
It is reported alongside for comparison and is **not** the primary.

## 5. Conditions

| arm | layers written | α | role |
|---|---|---|---|
| unsteered | — | — | baseline |
| single-V3 | {20} | 1.0 | the banked instrument, reproduced |
| multi-V3 | 0..20 | matched (rule §4) | **primary** — same delivered energy, KV corrected |
| multi-V3-loud | 0..20 | 1.0 | shows what uncontrolled coverage does |
| single-V4 | {20} | 1.0 | oracle ceiling, single-layer |
| multi-V4 | 0..20 | matched | oracle ceiling with the bypass closed |
| multi-random | 0..20 | matched | **control** — per-layer random directions, matched norms |

`multi-random` is not optional. P0.2's decision rule already established the principle: a gain that
random matches is generic perturbation, not delivery.

## 6. Predictions, stated before the run

- **H-C1 (bypass explains the null).** multi-V3 at matched energy beats single-V3 by **≥ +0.10**
  accuracy on L1b traps, CI excluding zero, and `multi-random` does **not** match the gain.
  → the channel was the bottleneck; **every write-side null in this project is re-opened**, and B4
  must be re-run through the corrected channel before its refutation stands.
- **H-C0 (bypass is irrelevant).** multi-V3 ≈ single-V3 within the reproducibility floor.
  → the write-side negative survives its strongest mechanical challenge. This is the outcome that
  makes the bounded causal negative publishable, and it is the outcome I expect.
- **H-C2 (energy, not coverage).** `multi-V3-loud` moves accuracy while matched multi-V3 does not,
  or parse rate collapses toward P0.2's 0.100 → the effect is magnitude, not delivery. Reported as
  a **negative** for axis C regardless of the sign of the accuracy change.
- **Degenerate outcome.** If matched multi-V3 and single-V3 both sit within the unsteered baseline's
  own run-to-run spread, the comparison is **UNINFORMATIVE** — explicitly *not* to be read as H-C0.
  (Same guard as P0.3's flat-curve clause.)

## 7. Reproducibility floor — binding

Greedy bf16 does **not** reproduce per item across runs (0.85–0.90 same-card agreement); reads are
bit-exact. So the unsteered baseline is re-run **in the same job** as the steered arms, and any
accuracy difference smaller than the baseline's own measured spread is reported as noise. Graded
k/N labels from repeated greedy draws; the `did-not-terminate` category is scored separately and
never silently folded into "wrong".

## 8. Provenance

- `nla/src/steer_multilayer.py`, `nla/src/kv_bypass_gate.py`, `nla/src/multilayer_vectors.py`
- gate: job **372668**, node g-08-01, 2026-09-03T03:25:26Z
- vectors + energy match: job **372695**
- seed **20260724**; sha256 of each script recorded in its job's stdout
