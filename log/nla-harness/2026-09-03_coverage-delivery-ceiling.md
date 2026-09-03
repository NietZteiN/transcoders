# 2026-09-03 — Identifier tokens can deliver at most ~49% of a direct edit, at any magnitude

**Thread:** nla-harness · **Jobs:** 374550, 374558 (gemma12b L32/48) · **Status:** the pre-registered matching rule REFUSED, twice, and the refusal is the result

---

## What was being calibrated

Experiment C compares steering at **one token** (`last_prompt`, every banked result) against the
**annotated identifier spans** (median **45** tokens, 18.4% of the prompt — carried in the stimuli
since Paper 2, never once used). Widening coverage at unchanged α is the uncontrolled experiment
P0.2 already ran, where the parse rate collapsed to 0.100, so the arms have to be matched on
**delivered perturbation**:

> r(α, coverage) = ‖h_final^steered − h_final^unsteered‖ / ‖h_final^unsteered‖, read at the **last
> prompt token** — the position every coverage setting shares and where the answer is produced.

## The refusal, and then the ceiling

First pass (grid to α = 4) refused: `id_spans` reached r = 0.482 against a 1.268 reference, ratio
0.38, outside the frozen [0.5, 2.0] band. The rule's own remedy is *widen and re-run*, which
consults displacement only and no outcome. So the grid went to **α = 64**.

| α | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| `id_spans` r | 0.128 | 0.204 | 0.339 | 0.482 | 0.552 | 0.589 | 0.609 | **0.618** |
| gain per doubling | +76% | +59% | +66% | +42% | **+15%** | +6.7% | +3.4% | **+1.5%** |

**It converged.** A 16× increase from α = 4 to α = 64 buys +28% total, and the final doubling buys
1.5%. Reference (`last_prompt`, α = 1.0) is **1.268**.

> **`id_spans` asymptotes at ≈ 0.62 — 48.8% of the reference.**
> This is a **ceiling, not a grid limit.** Widening further chases an asymptote.

The primary convention is therefore **UNREACHABLE BY MEASUREMENT**, not by insufficient search,
and it is reported as such rather than quietly replaced.

## What this means mechanically

A write at layer 32 on the identifier tokens reaches the answer position **only through attention
in layers 33–47**. Pushing those 45 source states arbitrarily hard still moves the answer
position's final state by at most about half what editing that position directly achieves.

The asymptote is an **upper bound on how much of the answer position's final state is reachable
from the identifier positions at this depth.** It is the same geometry as the KV bypass: a
layer-L write propagates upward only, so the influence of an edited source position is capped by
the attention mass the read position gives it in the layers above.

**An accidental vindication of a choice nobody validated.** The banked single-token site was
inherited from the original NLA paper and hardcoded. On delivery grounds it turns out to be the
*strongest available* site for moving the answer — no other placement reaches the readout as
hard. That does not make it the right site for the *hypothesis* (adversarial renaming is about
identifiers, which is why C exists), but it does mean the banked nulls were not weak for want of
delivery to the readout.

## Consequence for Experiment C

The primary arm is unreachable, so C runs:

1. **`last_prompt` @ α = 1.0** — the reference, the banked site.
2. **`id_spans`, energy-matched** @ α = 1.0/√45 ≈ **0.149** — the second convention, pre-registered
   alongside the first: equal *total injected energy*, answering "same intervention budget, spent
   at the identifiers instead of at the answer position."
3. **`id_spans`, max-delivery** @ α = **8.0** — the knee of the curve above, past which each
   doubling buys under 15%. **Selected on the displacement curve alone, before any generation, and
   declared here.** A per-token edit at 8× the local activation norm is far outside any regime this
   project has tested, so a parse-rate collapse (C-2) is an expected outcome and would itself be
   informative.

Arm 3 is an addition to the pre-registration, made **before any Experiment C generation exists**
and on displacement data only. Recording it explicitly because adding arms is exactly the freedom
this project constrains.

## Bounds

- One host, one layer, n = 12 items for the curves, medians. The *level* of the asymptote is an
  estimate; that there **is** one, and roughly where, is robust across 13 grid points.
- r is read at one position and one layer. A different summary of "how much the intervention did"
  could saturate elsewhere.
- Says nothing about whether identifier positions *matter* to the model — only about how much a
  **layer-32 write there** can move the answer position's final state.

## Provenance

`nla/src/coverage_calibrate.py` (jobs 374550 grid→4, 374558 grid→64), `nla/src/span_positions.py`,
seed 20260724. Artifact `data/nla/p0/steerv2/gemma12b/coverage_match.json`. Span mapping verified
60/60 items on CPU, decoded tokens confirmed to be the identifier names.

## New questions

- **Does the asymptote rise if the write is multi-layer?** The KV-bypass machinery already writes
  layers 0..L; identifier tokens edited at *every* layer would reach the answer position through
  the full stack rather than only layers 33–47. That is a direct, cheap test of whether the ~49%
  ceiling is a depth artefact.
- **Is ~49% the attention mass on identifier tokens?** If the ceiling equals the fraction of the
  answer position's state attributable to those positions, it is measurable independently from
  attention weights — and would connect Instrument 3 to Instrument 1's A_id measure directly.
