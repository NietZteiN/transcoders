# 2026-09-03 — A 16× α range delivers a 1.63× perturbation: the sweep saturates

**Thread:** nla-harness · **Status:** observation from a completed run (job 372844) · **Bears on:** B4's "across a 16× α range" claim

---

## What was measured, and why

Calibrating the KV-bypass energy match required the delivered perturbation as a function of α:

> r(α) = ‖h_final^steered − h_final^unsteered‖ / ‖h_final^unsteered‖ at the steered position,
> median over 12 items.

That is a byproduct of the frozen energy-matching rule, not a new experiment — but it answers a
question nobody had asked: **how much does α actually buy?**

## Result

Host `gemma12b` (`google/gemma-3-12b-it`), layer 32 of 48, V3 leave-one-out directions, n = 12.

| single-layer α | delivered r |
|---|---|
| 0.25 | 0.9778 |
| 0.5 | 1.1591 |
| 1.0 | 1.2722 |
| 2.0 | 1.4157 |
| 4.0 | 1.5978 |

**A 16× range in α delivers a 1.63× range in perturbation.** The curve is already at r ≈ 0.98 at
the smallest α tested — the steered state differs from the unsteered one by about its own norm —
and quadrupling α twice more adds 63%.

Multi-layer, for contrast, spans 7.29× over a 500× α range (0.2344 → 1.7091) and saturates at the
same ceiling, ≈ 1.7.

## The consequence for B4

The banked B4 result is reported as *"V1 does not beat V3 anywhere across a **16× range of
alpha**"* (`configs/b4_steering.yaml`, and the 2026-08-16 entry). That phrasing invites the reading
that the intervention space was swept broadly. **It was not.** If the same compression holds on the
banked host, the sweep covered a **1.6× range of delivered perturbation**, not a 16× one.

This does **not** refute B4. It bounds what B4 established. The honest statement is:

> V1 does not beat V3 across a range of injected coefficients that spans 16× in α but only ≈1.6×
> in delivered perturbation, measured at the steered position's final-layer residual.

## Scope — stated plainly, because this is the weak point

- Measured on **Gemma-3-12B at layer 32**, not on Qwen-2.5-7B at layer 20 where B4 ran. Under the
  2026-09-02 model constraint the Qwen measurement **cannot be made**, so the transfer is an
  inference, not a result.
- The mechanism that would make it transfer is generic: **RMSNorm/LayerNorm on the residual path
  at every block** compresses injected magnitude, and every model in the panel has it. Gemma's
  activation norms run ~1000× Qwen's, so the *absolute* numbers certainly do not transfer; the
  *saturation shape* plausibly does. **Plausibly is not demonstrated.**
- r is measured at one position (the steered one) and one layer (the last). A different summary of
  "how much the intervention did" could saturate differently.
- n = 12 items, medians. Adequate for a calibration, thin for a claim about a curve's shape — and
  this thread has already been burned once by reading shape off n = 60 (`2026-08-30_p1b-position-
  depth-results.md`).

## What this changes

Nothing about the KV-bypass decision rules; the energy match was frozen before this was seen and is
unaffected. What it changes is **wording**: any restatement of B4 that leans on "16× range" is
overstating the sweep's coverage and should carry the delivered-perturbation figure instead.

## Incidental

The **empirical** energy match selected multi α = **0.1** against the analytic 1/√33 = **0.1741** —
the analytic rule would have injected **1.74×** too much. Matched ratio 0.95, comfortably inside the
frozen [0.5, 2.0] acceptance band. LOO coherence of the contrastive direction at L32 is **0.4213**.

## New questions

- Does the saturation come from normalisation, or from the direction becoming increasingly
  orthogonal to what the upper layers propagate? Separable by measuring r layer-by-layer rather
  than at the output only.
- If delivered perturbation is what matters, **every α sweep in this project should be reported in
  r, not in α.** That is a cheap, retroactive re-labelling for any run whose vectors survive.
