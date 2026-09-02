### Target Date: 2026-09-02 (Llama L2 REPLICATES — the first effect here to survive every control)

Pre-registration: [`2026-09-02_llama-l2-prereg.md`](2026-09-02_llama-l2-prereg.md), frozen before
the replication draws existed. Context: the three-host read-side battery and its deflation
controls.

- **Setup:**
  ```
  hosts   Qwen2.5-7B (banked) · Gemma-3-12B · Llama-3.1-8B — 3 x 5 tiers x 60 items x 10 draws
  fix     censoring corrected first (490 regenerations, not 6,000); non-terminating rows
          excluded and counted via terminated()
  draws   replication = 10-14, generated after the rule was frozen; discovery draws NOT pooled
  stat    mean rho over all layers minus max(length, static, combined) — the strict baseline
  ```

- **Results:**

  | set | tier | length | static | strict | residual | **beats** | p |
  |---|---|---|---|---|---|---|---|
  | discovery | **L2** | +0.1669 | +0.0873 | +0.1669 | +0.3176 | **+0.1507** | 0.0149 |
  | **replication** | **L2** | +0.2037 | +0.1135 | +0.2037 | +0.3099 | **+0.1062** | **0.0149** |
  | discovery | L3 | +0.2723 | +0.2596 | +0.3166 | +0.3273 | +0.0106 | 0.0050 |
  | replication | L3 | +0.1904 | +0.2091 | +0.2340 | +0.1771 | **−0.0569** | 0.0896 |

  **✓ REPLICATED.** L2 clears the frozen bar (+0.1062 against +0.075 required, p = 0.0149) on
  draws it had never seen. The residual stream lands at +0.3099 against +0.3176 in discovery —
  near-identical — while the baseline it must beat *rose* from +0.1669 to +0.2037, so the margin
  shrank for the right reason.

  **The sibling control did its job.** L3 shares the relational route and stays flat in both sets
  (+0.0106, then −0.0569). A generic property of relational obfuscation, or a pipeline artefact,
  would move L3 with L2. It does not.

  **This is 1 cell of 9.** Gemma shows nothing at any tier; Qwen's analogous effect deflated to
  static complexity; and the dispatcher-span probe is **not beyond size and repetition** on all
  three hosts, with identical code-derived baselines confirming the pipeline measures what it
  claims.

- **What worked / hypothesis verdict:**
  - **H-LL2 ✓ REPLICATED.** On Llama-3.1-8B at L2 (dispatcher indirection), a linear read of the
    residual stream predicts item-level correctness **beyond reply length, beyond static code
    shape, and beyond their combination**, on independent draws, by a rule set in advance.
  - **The first effect in this programme to survive pre-registered replication and every surface
    control available.**

- **Observations:**
  - **A defective control was found and fixed on the way here, and it changed a verdict.** The
    original comparison used a COMBINED ridge over length plus static shape. That model scores
    worse out-of-fold than its own best component in **6 of 9 cells** — by up to 0.118 — because at
    n = 60 with grouped folds, adding features can lose power. Both positive verdicts landed on the
    two most degraded cells. Under `max(length, static, combined)`, **Gemma L2 went +0.105 →
    −0.013** (artefact) while Llama L2 held at +0.151. Without that fix this entry would be
    reporting two positives, one of them false.
  - **What this is not.** Beating a baseline is not identifying what is represented. The Qwen
    precedent is exactly this: an effect that looked structural resolved into text repetition once
    the right control ran. **The repetition control has not been run against this effect** — only
    against the span probe. Until it is, "the residual stream carries something reply length does
    not, on this host at this tier" is the whole claim.
  - **It is host-specific and that belongs in the headline.** It appears on the *weakest* of three
    models (L0 accuracy 0.567 against Gemma's 0.777) and on neither of the others. Small n (58
    items after exclusions), modest effect (ρ 0.31 against a 0.20 baseline).
  - **Five positives have been retracted in this programme** — the depth gradient, the tier effect,
    L1b's argmax advantage, the Qwen relational mechanism, and Gemma L2. This one is different only
    because it was pre-registered and replicated, which is the point of doing it that way.

- **New questions / new hypotheses:**
  - **Run the repetition control against this effect**, not just the span probe. It is the control
    that deflated the Qwen version and it is cheap. **This should happen before E3.**
  - **E3 — attribution graphs on Llama-3.1-8B L2.** This is what the charter commissioned E3 for:
    asking *what* is represented rather than *whether* something is. Llama-3.1-8B has both Llama
    Scope SAEs and transcoders, and is the only panel model that does.
  - **Why only Llama?** Gemma is more accurate and shows nothing; a possible reading is that a
    stronger model's reply length is already a good enough correctness predictor to leave no
    headroom — Gemma has the highest length baselines of the three. Testable, and it would explain
    the host-specificity without appealing to anything about Llama's internals.

- **Next Steps:** repetition control on this effect; then E3 attribution graphs on Llama L2.
