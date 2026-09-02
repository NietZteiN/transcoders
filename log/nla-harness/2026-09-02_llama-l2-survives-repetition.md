### Target Date: 2026-09-02 (Llama L2 survives the repetition control — 1 of 9 cells)

Follows [`2026-09-02_llama-l2-replicated.md`](2026-09-02_llama-l2-replicated.md), whose own Next
Steps named this control as the prerequisite for E3: it is the control that turned the Qwen
analogue from a structural claim into a surface statistic.

- **Setup:** `p1b_l2_mechanism.py` with repetition folded into the baseline, so the bar is
  `max(length, static, repetition, combined, all-three)` — five ways for a surface statistic to
  explain the effect. Scored on the replication draws (10–14) alone, discovery draws not pooled.

- **Results:**

  | set | tier | length | static | **repetition** | strict | residual | **beats** | p |
  |---|---|---|---|---|---|---|---|---|
  | discovery | **L2** | +0.1669 | +0.0873 | **+0.0005** | +0.1669 | +0.3176 | **+0.1507** | 0.0149 |
  | **replication** | **L2** | +0.2037 | +0.1135 | **+0.0363** | +0.2037 | +0.3099 | **+0.1062** | **0.0149** |
  | discovery | L3 | +0.2723 | +0.2596 | +0.2531 | +0.3166 | +0.3273 | +0.0106 | 0.0050 |
  | replication | L3 | +0.1904 | +0.2091 | **+0.2615** | +0.2615 | +0.1771 | **−0.0845** | 0.0896 |

  **✓ SURVIVES** at +0.1062, p = 0.0149 — unchanged, because repetition explains essentially none
  of it.

  **The control is not inert, which is the diagnostic part.** Repetition predicts nothing at L2
  (+0.0005, +0.0363) but is a *strong* predictor at L3 (+0.2531, +0.2615), where it becomes the
  binding baseline and drives L3 to **−0.0845**. So the control fires exactly where the Qwen
  precedent said it would and does not fire at L2. A control that merely passed everywhere would
  be far weaker evidence.

  **The full matrix — 8 of 9 cells reduce to a surface statistic:**

  | host | L2 | L3 | L1b |
  |---|---|---|---|
  | Qwen2.5-7B | +0.0345 | +0.0420 | −0.0362 |
  | Gemma-3-12B | −0.0129 | −0.1009 | −0.1535 |
  | **Llama-3.1-8B** | **+0.1109** | −0.0491 | −0.0884 |

- **What worked / hypothesis verdict:**
  - **✓ The effect is not text repetition.** On Llama-3.1-8B under dispatcher indirection, a linear
    read of the residual stream predicts item-level correctness beyond reply length, static code
    shape, text repetition and every combination — replicated on independent draws, by a rule
    frozen in advance, with a sibling tier null throughout.
  - **The first result in this programme to reach that bar.** Five earlier positives died to
    controls; this one has now passed the specific control that killed the closest analogue.

- **Observations:**
  - **A circular import was introduced and caught in 5 seconds.** Folding repetition into the
    mechanism scorer created a cycle — `p1b_span_probe` already imported `stimuli` from
    `p1b_l2_mechanism`. Fixed structurally by moving `repetition_features` into `p1b_ladder.py`,
    which both import and which imports neither. That also removed a **duplicated definition** of
    the function (it existed in both `p1b_span_probe` and `p1b_span_positions`), which was a latent
    bug in its own right: the two copies could drift and compute different "repetition" in
    different controls.
  - **Beating every surface baseline is not knowing what is represented.** That distinction is the
    entire remaining gap, and it is what E3 exists to close.
  - **Scope, stated plainly:** one host of three, one tier of five, n = 58 after excluding
    non-terminating rows, ρ = 0.31 against a 0.21 baseline. It appears on the *weakest* host
    (L0 accuracy 0.567 against Gemma's 0.777) and on neither of the others.
  - **A candidate explanation for the host-specificity that does not appeal to Llama's internals:**
    Gemma has the highest reply-length baselines of the three, so a stronger model's verbosity may
    already predict correctness well enough to leave no headroom. Testable, and it would explain
    why the effect appears where it does.

- **New questions / new hypotheses:**
  - **E3 — attribution graphs on Llama-3.1-8B L2.** The question is now *what* is represented, not
    *whether*. Llama-3.1-8B is the only panel model with both Llama Scope SAEs and transcoders.
  - **Test the headroom explanation** — does the size of the effect track the host's length
    baseline across the three? n = 3 hosts is too few to fit, but the ordering is checkable.
  - A third host with a *low* length baseline would be the sharpest test of whether this is about
    Llama or about headroom.

- **Next Steps:** E3 setup on Llama-3.1-8B L2 — dictionaries, `circuit-tracer`, and the
  error-node-mass reporting the charter requires for every attribution graph.
