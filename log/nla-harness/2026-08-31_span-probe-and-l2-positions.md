### Target Date: 2026-08-31 (dispatcher spans ARE encoded; and L2 reads at the prompt, not at the answer)

Two follow-ups to [`2026-08-31_l2-mechanism-results.md`](2026-08-31_l2-mechanism-results.md),
which concluded — from a *correctness* probe — that the L2 signal is static dispatcher complexity.
Both were named there as the next steps.

- **Setup:**
  ```
  span probe  job 360386, CPU, 2 min · nla/src/p1b_span_probe.py
              target = n_dispatcher_spans itself, not correctness. Ridge alpha = 1.0 frozen,
              GroupKFold(5) on snippet_id, mean rho over 28 layers, 200-draw permutation null.
              Baseline = code_chars + n_lines, because span count correlates with code size and a
              probe that predicts spans by counting characters has shown nothing.
  L2 grid     job 360385, h200, 17 min · nla/src/p1b_position_depth.py --tier L2 (the grid was
              hardcoded to L1b; a --tier flag was added). 60 items, 6 positions x 28 layers.
  ```

- **Results:**

  **1. The residual stream encodes dispatcher span count directly — and well beyond code size.**

  | tier | spans | code-size baseline | residual mean ρ | **beats size** | p | verdict |
  |---|---|---|---|---|---|---|
  | **L2** | 7.3 ± 4.4 | +0.5429 | **+0.8842** | **+0.3413** | 0.005 | **BEYOND CODE SIZE** |
  | L3 | 7.3 ± 4.4 | +0.7893 | +0.8723 | +0.0830 | 0.005 | not beyond |
  | L1b | 0, no variation | — | — | — | — | floor behaved |

  **ρ = +0.8842 for a count read off the residual stream**, +0.3413 beyond what code length and
  line count supply. The inference drawn from the correctness probe is now a direct measurement:
  **the model does represent how many dispatcher sites a program has.** L1b, with zero spans
  everywhere, gives the probe nothing to find — the floor behaves as a floor should.

  L3's failure to clear the bar is a baseline artefact, not an absence: stacking renaming on top of
  indirection inflates code size, so the size baseline rises to +0.7893 and there is little room
  left above it. Its residual ρ (+0.8723) is essentially L2's.

  **2. On L2, correctness is readable at the PROMPT and gains nothing at the answer** — the
  opposite of L1b.

  | position | best layer AUC | ρ@L20 | **L1b for comparison** |
  |---|---|---|---|
  | `last_prompt` | **L9 0.8143** | 0.7297 | 0.7522 |
  | `reply_q25` | L10 0.6992 | 0.5969 | 0.5893 |
  | **`reply_q50`** | L19 **0.5370** | **0.4289** | **0.5234** |
  | `reply_q75` | L20 0.7908 | 0.7908 | 0.6618 |
  | `answer_line` | L16 0.7948 | 0.7199 | **0.8299** |
  | `last_token` | L6 0.7955 | 0.7744 | 0.7578 |

  Length-only baseline 0.6675. **Primary (frozen rule): `answer_line` − `last_prompt` at L20 =
  −0.0307** [−0.1647, +0.0978] → **POSITION DOES NOT MATTER**, and here the point estimate is
  *negative*: the answer line is no better than the prompt. Permutation clean (null 0.4797 ± 0.1028,
  p = 0.015).

  **The tiers dissociate exactly as the static-complexity account predicts.** On **L1b** nothing is
  readable at the prompt beyond length and the signal appears only once the answer is committed
  (0.6685 → 0.8229) — late readout. On **L2** the signal is present at the prompt (0.8143, well
  above the 0.6675 length baseline) and does **not** improve at the answer. A static property of the
  code is available before the model computes anything, and needs no reasoning to become readable.

  **The mid-reasoning collapse replicates on a second tier.** `reply_q50` is the trough on L2 too
  (0.5370, and 0.4289 at layer 20 — below chance), matching L1b's 0.5234/0.4040. That is now the
  fifth independent observation of the same region being empty: `rt_cos` over `reasoning`
  (p = 0.59), the L1b binary probe, the L1b graded probe (ρ = +0.0075), and now the L2 grid.

- **What worked / hypothesis verdict:**
  - **✓ Dispatcher spans are encoded beyond code size** (+0.3413, p = 0.005). The mechanism entry's
    inference is confirmed by direct measurement.
  - **✗ Position does not matter on L2**, with a negative point estimate — the opposite pattern to
    L1b, and the one the static account predicts.
  - **Taken together: what the model represents about dispatcher indirection is a property of the
    code as written, available at the prompt, not a state that develops while it reasons.** No
    evidence of hidden-state simulation on the tier where the charter expected it.

- **Observations:**
  - **This is a shallow representation, and it should be described as one.** ρ = 0.884 for a count
    is close to "the model can count salient repeated structures in its input", which transformers
    do. It is a real internal representation of structural complexity, tied to a documented
    behavioural effect (r = −0.196), and it is **not** semantic understanding of control flow.
    Reporting it as the latter would be exactly the interpretability illusion the charter warns
    about.
  - **The two routes now have two different, fully characterised failure signatures.** Atom route:
    nothing readable until the answer is decided; the residual stream is redundant with a token
    count. Relational route: static structure readable immediately; nothing added by reasoning.
    Neither involves item-level state.
  - **Five observations of the empty mid-reasoning region, across two tiers and three instruments.**
    Whatever the model does between reading the prompt and emitting an answer, none of it is
    linearly decodable as correctness. That is now one of the better-supported claims in the
    programme and it has never been the target of an experiment.

- **New questions / new hypotheses:**
  - **Probe the mid-reasoning region for something other than correctness.** Five nulls for
    correctness do not mean the region is empty — they mean it does not encode *that*. Intermediate
    values, current hop, or partial results are the natural targets, and L2 items have a known
    dispatcher structure to key them to.
  - **Does span encoding survive controlling for token repetition?** The dispatcher object repeats a
    lexical pattern N times; a count probe may be reading repetition frequency. A control with
    matched repetition and different span counts would separate these.
  - The L3 baseline problem is worth fixing: match L2/L3 on code size before comparing, or use a
    size-residualised target.

- **Next Steps:** mid-reasoning probes for non-correctness targets; repetition control on the span
  result. P0.3-ext still blocked on `adversarial_rename`.
