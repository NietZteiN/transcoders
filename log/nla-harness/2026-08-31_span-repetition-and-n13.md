### Target Date: 2026-08-31 (repetition explains most of the span result; and my N13 flag was half wrong)

Three closing controls, all CPU, all on data already on disk. Two of them cut back claims made
earlier the same day; the third corrects a flag I raised on 2026-08-29.

- **1. The span result is mostly repetition statistics.**

  [`2026-08-31_span-probe-and-l2-positions.md`](2026-08-31_span-probe-and-l2-positions.md) reported
  that the residual stream encodes `n_dispatcher_spans` at ρ = +0.8842, **+0.3413 beyond code
  size**, and flagged in its own Next Steps that a dispatcher object repeats a lexical pattern N
  times so a count probe might be reading repetition frequency. It is.

  | baseline | ρ |
  |---|---|
  | code size only (chars, lines) | +0.5429 |
  | **repetition only** (max token freq, distinct tokens, total tokens, duplicated lines, max line freq) | **+0.8346** |
  | size + repetition | +0.8195 |
  | **residual stream** | +0.8842 |
  | **residual − (size + repetition)** | **+0.0647** (bar +0.10) |

  **VERDICT: NOT BEYOND SIZE AND REPETITION.** Five surface counts of the source text predict span
  count at +0.8346 on their own. The residual stream adds **+0.0647**, under the frozen bar,
  though the permutation p remains 0.005 against chance.

  **So the claim shrinks again, and it should.** "The model represents how tangled a program is"
  becomes: *the residual stream carries a signal about dispatcher count that is largely predictable
  from how repetitive the text is.* That is a surface-statistical representation, not a structural
  one. The chain — correctness → static complexity → repetition counts — is three deflations deep,
  and each step was a control this project's own discipline demanded.

- **2. The residual stream goes quiet about the INPUT mid-reply, not just about correctness.**

  Five nulls for correctness in the mid-reasoning region say that region does not encode *that*.
  `n_dispatcher_spans` is a target known to be encoded at `last_prompt`, so probing for it across
  positions distinguishes "the stream goes quiet" from "it stays informative about the code but
  not the outcome".

  | position | span ρ | beats size baseline (+0.5429) |
  |---|---|---|
  | `last_prompt` | **+0.8879** | **+0.3450** |
  | `reply_q25` | +0.2483 | −0.2947 |
  | **`reply_q50`** | **+0.1773** | **−0.3656** |
  | `reply_q75` | +0.2878 | −0.2551 |
  | `answer_line` | +0.5102 | −0.0116 |
  | `last_token` | +0.6063 | +0.0634 |

  **The same U-shape the correctness probes found, on a completely different target.** Information
  about the input's structure is strongly present at the final prompt token, collapses through the
  reply, and partially returns at the end.

  **The honest reading is deflationary, not mystical.** Activations at reply position *t* represent
  what the model is generating at *t*; there is no reason they should carry the prompt's structure,
  and the KV cache holds that information without it needing to be re-represented. So the
  mid-reasoning null is not "the model stops thinking" — it is that **the residual stream at reply
  positions is about the reply**, which also explains why five correctness probes found nothing
  there. That is a simpler account than any I offered earlier this week.

- **3. My N13 flag was half wrong, and the half that was right is small.**

  On 2026-08-29 I flagged that N13's answer-entropy positive "appears to have been estimated from
  the same generations that supplied its labels" and put re-estimation at the top of the ledger.
  Recomputing directly from `data/nla/n13/answer_entropy{,_hardened}.jsonl`:

  | label source | unhardened | hardened |
  |---|---|---|
  | `banked_correct` — a **separate greedy run**, held out by construction | **0.8400** | **0.8428** |
  | `modal_correct` — plurality of the same 8 samples | 0.8428 | **0.8694** |
  | `any_correct` — same 8 samples | 0.8055 | 0.8222 |

  **The reported 0.843 and 0.869 are the `modal_correct` figures**, so the same-sample dependency
  is real. But entropy comes from 8 samples at T = 0.8 while `banked_correct` comes from an
  independent greedy run, so **a held-out estimate was always available, and it is 0.8400 /
  0.8428.** The inflation is **0.026**, not the 0.18 the instability measure suffered on
  2026-08-29 — because there both quantities came from the same five greedy draws.

  **What does need correcting is a secondary claim.** The normalization entry argued that hardening
  *strengthened* the signal, 0.843 → 0.869, and read that as "a signal getting sharper when label
  noise is removed is what a real effect does". Held out, hardening moves it **0.8400 → 0.8428**,
  a gain of **+0.003**. Most of the +0.026 lived in the label, not the signal.

  **N13's positive stands at ≈ 0.84**, and remains the strongest predictor in the programme.

- **What worked / hypothesis verdict:**
  - **Span encoding ✗ not beyond repetition** (+0.0647, bar +0.10).
  - **Mid-reasoning ✓ quiet about the input too** — and explained by reply positions representing
    the reply, not by anything about reasoning.
  - **N13 ✓ survives held-out re-estimation at 0.8400/0.8428**; its "hardening strengthens it"
    corollary ✗ does not.

- **Observations:**
  - **Five deflations this week, and the last three were self-inflicted by design** — each was a
    control I wrote into a Next Steps list before running it, which is the intended failure mode.
    The chain from "relational route carries item-level information" down to "the residual stream
    partly reflects how repetitive the text is" was walked one control at a time and nothing had to
    be retracted after the first two entries.
  - **I over-flagged N13 and should have checked before writing.** The 2026-08-29 entry asserted the
    dependency on the basis of the ledger's description rather than the data, and put it at the top
    of the priority list. The data was on disk and the check took two minutes.
  - **What survives the week, stated plainly:** theme-level content is represented and item-level
    content is not; the behavioural predictors (answer entropy 0.84, reply length, parse failure)
    beat every internal measure; and the internal measures that looked like exceptions each reduced
    to a surface statistic under control.

- **New questions / new hypotheses:**
  - The mid-reasoning account is now testable rather than mysterious: if reply activations are about
    the reply, then a probe for *reply* properties (what the model is about to say next, partial
    output) should work there where prompt-property probes fail. That would confirm the account
    rather than leaving it as the least-bad explanation.
  - Whether any internal measure in this programme beats a surface statistic remains open, and on
    present evidence the answer is no. Framing that as the paper's result — with the ladder of
    controls as the method — is more defensible than the mechanistic claim the study set out to make.

- **Next Steps:** reply-property probes at mid-reasoning; propagate the N13 correction into
  `00_STATE.md` and the 2026-08-15 normalization entry's forward references.
