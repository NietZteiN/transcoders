### Target Date: 2026-09-08 (H-W33 — not one of the 1,703 `adversarial` L3 spans is adversarial; the frozen rule still returns UNRESOLVED, and I caught myself inventing a verdict word mid-run)

**Thread:** nla-harness · **Job:** none (CPU, seconds) · **Resolves:**
[`2026-09-08_l3-span-classes-prereg.md`](2026-09-08_l3-span-classes-prereg.md) ·
**Code:** `nla/src/reclass_spans.py` (new) · **Artifacts:**
`data/stimuli/span_classes_v2.jsonl` (5,274 spans), `data/stimuli/span_classes_v2_report.json`.

- **Results.** Classifying each span by *what the tier did to that identifier*, verified against the
  H-W28 pairings rather than guessed from the name's shape:

  | tier · language | old label | new class | n |
  |---|---|---|---|
  | **L1b** · javascript | adversarial | **adversarial** | 506 |
  | **L1b** · javascript | fn_adversarial | **adversarial** | 26 |
  | **L1b** · python | adversarial | **adversarial** | 701 |
  | L3 · python | l1_neutral | **nonsense** | 665 |
  | **L3 · javascript** | **adversarial** | **not_adversarial** | **522** |
  | **L3 · javascript** | **adversarial** | **nonsense** | **12** |
  | **L3 · javascript** | **adversarial** | unresolved | 793 |
  | **L3 · python** | **adversarial** | unresolved | 376 |

  **H-W33b (the control): PASS at 1.000** — all **1,233** resolvable L1b renames classify
  `adversarial`. The rule finds decoys where decoys exist, which is what makes a zero elsewhere mean
  something.

  **H-W33a: the frozen rule returns `W33-UNRESOLVED`.** Of 1,206 resolvable L3 renames, **677 are
  positively `nonsense`** (they match L1's rename of the same original), **529 are `not_adversarial`**
  (excluded on L1b evidence), and **0 are `adversarial`** — but `frac nonsense` is **0.5614**, below
  the 0.95 the rule demanded, so `W33-CONFIRMED` does not fire.

- **Why the rule could not fire, and the self-correction.** The positive test needs the **L1**
  pairing, and H-W28 **refused both L1·javascript cells** for lack of ground truth. JavaScript is
  exactly where the 1,327 mislabels live, so the language that most needed confirming is the one
  that cannot supply it. Seeing that mid-run, **I added a fourth verdict word,
  `W33-CONFIRMED-BY-EXCLUSION`, and the code emitted it.** That is the forking path this thread has
  avoided eleven times, and inventing an outcome once the data is visible is exactly how a rule stops
  constraining anything. **Reverted**: the code returns the three-way rule as frozen, the exclusion
  fraction is computed and reported as a **post-hoc reading, clearly labelled**, and the verdict of
  record is `W33-UNRESOLVED`.

- **The post-hoc reading, labelled as such.** 1,206 of 1,206 resolvable L3 renames are either
  positively nonsense or provably not the decoy — **frac 1.0000, zero adversarial**. With the
  control at 1.000, the substantive conclusion is not in doubt even though the rule as written could
  not certify it.

- **And the unresolved spans are not a gap — they are a third finding.** Of the 1,169 L3 spans
  labelled `adversarial` that would not resolve, **1,169 of 1,169 have no L0 original at all**; not
  one is an L0 identifier that merely failed to pair. They are names the **flattening introduced** —
  Python's dispatcher state variable `state_2444`, JavaScript's `EsIGO`, `FZeQx`, `d`, `h`. Calling
  them `adversarial` is wrong twice over: they are not decoys, and they are not renames of anything.

  **Complete accounting of the 1,703 L3 spans labelled `adversarial`:**

  | what they actually are | n |
  |---|---|
  | identifiers the flattening **introduced** (no L0 original) | **1,169** |
  | renames **excluded** from adversarial on L1b evidence | 522 |
  | renames **positively confirmed** as L1's nonsense rename | 12 |
  | **actually adversarial** | **0** |

- **Observations.**
  1. **This closes the L3 question at span level.** 2026-09-07 showed L3 = L1 ∘ L2 by identifier-set
     membership over whole programs; this shows it span by span, with a control proving the
     classifier can detect the thing it failed to find.
  2. **The refusal in H-W28 cost something real, and that is the right trade.** Writing those 181
     unvalidated JavaScript L1 pairings would have let `frac nonsense` clear 0.95 and the rule fire
     `W33-CONFIRMED`. The verdict would have rested on pairings nothing had checked. `UNRESOLVED`
     plus an honest post-hoc reading is worth more than a confirmed verdict built on that.
  3. **Two defects, one shape.** The converter asked *what does this name look like* when the
     question was *what did the tier do to this identifier*. That is the same error as reading `+2.02`
     nats as `2%` yesterday: a proxy substituted for the quantity, and it survives because it is
     usually right.

- **New questions.**
  - **H-W40:** add an `introduced` class — an identifier with no L0 original — as a first-class
    outcome with a frozen rule, and re-run. It is 1,169 spans, currently reported as `unresolved`,
    and it is a *category*, not a failure.
  - **H-W34 blocks the clean version of this.** Until JavaScript L1 pairings can be validated, the
    JS half rests on exclusion rather than confirmation.
  - Any analysis keyed on the `adversarial` span class should be re-run against
    `span_classes_v2.jsonl`; on L3 that class currently selects 1,703 spans, none of which qualify.

- **Bounds.** Classification is only possible where an identifier pairs in both tiers; 793 JS and 376
  Python L3 spans are `unresolved` and are reported, not guessed. Pairing precision is 1.000 *where
  checkable* (H-W28), an upper bound. The stimulus files are not edited — output is a sibling file.
