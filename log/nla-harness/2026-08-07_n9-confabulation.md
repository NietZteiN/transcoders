### Target Date: 2026-08-07 (N9 — the confabulation rate, measured; and no evidence of planning)

- **Hypotheses / what we're testing:** **EXPLORATORY, not pre-registered, no FDR, no confirmatory claim.** Two questions, both prompted by the user asking whether any further analysis was worth running and whether interesting individual readings could be surfaced.
  1. The project has carried one qualitative statement since 2026-08-04 — *"the readings are reliable about themes and confabulate specifics"* — that has never been measured. **It turns out to be mechanically checkable.**
  2. Do readings show **look-ahead / planning** — naming content the trace has not written yet?

- **Setup:** `src/analysis/n9_confabulation.py` (rate) and `src/analysis/n9_examples.py` (candidate mining), env `transcoders-mi`, **CPU only**, seed 20260724. Corpus = the merged 9,511 readings (banked 5,090 + dense 4,421). Outputs `data/nla/n9/2026-08-07/{confabulation,candidates}.json`.
  - **Why this measurement is unusually trustworthy:** every case has a ground-truth language from the stimulus, and the readings name a language *unprompted* in 91% of cases. So "did it name the right language?" is a per-reading correctness label needing **no judge, no annotation and no model** — a rare thing in this project, and the reason it carries more weight than anything the judge produced.
  - The verbalizer never sees the code, only a 3,584-number activation, so a named language is always an inference, never a copy.

- **Results:**
  - **91.1%** of 9,511 readings name a programming language. **37.2%** of those name the wrong one.
  - **The asymmetry is the finding — it is a prior, not noise:**

    | task's real language | readings naming one | wrong | what they claim |
    |---|---|---|---|
    | python | 4,687 | **1.0%** | python 4,642 |
    | javascript | 3,973 | **79.9%** | **python 3,162** · javascript 798 |

    On JavaScript code the readings say "Python" 3,162 times against 798 "JavaScript". The language field of a reading carries **almost no information** about the code it was taken from.
  - **Where it goes wrong is structured**, and this is the mechanism:

    | read at | wrong-language rate |
    |---|---|
    | a token of the code | **12.2%** |
    | a dispatcher variable | 12.4% |
    | the answer line | 43.1% |
    | the model's own reasoning prose | **46.0%** |

    The claim tracks **whether literal syntax sits at the read position**, not what language the task is in.
  - **A hypothesis of mine that failed, worth recording:** the top confabulation examples all sat at `u ≈ 0.00–0.01`, so I predicted the error rate would *decay* with position as language-specific evidence accumulated. It does the opposite — on JavaScript cases it **rises** 85.5% → 91.8% across deciles (r = +0.12). Deeper into a trace the model writes language-agnostic prose, so there is *less* syntax to go on, not more. The by-kind split above is the correct explanation; the positional story was wrong.
  - **Does faithfulness catch it? Barely.** Wrong-language rate by `rt_cos` quintile: **43.9 → 37.4 → 36.3 → 37.0 → 31.4%**. Correlation −0.087; Q5−Q1 gap −12.5 points (95% CI −18.6 to −6.5, case-clustered). Real, small, and nowhere near sufficient: **the best-reconstructed fifth of readings still names the wrong language 31% of the time.** This is the sharpest available demonstration that a high recovery score means *the sentence carried the vector*, not that the sentence is true — the caveat printed all over the artifact, now measured.
  - **Look-ahead / planning: no evidence.** A first detector found 317 readings naming words that appear only later in the trace. But it conflated planning with **sentence completion** — a read on `" to"` inside *"…equal to| the maximum value"* trivially "anticipates" the next three words. Requiring the anticipated word to first appear **≥400 characters later** cut it to 43. Those 43 then failed a null: does a reading from a *different case* anticipate this trace just as well?

    | | words named ≥400 chars ahead |
    |---|---|
    | the case's own reading | **0.548** |
    | a foreign reading (null) | **0.532** |
    | excess | **+0.016**, 95% CI [−0.022, +0.053] |

    The ≥3-word rate is actually *lower* for own readings (2.0% vs 2.2%). The apparent look-ahead was generic reasoning vocabulary — "evaluate", "final", "check" — that any trace eventually contains. **The 43 candidates were not shipped.**
  - **Four readings curated for the artifact**, surfaced by the judge and then read by hand (the judge is unreliable per item, κ 0.05, so it selects candidates and never decides). Two are substantive **contradictions** — the flagship `JavaScript/63:L1b` step describes a *three-term* recursion while the reading calls it an *iterative factorial* with a two-case base; a `cruxeval/113` step transforms characters by position while the reading reports a binary conversion of a different string. Two are **right-shape / invented-particulars** — a slicing step `theta = alpha + value` (40+7=47) read as an assignment using `:=` with `x + y = 10 + 5`, and an answer-line step for `rounded_avg` read as `calculate()`.

- **What worked / hypothesis verdict:** N/A (exploratory). The long-standing qualitative claim is now quantified: **themes reliable, specifics inferred from a prior.** The planning question gets an explicit, null-controlled **negative**.

- **Observations:**
  1. **The best measurement in this project needed no model.** N7 spent two GPUs and 5,653 judged comparisons to produce a score that could not rank a single item. A regex against a known ground-truth field produced a labelled per-reading correctness signal on all 9,511 for free. Worth remembering when the next instrument is designed: look for a mechanically checkable claim *inside* the output before reaching for a judge.
  2. **The same null discipline that killed HT14 killed the planning story**, and it took ten minutes. Any "the reading anticipates X" claim needs a foreign-reading control, because generic reasoning vocabulary appears everywhere.
  3. Two of my own hypotheses failed inside this one entry (confabulation-decays-with-position; look-ahead-is-real). Both were cheap to test and both would have been plausible-sounding paragraphs if I had not tested them.
  4. **This sharpens what the NLA is good for.** It is trustworthy about *what kind of operation* is happening — a walkthrough, an assignment, a base case, committing to an answer — and untrustworthy about every particular: language, function names, operators, values. Any downstream use (E7 triangulation especially) must key on the former and ignore the latter.

- **New questions / new hypotheses:** (a) Is the Python prior a property of the *verbalizer* or of the *subject model's residual stream*? Reading the same JavaScript code through a JavaScript-heavy prompt would separate them. (b) Does the same prior show up in non-language specifics — are invented function names drawn from a similarly narrow distribution? `calculate()`, `my_function`, `is_prime` recurred across unrelated cases, which suggests yes and would be cheap to check. (c) The 12% floor on code-token reads is not zero — what are those?

- **Next Steps:** none required; the N4–N8 programme is complete and this is a labelled exploratory addendum. If the line continues, (a) above is the sharpest follow-up and needs one generation run.

---

#### Addendum, same day — two ad-hoc results made reproducible, and the page finished

No new compute: the box's GPUs are busy, so the follow-up run proposed above was **not** started.
Consolidation only, all CPU.

- **Two results in this entry had been computed in throwaway shell commands** and their numbers
  quoted here, which the charter's provenance rule does not allow. Both now live in version control
  and reproduce identically under the same seed:
  - [`../../src/analysis/n9_lookahead_null.py`](../../src/analysis/n9_lookahead_null.py) — the
    foreign-reading null. Re-ran: own **0.5481** vs foreign **0.5319**, excess **+0.0161**
    (95% CI −0.0217 to +0.0533), ≥3-words rate 2.00% own vs 2.23% foreign →
    `NO EVIDENCE OF LOOK-AHEAD (null not beaten)`. Output `data/nla/n9/2026-08-07/lookahead_null.json`.
  - the **positional breakdown** is now emitted by `n9_confabulation.py` rather than recomputed by
    hand. On the dense readings the JavaScript error rate runs 82% → 92% across deciles,
    **r = +0.134** — confirming the direction, and confirming that my "prior gets corrected as
    evidence accumulates" prediction was backwards.
- **Page changes** (10 figures now, 10.64 MB):
  - the planning claim is **shown, not asserted** — two bars, own vs foreign, deliberately the same
    height, with the interval printed beneath;
  - a card titled *"A prediction of mine that was wrong"* carries the positional result, because the
    failed prediction is what identified the right explanation (the code-vs-reasoning split);
  - a card titled *"What would settle this"* states the open question — is the Python default a
    property of **the reader** or of **the model being read**? — says plainly that nothing measured
    separates them, and records that the run is **not started because the GPUs are busy**. The
    limitation belongs on the page rather than in a file nobody opens.
- **A third instance of the same clipping bug**: the null figure's footnote overran its viewBox, as
  the heatmap's had. Both now render in HTML beneath the SVG. Worth generalising — any explanatory
  line longer than a few words does not belong inside a fixed viewBox.
- All three regression tests pass; 10 figures parse; JS clean.
