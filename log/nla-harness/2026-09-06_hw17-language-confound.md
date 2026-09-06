### Target Date: 2026-09-06 (Filed BEFORE the result: H-W17's subset is a language split, and my frozen rule underweights it)

Addendum to [`2026-09-06_meaning-clean-prereg.md`](2026-09-06_meaning-clean-prereg.md). **Append-only.**
Written while job **379908** is still queued — **no H-W17a number exists**, only the H-W17b
representativeness statistics the prereg froze, computed on CPU from banked data.

- **H-W17b, run as pre-registered** — the 20 anchored items vs the 29 others, Mann–Whitney, BH-FDR
  across the five frozen properties:

  | property | anchored (n=20) | others (n=29) | p | q |
  |---|---|---|---|---|
  | `G_sum` unit | 115.13 | 109.12 | 0.324 | 0.540 |
  | L0 accuracy | 0.800 | 0.759 | 0.746 | 0.746 |
  | prompt tokens | 224.3 | 265.1 | 0.063 | 0.157 |
  | spans per item | 7.60 | 7.69 | 0.681 | 0.746 |
  | **is JavaScript** | **0.05** | **1.00** | **1e-9** | **0.000** |

  **1 of 5 differs at q < 0.05, so the frozen rule does NOT flag the result corpus-bounded.**

- **The frozen rule is satisfied and is nonetheless the wrong instrument here, which I am recording
  before the estimate exists rather than after.** The single difference is not a shift in a
  distribution — it is an **almost perfect partition**: the anchored items are **19/20 Python**, the
  others are **29/29 JavaScript**. Whether L1 renaming is thorough enough to anchor a span is, on
  this corpus, **very nearly the same variable as the language**.

  A "≥ 2 of 5 properties" rule counts *how many* dimensions differ and is blind to *how completely*
  any one of them does. Two mild shifts would have tripped it; one total separation does not. That
  is a defect in the rule I wrote, not in the data.

- **Consequences, fixed now.**
  - **H-W17a will be a Python result.** It must be reported as *"on 20 Python items"*, never as
    *"on 20 of the 49 items"*, in this ledger and in anything downstream.
  - **The nonsense-vs-truth question is unanswerable for JavaScript on this corpus.** Not
    under-powered — **unaskable**: 0 of 29 JavaScript items carry a single `rename_map`-anchored
    span. Any future claim about L1 vs L1b mechanisms is Python-only until the stimuli change.
  - **The rule stands as frozen.** H-W17a's verdict will be reported as the rule computes it, with
    this entry cited beside it. **I am not retuning a threshold after seeing the data**; I am
    recording that the threshold measures the wrong thing, which is a different act and one this
    thread has had to perform six times now.
  - **The other W results are not affected in the same way.** The battery, dose and H-W16a arms all
    ran on the full 49-item mixed-language set; only the L1 arms inherit this partition.

- **New question this raises about the stimuli themselves (H-W19):** why does L1 renaming record a
  `rename_map` for Python and not for JavaScript? If it is a generation-pipeline asymmetry rather
  than a property of the languages, then **the L1 tier is not comparable across languages anywhere
  in the study** — which would reach Papers 2–3's L1 conditions, not just this thread. Worth an hour
  of CPU on the stimulus builder before any cross-language L1 claim is made.
