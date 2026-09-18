### Target Date: 2026-09-18 (H-R25 / H-R26 — Phi's parse failure is not the generation budget, and NO-PERMITTED-HOST stands)

- **Hypotheses / what we're testing:** H-R25 was raised in
  [`2026-09-18_panel-damage-results.md`](2026-09-18_panel-damage-results.md) as a free diagnostic; H-R26
  was pre-registered in [`2026-09-18_phi-budget-prereg.md`](2026-09-18_phi-budget-prereg.md) with the
  prediction **`PARSE-IS-BUDGET`**, and **`DAMAGE-PRESENT-BUT-SMALL` at ≈ +0.047** conditional on it.

- **Setup:** `microsoft/Phi-3.5-mini-instruct`, greedy, same 146 PACKS-PAIRED snippets / 1 722 cases, same
  packs, chat template, `unsteered` only. Jobs 411240 / 411241 (g-06-01, g-07-09), ~17 min GPU total.
  Only `--max-new-tokens` changed: **512 → 1536**.

- **Results:**

  **H-R25 `FORMATTING-LIMITED` (exploratory, on banked rows).** L0 condition, greedy:

  | model | acc | parse | **acc given parsed** |
  |---|---|---|---|
  | `codegemma-7b-it` | 0.8124 | 0.9988 | 0.8134 |
  | `Llama-3.1-8B-Instruct` | 0.8200 | 0.9849 | 0.8325 |
  | **`Phi-3.5-mini-instruct`** | **0.5639** | **0.7677** | **0.7345** |
  | `CodeLlama-13b-Instruct` | 0.7631 | 0.9483 | 0.8047 |
  | `CodeLlama-7b-Instruct` | 0.8507 | 1.0000 | 0.8507 |

  Phi answers at **0.7345 when it answers at all** — far above the 0.500 chance floor — so its 0.5639 is
  a *formatting* ceiling, not a capability floor. A second exploratory cut, damage restricted to cases
  **both** conditions parsed, found Phi's deficit survives: **+0.0469 [+0.0181, +0.0763]** from
  0.7284 → 0.6815. **Both cuts are post-hoc** and were used only to generate H-R26.

  **H-R26a `PARSE-NOT-BUDGET` — prediction REFUTED.**

  | budget | condition | acc | parse | acc\|parsed |
  |---|---|---|---|---|
  | 512 | L0 | 0.5639 | 0.7677 | 0.7345 |
  | **1536** | L0 | 0.5679 | **0.7689** | 0.7387 |
  | 512 | deranged | 0.5093 | 0.7509 | 0.6783 |
  | **1536** | deranged | **0.5093** | **0.7509** | **0.6783** |

  Tripling the budget moved L0 parse by **+0.0012**, and the deranged condition is **identical to four
  decimal places on all three statistics** — i.e. not one snippet in that condition was being truncated.
  The bar was ≥ 0.95; it returns 0.7689.

  **H-R26b NOT READ**, per the gate frozen in the pre-registration. The damage at 1 536 tokens was
  computed into `stats/2026-09-18_r26_phi.json` for the record but is **not reported and carries no
  verdict**, because its precondition failed.

- **What worked / hypothesis verdict:**
  - **H-R25 SUPPORTED** — Phi is formatting-limited, and that is worth knowing independently: the
    charter's panel includes Phi-3.5-mini for the main study, where a 23 % silent answer-loss would be
    attributed to the model's reasoning rather than to its output format.
  - **H-R26a REFUTED my prediction.** I nominated truncation and cited the length↔parse correlation
    (−0.273, the strongest in the panel) and the fact that partially-parsed snippets carried more cases
    and longer replies. **That evidence was consistent with truncation but did not distinguish it from
    the simpler reading — that Phi's longer/denser problems are just the ones it formats worst.** The
    budget test separates them cleanly, and the simpler reading wins.
  - **`NO-PERMITTED-HOST` stands unqualified.** The one cell that could have overturned it was withheld
    by the floor gate, and the attempt to rescue that cell has failed on its own pre-registered
    precondition. The ASE line stays closed.
  - **Determinism re-confirmed for free:** the deranged condition's numbers are bit-identical across two
    independent runs at different budgets — an incidental third confirmation of H-R18a.

- **Observations:**
  - **The discipline mattered twice in one experiment.** The floor gate (frozen in H-R23) stopped a
    near-chance model from being reported as a host; the H-R26a precondition (frozen this morning) then
    stopped a *failed* rescue from being quietly converted into "well, the damage is +0.047 anyway". Both
    were written before the relevant data existed. Without either, this block would now be claiming a host.
  - **What Phi's parse failure actually is remains unknown.** It is not budget. The raw reply text is not
    banked (`reply_chars` only), so distinguishing prose-instead-of-JSON from a divergent JSON shape from
    partial refusal needs a re-run with text capture. Given `NO-PERMITTED-HOST` is settled, that is
    **not worth GPU now** — but it is a real gap in the harness: **no run in this family can diagnose its
    own parse failures after the fact.**
  - Cost note: H-R25 was free (banked rows) and H-R26 was ~17 min. The expensive thing in this whole
    sequence was never the compute.

- **New questions / new hypotheses:**
  - **H-R27 (harness debt, cheap):** bank a truncated copy of the raw reply (say first/last 400 chars) for
    every run. Every parse-rate finding in this thread — H-R9's phantom keys, H-R18's compliance story,
    H-R25's formatting ceiling — has had to be inferred indirectly because the text is discarded.
  - **H-R24 unchanged and now fully unblocked as a decision:** the renaming route is exhausted on every
    permitted model, and the rescue attempt failed. Build an L2/L3 flattening generator, or write the ASE
    line up as a bounded negative replication. **Still a scope call for the human.**

- **Next Steps:** none in the ASE block without H-R24. H-R27 is cheap and would pay for itself in any
  future run.
