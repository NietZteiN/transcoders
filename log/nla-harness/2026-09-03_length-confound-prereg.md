### Target Date: 2026-09-03 (L — is the L1b effect partly token inflation? Rule frozen before running)

Raised by [`2026-09-03_patch-alignment-impossible.md`](2026-09-03_patch-alignment-impossible.md):
adversarial renaming inflates prompts by a median of **+27** tokens, and nothing in Papers 2–3 or in
this thread has controlled for that. **This entry is written and committed before the analysis runs.**
It decides whether the length-matched corpus fork (H-T3) has a motivation independent of T.

- **Hypotheses / what we're testing:** Exploratory-with-predictions; there is no prior estimate to
  power against, so effect sizes below are stated as decision boundaries, not expectations.
  - **H-L1:** per-item obfuscation cost rises with the amount of renaming. Predict Spearman
    **ρ(G, Δ) > 0**, roughly 0.3–0.5.
  - **H-L2 (the one that matters):** *given* how much renaming an item received, does the extra
    **token length** the decoys cost do independent work? CONFIRM (a real confound) if the partial
    correlation of G on per-rename inflation, controlling the number of renames, has a bootstrap CI
    excluding 0 **and** |ρ_partial| ≥ 0.25. REFUTE otherwise.

- **Setup (frozen).** No GPU. Per item *i*, all from banked artifacts, n = 60:
  - **G_i** = [log P(y_clean | x_l0) − log P(y_clean | x_l1b)] / |y_clean|, the clean-half unit from
    R2 (`data/nla/p0/trace_llr/gemma12b/llr_rows.jsonl`, rows `l0prompt|clean` vs `noop#1|clean`).
    Its denominator is common to both terms, so G carries **no mechanical dependence on prompt
    length** — this is why G is the readout rather than a raw logprob.
  - **Δ_i** = len(l1b_prompt_ids) − len(l0_prompt_ids), from `traces.jsonl`.
  - **n_r(i)** = number of entries in the item's `rename_map` — the "amount of renaming".
  - **infl_i = Δ_i / n_r(i)** — the length-only component: extra tokens *per rename*.
  - Spearman throughout (Δ and n_r are counts and skewed). Partial correlation by ranking all three,
    regressing rank(G) and rank(infl) each on rank(n_r) by OLS, and correlating the residuals.
    Percentile bootstrap over items, 10,000 resamples, seed 20260724. Two primary tests → **BH-FDR**
    across the two, per the charter's stats stack.
  - **Secondary, reported but pre-labelled underpowered:** the same predictors against the binary
    L0✓→L1b✗ flip. Only **6** items flip, so this cannot support a claim either way and is recorded
    to prevent it being run later as if fresh.

- **Decision table (frozen):**

  | verdict | condition | consequence |
  |---|---|---|
  | **L-CONFOUND** | H-L2 confirmed | token inflation does independent work → the length-matched tier (H-T3) is justified **on its own merits**, as a control for a confound in the published L1b measure, and T comes along for free |
  | **L-CLEAN** | H-L2 refuted | inflation is a by-product of renaming, not a driver → **drop H-T3**; run H-T5 (patching restricted to the 355 exact blocks) as the bounded end of T |
  | **L-UNINF** | n_r or G degenerate | report and stop |

- **What this cannot settle.** One host, n = 60, one tier, a likelihood readout rather than accuracy.
  `L-CONFOUND` would mean *this instrument's* readout is length-sensitive and would license building
  the tier; it would **not** by itself establish that Papers 2–3's accuracy findings are confounded —
  that needs the tier and a re-run, which is exactly what the verdict would authorise.

- **Next Steps:** implement `nla/src/length_confound.py`, run, log results in a new dated entry.
