### Target Date: 2026-09-03 (L results — `L-CLEAN`; and the renaming dose doesn't predict the cost at all)

Result for the rule frozen in [`2026-09-03_length-confound-prereg.md`](2026-09-03_length-confound-prereg.md)
(committed `ef246c9` **before** this ran). No GPU.

- **Hypotheses / what we're testing:** **H-L1** — obfuscation cost rises with the amount of renaming
  (predicted ρ ≈ +0.3…+0.5). **H-L2** — given the amount of renaming, does the extra *token length*
  do independent work? CONFIRM if |ρ_partial| ≥ 0.25 with CI excluding 0.

- **Setup:** `nla/src/length_confound.py` (sha256 38b80e706a11…), CPU, env `nla-mi`, seed 20260724,
  10,000-resample bootstrap and permutation, BH-FDR across the two primary tests. n = 60.
  Readout G = the R2 clean-half unit (length-invariant: both terms share the |y_clean| denominator).
  Output `data/nla/p0/trace_llr/gemma12b/length_confound.json`.

- **Results:**
  - Descriptives: mean G **+0.3071**, median Δ **+27** tokens, median **5** renames per item, median
    inflation **+6.00 tokens per rename**. The two predictors are tangled as expected,
    Spearman(Δ, n_renames) = **+0.447**.
  - **H-L1: ρ(G, Δ) = −0.166**, CI95 **[−0.402, +0.087]**, p = 0.206, q = 0.228.
  - **H-L2: ρ_partial(G, inflation | n_renames) = −0.159**, CI95 **[−0.420, +0.123]**, p = 0.228,
    q = 0.228 — |ρ| below the 0.25 boundary **and** CI spanning zero.
  - Secondary, pre-labelled underpowered (6 flips): ρ(Δ, flip) = +0.169;
    partial(inflation, flip | n_renames) = **+0.001**.

- **What worked / hypothesis verdict:**
  - **H-L2 ✗ → `L-CLEAN`.** Token inflation does no independent work. Per the frozen decision table
    this **drops H-T3** (the length-matched corpus fork) — it has no motivation beyond T itself.
  - **H-L1 ✗, and wrong in the direction I predicted.** I pre-registered ρ ≈ +0.3…+0.5; the estimate
    is **−0.166** with a CI spanning zero. **The amount of renaming does not predict what the
    obfuscation costs the model.** Recording the failed prediction rather than quietly reporting the
    null: this was a genuine miss, and it is the more interesting of the two results.
  - **Reassuring for Papers 2–3:** the token-count confound flagged yesterday exists in the stimuli
    (median +27 tokens) but is **not** detectably driving this instrument's readout. That bounds the
    worry; it does not license ignoring it in accuracy-based or attention-based measures, which were
    not tested here.

- **Observations:**
  - If L1b's cost were a dosage effect — more renamed identifiers, more damage — H-L1 would be
    strongly positive. It is flat-to-slightly-negative. That is consistent with the Papers 2–3
    framing of L1b as a **semantic** trap whose bite depends on the *plausibility of the specific
    decoy*, not on how many identifiers were touched. It also means per-item variance in G is
    carried by something we are not measuring, which is worth naming as an open question rather than
    leaving as residual.
  - **I am reversing yesterday's recommendation to run H-T5**, on arithmetic I should have done then:
    the 355 length-matched replace blocks hold **403 tokens across 60 items** — a median of ~6–7
    exactly-alignable identifier tokens against a median of **56** identifier tokens per item, i.e.
    **~12 %** of them. If the corruption is distributed across identifiers at all, patching 12 % of
    them recovers ~12 % of the gap and lands below any sensible threshold, so **H-T5 returns a null
    by construction**. Its only informative version — exact patching vs mean-broadcast patching at
    matched token count — validates the *method*, not the science question. Cheap is not the same as
    worth running.
  - **T therefore ends here on this corpus**, with the requirement stated for anyone who picks it up:
    token-level activation patching between obfuscation tiers needs stimuli that are token-length
    matched **by construction**, and this corpus cannot be retrofitted into that.

- **New questions / new hypotheses:**
  - **H-L3:** what *does* carry the per-item variance in G, if not renaming dose? Candidates already
    on disk: decoy-term plausibility (`gloss_decoy` / `terms_decoy`), identifier frequency in
    pretraining-like corpora, and snippet length. All testable with no GPU on the banked 60.
  - **H-L4:** H-L1's flat result is a claim Papers 2–3 can test at scale on Dataset B's 250 snippets
    with accuracy rather than likelihood, where n and the event count are not the binding constraint.
    That is the properly-powered version of the question this entry could only gesture at.
  - Retired: **H-T3** (no independent motivation) and **H-T5** (null by construction).

- **Next Steps:**
  1. Close the patching/attribution line on this corpus; record the length-matched-stimuli
     requirement in [`../../docs/CHECKLIST.md`](../../docs/CHECKLIST.md) as a prerequisite for E3.
  2. H-L3 is the cheapest live question left in this thread and needs no GPU.
  3. Unchanged: the 6/60 flippable denominator gates every accuracy-level claim; growing the corpus
     from Dataset B remains the prerequisite for any rescue experiment.
