### Target Date: 2026-08-30 (the tier positive does not survive its own baseline — retracted)

Closes the limitation flagged in
[`2026-08-30_p1b-readout-controls.md`](2026-08-30_p1b-readout-controls.md), and **retracts that
entry's headline positive.** That entry is append-only and stands as written, including the
limitation that predicted this; the correction is here.

- **Setup:**
  ```
  new     steer_run.py --baseline-only --max-new-gen 2048 (job 359528, h200 g-07-06, 00:21:25)
          records l0_reply_chars and l1b_reply_chars; `reply_chars` keeps its old meaning (L1b)
          so banked files stay readable. --baseline-only added because the cheapest previous way
          to collect lengths was a full 900-generation run.
  labels  the established 10-draw consensus, so numbers stay comparable with the retracted entry
  probe   Ridge, alpha = 1.0 frozen · GroupKFold(5) on snippet_id · graded k/10 target
  ```

  **The comparison is "beats its own length baseline", not "higher rho".** The tiers differ in
  accuracy and in how much their reply lengths vary, so raw rho is not comparable across them.
  What is comparable is how much the residual stream adds *over a token count within each tier*.

- **Results:**

  | tier | length baseline ρ | residual argmax | **beats length by** | mean reply |
  |---|---|---|---|---|
  | **L0 (clean)** | **+0.4770** | L20 +0.5201 | **+0.0431** | 1,606 chars |
  | **L1b (renamed)** | **+0.3203** | L26 +0.3818 | **+0.0615** | 1,799 chars |
  | contrast | | Δ raw ρ +0.1383 | **Δ = −0.0184** | |

  **The entire raw-rho gap was a length-baseline gap.** Reply length predicts correctness far
  better on clean code (+0.4770) than on renamed code (+0.3203), a difference of **+0.1567** —
  which is almost exactly the +0.1383 raw rho gap reported yesterday. Once each tier is measured
  against its own baseline, the residual stream adds **+0.0431 (L0)** and **+0.0615 (L1b)**, and
  the renamed tier adds slightly *more*.

  **VERDICT: NO TIER DIFFERENCE IN INCREMENTAL VALUE.** The claim "adversarial renaming degrades
  the readability of the model's own correctness from its residual stream" is **withdrawn**. It was
  an artefact of comparing L0's rho against L1b's baseline, which is exactly the substitution the
  limitation note warned about — the note was right and the headline should have been held until
  this ran.

- **What worked / hypothesis verdict:**
  - **The tier positive ✗ RETRACTED.** There is no evidence that obfuscation degrades what the
    residual stream carries about correctness at this site.
  - **The `last_prompt` ≈ length finding is CONFIRMED and generalised.** It now holds on **both**
    tiers: the residual stream at the final prompt token adds **+0.04 to +0.06** over a token
    count, clean or obfuscated. Yesterday's figure of +0.010 for L1b used a length baseline from a
    different generation regime; measured consistently it is +0.0615, still negligible.
  - **A real tier effect exists, but it is behavioural, not representational:** reply length is a
    much better correctness predictor on clean code (+0.4770) than on renamed code (+0.3203). The
    model's *verbosity* tracks its correctness less well once identifiers are adversarial. That is
    a finding about generation, not about the residual stream, and it is measured rather than
    inferred.

- **Observations:**
  - **This is the second time in two days a control reversed a conclusion, and both times the
    control was cheap and the conclusion was attractive.** The depth gradient fell to a second
    label draw; the tier positive fell to a missing baseline that had been explicitly flagged. The
    pattern is clear enough to state as a rule: **in this programme, a positive that has not yet
    been measured against its own baseline is not a positive.**
  - **The week's tally is now honest: no positives.** Phase 0 closed on three pre-registered
    negatives, the read-depth question is uninformative, position is null, the readout signal is
    late readout, and the tier effect is a baseline artefact. What the week did produce is a much
    sharper *statement* of the central negative and four measurement tools — graded labels, the
    reproducibility floor, the dense probe, and per-tier baselines — that the earlier work lacked.
  - **`last_prompt` ≈ length, on both tiers, is the finding.** Every item-level read in this
    programme was taken there. That single sentence now explains B4, B5, N11, N13 and P0.4 without
    invoking anything about beliefs at all.

- **New questions / new hypotheses:**
  - **Why does reply length predict correctness so well, and why less so under renaming?**
    +0.4770 is larger than any internal measure in the programme. If the readout paper needs a
    positive, the honest one may be behavioural: length, answer instability (AUC 0.869), and
    parse failure are all cheap, all beat the residual stream, and their *degradation* under
    obfuscation is itself a result.
  - The L2/L3 tier probe is still unrun and is now the only remaining representational question
    with a plausible positive.
  - **Recompute the 2026-08-30 graded-label numbers against per-tier baselines** where they were
    quoted against +0.3626; the conclusions there do not change sign, but the margins were
    computed against a baseline from a different generation regime.

- **Next Steps:** L2/L3 tier probe; consider reframing the readout paper's positive as behavioural.
  P0.3-ext still blocked on `adversarial_rename`.
