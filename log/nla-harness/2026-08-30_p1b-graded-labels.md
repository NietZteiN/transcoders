### Target Date: 2026-08-30 (graded labels — the prompt site is worth a token count, and the answer-line signal is readout)

Follows [`2026-08-30_p1b-position-depth-results.md`](2026-08-30_p1b-position-depth-results.md),
which retracted a depth gradient after binary labels differing on 8 of 60 items produced opposite
curves. Exploratory analysis of already-inspected data — **not** a hypothesis test — except for
one rule frozen before running: the split-half identifiability criterion.

- **Setup:**
  ```
  target  k/10, the fraction correct across the TEN independent unsteered generations already on
          disk (P0.4 L06/L13/L20 + repro A1/A2/B1/D1/D2/E1/E2). Zero new GPU time.
          mean 0.550, sd 0.451 · 19 items never right, 26 always, 15 in between
  model   Ridge, alpha = 1.0 frozen (not tuned; p >> n at d = 3584, n = 60)
  cv      GroupKFold(5) on snippet_id · out-of-fold Spearman · length-only baseline
  frozen  SPLIT-HALF IDENTIFIABILITY: partition the 10 draws into two disjoint 5s, build a curve
          from each, repeat 20x. IDENTIFIABLE iff mean curve Spearman >= 0.70 AND argmax agrees
          on >= 50% of splits. 0.70 is the same reliability line used for N10b on 2026-08-29.
  ```

- **Results:**

  **The length baseline is the number everything must beat: ρ = +0.3626.**

  | position | mean ρ | ρ@L20 | best layer | vs length |
  |---|---|---|---|---|
  | `last_prompt` | +0.2895 | +0.3253 | L26 **+0.3730** | **+0.010 — nothing** |
  | `reply_q25` | +0.1245 | +0.1794 | L11 +0.2669 | worse |
  | **`reply_q50`** | **+0.0075** | −0.0704 | L27 +0.2247 | worse |
  | `reply_q75` | +0.2433 | +0.3557 | L16 +0.4243 | +0.062 |
  | **`answer_line`** | **+0.3948** | +0.4665 | L25 **+0.5702** | **+0.226** |
  | `last_token` | +0.3694 | +0.3978 | L26 +0.5469 | +0.184 |

  **Three findings, in order of how much they matter.**

  **1. At `last_prompt` the residual stream is worth about as much as counting tokens.** Best layer
  ρ = +0.3730 against a length-only baseline of +0.3626 — an advantage of **+0.010**. **Every
  item-level read in this programme has been taken at this site**: B4's vectors, P0.1, P0.2, P0.4,
  and both probes this week. That is a clean, quantitative explanation for the entire ledger of
  item-level nulls. They were not measuring a model without item-level content; they were reading
  at a position where a linear decoder does no better than reply length.

  **2. Mid-reasoning carries nothing, now on a third instrument.** `reply_q50` mean ρ = **+0.0075**,
  and −0.0704 at layer 20. Prior evidence: `rt_cos` across the `reasoning` class adds nothing over
  length (p = 0.59, 2026-08-29); the binary probe gives AUC 0.52 there (2026-08-30). Three
  measurements, three nulls, same region.

  **3. The answer-line signal is real — and the evidence favours the sceptical branch.** ρ = +0.5702
  at L25 against a +0.3443 length baseline at that position, and its depth curve rises
  monotonically (+0.19 at L0 → +0.5 from L17 on). But the pre-registered branch from
  [`2026-08-30_p1b-position-depth-prereg.md`](2026-08-30_p1b-position-depth-prereg.md) asked
  whether this is **EARLY KNOWLEDGE** or **LATE READOUT ONLY**, and the deciding evidence is
  `reply_q50`: if the model represented its eventual correctness *before* committing, mid-reasoning
  would carry signal. It carries **+0.0075**. **The answer-line signal is most consistent with
  reading a decided answer rather than predicting one, and no predictive claim is licensed.**
  This is the branch the pre-registration wrote down first precisely because the alternative was
  more attractive.

  **Graded labels substantially stabilise the curve — but not its peak.**

  | position | curve Spearman between halves | argmax agreement | verdict |
  |---|---|---|---|
  | `last_prompt` | 0.7873 ± 0.1008 | **0.00** | **NOT IDENTIFIABLE** |
  | `answer_line` | **0.9435** ± 0.0232 | 0.50 | **IDENTIFIABLE** |

  Against binary labels, which gave *opposite* curves, graded labels give split-half correlations
  of 0.79–0.94. So the broad shape does replicate; what does not, at `last_prompt`, is the location
  of the maximum — argmax agreed on **0 of 20** splits. That is the signature of a **flat top**
  (the curve sits between +0.32 and +0.37 across L16–L27), not of a noisy measurement, and it
  sharpens yesterday's retraction: the shape is recoverable, the peak layer is not.

  **Permutation control**, `last_prompt`/L20: observed +0.3253, null **+0.0203 ± 0.136**,
  max +0.4014, **p = 0.0199**. Real but modest — and once again a single null draw (0.40) exceeds
  the observed value, the third time this week that a distribution was necessary.

- **What worked / hypothesis verdict:**
  - **The graded target is the right instrument** and it cost nothing: it turned a statistic whose
    argmax flipped between draws into curves correlating 0.79–0.94 across independent halves.
    **Ten cheap label draws beat one careful one.**
  - **`last_prompt` ≈ length** (+0.010). The incumbent read site adds essentially nothing over a
    token count.
  - **Mid-reasoning is empty** on three independent measures.
  - **LATE READOUT ONLY** is the supported reading of the answer-line effect. The readout paper's
    most attractive hypothesis — that the model knows early and we were reading at the wrong place
    — is **not** supported: we were reading at the wrong place, but there is nothing to read
    earlier either.

- **Observations:**
  - **This reframes the programme's central negative without weakening it.** "There is no
    item-level belief at this site to edit" (Phase 0, licensing row 4) can now be stated more
    precisely: at the site everything was measured, a linear decoder ties a token count; the only
    place item-level content is decodable is one token from the answer, where decoding correctness
    and decoding the answer are nearly the same operation.
  - **The `answer_line` finding is the readout paper's positive and its limitation simultaneously.**
    ρ = 0.57 is the largest internal effect in the programme — larger than N10b's, and unlike it,
    it beats its baseline. It just cannot be called prediction.
  - For calibration: reply length ρ = 0.363 here; N13's answer instability gives AUC 0.869 on a
    binary target. A behavioural signal still beats every internal one.
  - The flat-topped `last_prompt` curve explains why P0.1's coherence argmax (L13) was never going
    to survive: the underlying surface has no sharp maximum to find, in coherence or in
    decodability.

- **New questions / new hypotheses:**
  - **Test LATE READOUT ONLY directly.** Probe at `answer_line` for the *emitted answer* rather
    than for correctness. If the answer is decodable at the same ρ, the correctness signal is a
    corollary of it and the interpretation closes. Free — the replies are saved.
  - **Does the `last_prompt` ≈ length result hold on the L0 (clean) tier?** If the residual stream
    beats length on clean code and ties it only under adversarial renaming, that is a statement
    about what obfuscation does to the representation, which is the study's actual subject.
  - The stimulus set remains the binding constraint for anything shape-dependent: 60 items, 15 of
    them label-contested even at 10 draws.

- **Next Steps:** the answer-decodability control above (free); then the L0-tier comparison.
  P0.3-ext still blocked on `adversarial_rename`.
