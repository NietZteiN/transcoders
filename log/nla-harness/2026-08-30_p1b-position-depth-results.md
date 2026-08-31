### Target Date: 2026-08-30 (position × depth — and a retraction of this morning's depth gradient)

Pre-registration: [`2026-08-30_p1b-position-depth-prereg.md`](2026-08-30_p1b-position-depth-prereg.md).
**This entry retracts a finding reported earlier today in
[`2026-08-30_p1b-read-depth-results.md`](2026-08-30_p1b-read-depth-results.md).** That entry is
append-only and stands as written; the correction is here and is the most important thing below.

- **Setup:**
  ```
  job     359321, h200, 00:16:09 · nla/src/p1b_position_depth.py
  items   60 L1b stimuli, fresh greedy generations at max_new_gen 2048 (no censoring)
  acts    one forward pass over prompt+reply · 28 layers x 6 positions x 3584 dims
  probe   logistic, L2, C = 1.0 frozen · GroupKFold(5) on snippet_id · bootstrap over snippets
  extra   nla/src/p1b_consensus_labels.py — 10 independent label draws already on disk
          (P0.4 L06/L13/L20 + repro A1/A2/B1/D1/D2/E1/E2)
  ```

- **Results:**

  **H-PD1 → POSITION DOES NOT MATTER, by the frozen rule.** `answer_line` vs `last_prompt`, both
  at layer 20, paired on the 59 items valid for both:

  | quantity | value | threshold | pass |
  |---|---|---|---|
  | Δ AUC | **+0.1123** | ≥ +0.10 | ✓ |
  | bootstrap CI | **[−0.0303, +0.2564]** | excludes 0 | **✗** |
  | beats length-only | +0.2245 (0.8229 vs 0.5984) | ≥ +0.05 | ✓ |

  Two of three criteria pass and the CI does not. The pre-registration said in advance that n = 60
  with ~28 positives is underpowered and that a null is not evidence of absence; that is exactly
  what happened, and the verdict stands as written rather than being talked around.

  **H-PD2 → NOT EVALUATED**, by the rule's own terms.

  **Permutation control clean.** At `answer_line`/L20: null **0.4752 ± 0.0936**, observed 0.8229,
  p = 0.005, no leak. Note the null's **maximum over 200 draws is 0.7627** — a single permutation
  can reach 0.76 here. That is the second time in two days this has vindicated using a
  distribution rather than one draw.

  **The position surface (exploratory, and the largest effect in the run).** Correctness AUC at
  layer 20, and each position's best layer:

  | position | L20 AUC | best layer | length-only |
  |---|---|---|---|
  | `last_prompt` | 0.6685 | L13 **0.7522** | 0.6116 |
  | `reply_q25` | 0.4632 | L12 0.5893 | 0.6116 |
  | **`reply_q50`** | **0.4040** | L13 0.5234 | 0.6116 |
  | `reply_q75` | 0.5938 | L13 0.6618 | 0.6116 |
  | **`answer_line`** | **0.8229** | L19 **0.8299** | 0.5984 |
  | `last_token` | 0.7132 | L19 0.7578 | 0.6116 |

  **The shape is a U, not a ramp.** Correctness is decodable at the prompt (0.75), collapses to
  chance *mid-reasoning* (0.52, and 0.40 at layer 20), and returns strongly at the answer (0.83).
  If it survives replication, "the model's eventual correctness is partly set before it reasons,
  unreadable while it reasons, and readable again once it commits" is a far more interesting claim
  than the monotone story — but the primary CI spans zero and the retraction below is a direct
  warning against believing a curve shape at this n.

  **RETRACTION — this morning's depth gradient does not replicate.** Two probes at the **same
  position** (`last_prompt`), same features, same folds, same frozen C, differing only in which
  correctness labels they used:

  | layer | run A (labels @1100) | run B (labels @2048) |
  |---|---|---|
  | 0 | 0.6205 | 0.6942 |
  | 13 | 0.6775 | **0.7522** |
  | 20 | 0.7299 | 0.6685 |
  | 27 | **0.7612** | 0.5737 |
  | **argmax** | **L27** | **L13** |
  | mean over 28 layers | 0.6724 | 0.6543 |

  The two label sets **agree on 86.7% of items — 8 of 60 differ**, which is exactly the
  reproducibility floor measured on 2026-08-29. **The level is stable and the shape is not.**
  The claim in this morning's entry — "correctness decodability rises monotonically to the final
  layer, argmax L27" — and the inference drawn from it, that the readout improves with depth, are
  **withdrawn**. They were one draw of an unstable statistic.

  **Denoising the labels does not rescue the shape.** Ten independent unsteered generations of
  these 60 items exist on disk. Their vote distribution:

  | votes correct (of 10) | 0 | 1 | 2 | 3 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|
  | items | 19 | 2 | 1 | 5 | 3 | 1 | 1 | 2 | 26 |

  **45 of 60 items are unanimous; 15 are contested.** Probing with majority-vote labels and with
  the unanimous subset:

  | label treatment | n | L13 | L20 | L27 | argmax | mean |
  |---|---|---|---|---|---|---|
  | run A | 60 | 0.6775 | 0.7299 | 0.7612 | **L27** | 0.6724 |
  | run B | 60 | 0.7522 | 0.6685 | 0.5737 | **L13** | 0.6543 |
  | consensus of 10 | 60 | 0.6880 | 0.7250 | 0.7531 | **L27** | 0.6698 |
  | unanimous only | 45 | **0.8563** | 0.7692 | 0.7632 | **L13** | 0.7429 |

  Four treatments, argmax L27 / L13 / L27 / L13. **The depth curve's shape is not identifiable at
  this sample size.** The unanimous subset's 0.8563 is **not** a better measurement — contested
  items are precisely the borderline ones, so dropping them removes the items the probe cannot
  predict. That is a selection effect and must not be quoted as an improved AUC.

- **What worked / hypothesis verdict:**
  - **H-PD1 ✗ POSITION DOES NOT MATTER** as frozen — point estimate +0.1123 clears the threshold,
    CI [−0.0303, +0.2564] does not exclude zero.
  - **H-PD2 — not evaluated.**
  - **H-PD3 — the surface is reported; its shape is exploratory and, on today's evidence about
    shape stability, should not be built on without more items.**
  - **What IS established:** correctness is decodable from the residual stream at the final prompt
    token, **AUC ≈ 0.65–0.75**, permutation-clean at p = 0.005, and beating a length-only baseline.
    The *level* replicates across every label treatment. **Which layer is best does not.**

- **Observations:**
  - **The reproducibility floor has now produced a false finding and then caught it, inside 24
    hours.** The floor was measured on Friday as an abstract property; today it flipped the sign of
    a depth curve. That is the strongest possible argument for having measured it, and it means
    **any curve-shape claim in this programme at n = 60 needs a second label draw before it is
    reported.** This is a cheap, general protocol change.
  - **The corpus is now the binding constraint, not GPU time.** Every experiment this week ran in
    under two hours. What limits the conclusions is 60 items with 15 of them label-unstable, and
    no amount of compute fixes that.
  - The U-shape in position is a larger effect (range 0.31) than the depth wobble that just proved
    unstable (range ~0.18), which is mild grounds for taking it more seriously — but only mild.
  - The `answer_line` result agrees in direction with 2026-08-29's finding that `rt_cos` pays at
    the answer line and nowhere in the reasoning trace. Two instruments, same place. That
    convergence is the most promising thing in the readout direction and it is now the third time
    the mid-reasoning region has come up empty.

- **New questions / new hypotheses:**
  - **Use the vote fraction as a graded label.** k/10 correct across the ten existing draws is a
    continuous, far lower-variance target than one binary grade, and it needs no new GPU time — a
    Spearman correlation between the residual stream and k/10 would be much better powered at
    n = 60 than the binary AUC that just proved unstable. **This is the obvious next move and it
    is free.**
  - **Does the mid-reasoning collapse survive?** 0.40 at `reply_q50`/L20 is *below* chance, which
    is odd enough to be worth one replication with a second label draw before it is interpreted.
  - Expanding the stimulus set is the only way to settle shape questions. Dataset B holds 250
    snippets but `load_pairs` yields 60 usable L0/L1b pairs with execution-validated ground truth;
    raising that is a stimulus-engineering task, not an analysis one.

- **Next Steps:** the graded k/10 analysis (free, CPU); then decide whether the readout paper's
  positive is the answer-line convergence. P0.3-ext still blocked on `adversarial_rename`.
