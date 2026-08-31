### Target Date: 2026-08-30 (behavioural predictors — real, smaller than they look, and a flag on N13)

Free analysis of the ten unsteered draws already on disk. Prompted by the observation that every
internal measure in this programme loses to a behavioural one. Script: `nla/src/p1b_behavioural.py`.

- **Hypotheses / what we tested:**
  The ten draws differ only through the nondeterminism floor characterised on 2026-08-29, so the
  spread of an item's answers across draws **is** an instability measure — obtainable from repeated
  greedy runs with no sampling at all. Three predictors, both tiers: reply length, answer
  instability (1 − modal answer's share), and parse-failure rate.

- **Results:**

  **Same-draw estimates (the naive ones):**

  | tier | mean instability | ρ vs correct | **AUC** | parse-fail ρ |
  |---|---|---|---|---|
  | L0 | 0.207 | −0.5492 | **0.8270** | −0.4743 |
  | L1b | 0.223 | −0.5584 | **0.7834** | −0.4178 |

  **These are inflated, and the dependency is structural.** Instability and graded correctness are
  computed from the *same* draws, and they are mechanically entangled: y = 0.5 forces at least two
  distinct answers, hence instability ≥ 0.5. It is not circular — consistently *wrong* items also
  have low instability, which is why ρ is −0.55 and not −0.9 — but the dependency should be removed
  rather than argued about.

  **Split-draw control — instability from draws 1–5, correctness from draws 6–10, sharing no
  generation:**

  | tier | ρ vs correct | **AUC** | parse-fail ρ |
  |---|---|---|---|
  | L0 | −0.3644 | **0.6450** (was 0.8270) | **−0.5032** |
  | L1b | −0.4588 | **0.7368** (was 0.7834) | −0.3089 |

  **Roughly a third of the apparent signal on L0 was the shared-draw dependency**, and the honest
  numbers are 0.6450 / 0.7368. Two consequences:

  - **The tier ordering reverses.** Same-draw says L0 (0.827) > L1b (0.783); held out, L1b (0.737)
    **>** L0 (0.645). The "degradation under obfuscation" reading of these predictors is not
    supported — if anything instability is a *better* correctness signal under adversarial renaming.
  - **Parse failure is the more robust of the two on clean code** (ρ = −0.5032 held out, slightly
    *stronger* than its same-draw estimate) and the weaker one under renaming (−0.3089).

- **What worked / hypothesis verdict:**
  - **Answer instability is a real correctness signal and it is free** — AUC 0.645–0.737 from
    repeated greedy runs, no sampling, no instrument. It still exceeds what the residual stream adds
    over a token count at `last_prompt` (+0.04–0.06).
  - **But it is not the 0.869 headline number**, and the difference is the estimator, not the data.
  - **No support for behavioural degradation under obfuscation.** The instability effect is flat to
    slightly reversed across tiers; only parse-failure weakens (−0.5032 → −0.3089), on one measure.

- **Observations:**
  - **This flags a banked result. N13's free positive — answer instability → correctness at
    AUC 0.869 — was, on the description in the ledger, estimated from the same generations that
    supplied the correctness labels.** If so it carries the same inflation demonstrated here, where
    the identical dependency cost 0.18 AUC on L0. **N13's 0.869 should be re-estimated on held-out
    draws before it is quoted again**, and it is currently the strongest positive in the whole
    programme, so this matters more than any result in this entry.
  - **The nuisance became the instrument.** Two days were spent characterising run-to-run
    nondeterminism as a threat to validity; that same nondeterminism, measured deliberately, is a
    better correctness predictor than anything read out of the residual stream at the prompt site.
    The floor is not only a caveat — it is a signal, and it is free.
  - Every honest number in this programme now sits in a narrow band: instability 0.65–0.74, length
    ρ 0.32–0.48, residual-stream increment 0.04–0.06. Nothing internal is close to the behavioural
    measures, and the behavioural measures are themselves modest.

- **New questions / new hypotheses:**
  - **Re-estimate N13 held out.** Highest priority in this entry; it is a re-analysis, not a re-run.
  - **Does instability beat length, or duplicate it?** Both are behavioural and both are available;
    the incremental value of instability over reply length has not been computed, and after the
    tier retraction the lesson is to compute the increment before claiming the predictor.
  - Five draws per half is a thin instability estimate. The ladder run in flight adds five draws per
    tier across all five tiers, which will allow this to be redone with a wider base and on the
    relational tiers.

- **Next Steps:** the ladder (in flight, array 359698); N13 re-estimation; instability-over-length
  increment.
