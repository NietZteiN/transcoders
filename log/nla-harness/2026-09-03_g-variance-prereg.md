### Target Date: 2026-09-03 (L3 — what carries the per-item obfuscation cost? Rule frozen before running)

Raised by [`2026-09-03_length-confound-results.md`](2026-09-03_length-confound-results.md): the
*amount* of renaming does not predict what obfuscation costs the model (ρ = −0.166 against a
pre-registered +0.3…+0.5), so the per-item variance in **G** is carried by something else.
**Committed before the analysis runs.** No GPU; banked 60 items.

- **Hypotheses / what we're testing:** Genuinely **exploratory and hypothesis-generating** — there is
  no prior to power against and n = 60. Stated so that a survivor is treated as a lead needing
  replication, not a finding.
  - **H-L3:** at least one cheap, on-disk property of the stimulus predicts G. CONFIRM if some
    predictor reaches **q < 0.05** (BH-FDR across the six) **and** |ρ_partial| ≥ 0.30.
    REFUTE (`L3-NULL`) otherwise — G's variance is not explained by anything cheaply available, and
    the next step would have to be a judge-scored or embedding-based plausibility measure.

- **Setup (frozen).** G = the R2 clean-half unit, per item, n = 60.
  **Nuisance control, applied to every test:** |y_clean|, the clean reply's token count. G divides a
  difference of sums by |y_clean|, so longer replies average over more tokens and shrink G toward its
  mean — a correlation with reply length would be partly **mechanical**. Every predictor below is
  therefore tested as a **partial** Spearman controlling rank(|y_clean|), and reply length itself is
  reported separately as a diagnostic, **not** as a candidate explanation.
  Six predictors, fixed now, no additions after seeing results:

  | # | predictor | definition | why |
  |---|---|---|---|
  | 1 | snippet size | `len(l0_prompt_ids)` | bigger problems may simply cost more |
  | 2 | identifier surface | identifier tokens in L1b ÷ `len(l1b_prompt_ids)` | scale-free version of "how much of the prompt is renamed" — the dose question the token *count* failed to answer |
  | 3 | decoy/true overlap | \|set(terms_decoy) ∩ set(terms_true)\| ÷ \|set(terms_true)\| | items whose decoy shares terms with the truth should be *less* displaced → predict **negative** |
  | 4 | decoy verbosity | mean token length of the decoy identifiers | rarity/complexity proxy for the decoy names |
  | 5 | language | python = 0, javascript = 1 | the corpus spans both; a language effect would confound everything else |
  | 6 | clean-task difficulty | `l0_correct` (0/1) | whether the model solves the un-obfuscated version at all |

  Spearman throughout; partial by ranking and regressing out rank(|y_clean|) by OLS. Percentile
  bootstrap over items, 10,000 resamples, seed 20260724; two-sided permutation p; **BH-FDR across the
  six**. Joint rank regression of G on all six reported with **adjusted** R² and an explicit
  overfitting caveat (6 predictors, 60 items).

- **Power, stated in advance.** At n = 60 and α = 0.05, a true ρ = 0.35 is detected ~55 % of the time
  before any multiplicity correction; with BH across six it is lower. **`L3-NULL` therefore means
  "no large effect among these six", not "nothing predicts G".** This is recorded now so the null
  cannot later be reported as if it were decisive.

- **Decision table (frozen):**

  | verdict | condition | consequence |
  |---|---|---|
  | **L3-SIGNAL** | ≥ 1 predictor, q < 0.05 and \|ρ_partial\| ≥ 0.30 | a lead; pre-register it as a confirmatory test on Dataset B's 250 snippets before it is claimed |
  | **L3-WEAK** | ≥ 1 predictor q < 0.05 but all \|ρ_partial\| < 0.30 | direction worth recording, effect too small to chase at this n |
  | **L3-NULL** | none reach q < 0.05 | nothing cheap explains G; a plausibility measure (judge or embedding) is the only remaining route, and is out of scope until something else motivates it |

- **Predictions.** Best candidate: **#2 identifier surface** (positive) — the dose question asked in a
  scale-free way. **#3 overlap negative.** I expect **#5 language** to be null and would treat a
  language effect as a warning about corpus construction rather than a result. Given the H-L1 miss I
  hold these loosely; my confidence in #2 is lower than it would have been yesterday.

- **Next Steps:** implement `nla/src/g_variance.py`, run, log in a new dated entry.
