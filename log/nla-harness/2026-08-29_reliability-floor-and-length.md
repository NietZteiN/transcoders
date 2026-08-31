### Target Date: 2026-08-29 (three calibration results — and a correction to yesterday's P0.4 entry)

Three cheap tests, run together because they answer one question: **what do this programme's
per-item numbers actually support?** One of them corrects a causal claim made in
[`2026-08-28_p04-depth-results.md`](2026-08-28_p04-depth-results.md). That entry is append-only
and stands as written; the correction is here.

- **Hypotheses / what we tested:**
  - **R1 — the per-item reproducibility floor.** P0.4 found greedy bf16 decoding agreeing only
    0.83–0.95 per item across cards, and *attributed it to card-to-card differences*. Every paired
    test in the programme (B4's primary gate, N13's matched 2×2, P0.4's own criterion (a)) treats
    per-item correctness as a fixed property. Measure the floor and find where it comes from.
  - **R2 — N10b split-half reliability.** The programme's one positive result is an item-level
    **AUC 0.764**. An AUC is a population statement. Can the score rank a *single* file?
  - **R3 — B6 judge-free.** Does `rt_cos` add anything over reply **length** in predicting
    correctness? `rt_cos` is known to correlate +0.28–0.31 with verbalizer output volume, which
    makes length the mandatory covariate rather than an afterthought.

- **Setup:**
  ```
  R1  nla/scripts/p04_repro_sbatch.sh + nla/src/p04_repro_score.py
      replicate = full unsteered baseline (60 L0 + 60 L1b) + V4_oracle at alpha=1.0 on 60 items,
      layer 20, last_prompt, seed 20260724 — identical to the P0.4 L20 arm in every respect.
      A1, A2  job 358068, g-07-10, GPU-ceae46c5  (SAME physical card, verified by UUID)
      B1      job 358069, g-07-11, GPU-7db9acf2  (different card)
  R2  nla/src/n10b_split_half.py — 200 random half-splits of each item's 14 reads,
      Spearman-Brown corrected to full length. Neutral arm (the pre-registered primary).
  R3  nla/src/b6_length_vs_faithfulness.py — 512 obfuscated-corpus cases with an outcome and a
      token count; logistic correctness ~ length vs ~ length + candidate; LRT + 5-fold CV AUC.
  ```
  Decision rules for R2 and R3 were written into the scripts' docstrings **before** either was
  run. R1 has no rule — it is calibration, not a hypothesis test.

- **Results:**

  **R1 — the floor is run-to-run, NOT card-to-card. This corrects yesterday's entry.**

  | comparison | baseline L1b | baseline L0 | steered V4 α=1 |
  |---|---|---|---|
  | **A1 vs A2 — same physical GPU** | **0.9000** (6/60 differ) | 0.9167 | **0.8500** (9/60) |
  | A1 vs B1 — different node | 0.9500 (3/60) | 0.9167 | 0.8667 (8/60) |
  | A2 vs B1 — different node | 0.9167 (5/60) | 0.9333 | 0.8833 (7/60) |

  **Two runs of the identical command, on the same card, with the same seed and greedy decoding,
  disagree on 6 of 60 items.** Cross-node agreement (0.95 / 0.92) is *not worse* than same-GPU
  agreement (0.90) — the three pairings are indistinguishable.

  **The card-to-card explanation given in
  [`2026-08-28_p04-depth-results.md`](2026-08-28_p04-depth-results.md) — "non-associative
  reduction order and autotuned kernel selection differ across physical cards" — is not supported.**
  The divergence is present between two runs on one card, so pinning hardware does not remove it.
  `torch.manual_seed(SEED)` is set but is irrelevant here: greedy decoding draws no RNG, so the
  seed never touches this path. The nondeterminism is in float kernel execution — per-process
  autotuning and backend selection — not in anything a seed controls. **This is worse for the
  programme than the entry said, because it cannot be fixed by pinning a node.**

  Two further facts:
  - **Steering makes it worse**: 0.85 steered vs 0.90 unsteered on the same card. The hook's
    arithmetic amplifies divergence rather than being neutral to it.
  - **Marginals stay tight while items churn.** Marginal gaps run 0.0167–0.0833 while 10–15% of
    individual items flip. Exactly the P0.4 pattern, now reproduced within a single card.

  **R2 — N10b is POPULATION ONLY on every metric.** Spearman-Brown reliability of the full
  14-read instrument, neutral arm:

  | metric | half r | **reliability (full)** | odd/even | half-AUC (sd) | verdict |
  |---|---|---|---|---|---|
  | `specific_frac` | 0.409 | **0.581** | 0.379 | 0.692 (0.015) | POPULATION ONLY |
  | `generic_frac` | 0.442 | **0.613** | 0.432 | 0.670 (0.012) | POPULATION ONLY |
  | `mm_mean` | 0.354 | **0.523** | 0.482 | 0.791 (0.017) | POPULATION ONLY |

  Nothing reaches the pre-stated 0.70. On the hard-negative stratum `mm_mean` falls to **0.477**.

  **The best population discriminator is the least reliable per item**: `mm_mean` carries the
  highest AUC (0.856 headline) and the lowest reliability (0.523). Meanwhile the half-AUC standard
  deviation is only 0.015 — **the AUC is stable under which reads you take; the per-item ranking
  is not.** Those are different properties and only the first has ever been measured here.

  **R3 — `rt_cos` adds nothing over reply length.** Primary stratum, all reads pooled, n = 512:

  | candidate | LRT p | ΔMcFadden R² | CV AUC | verdict |
  |---|---|---|---|---|
  | `rt_cos` over length | 0.266 | 0.0018 | 0.6933 → 0.6906 (**−0.0027**) | **ADDS NOTHING** |
  | `act_norm` over length | 0.027 | 0.0069 | 0.6933 → 0.6955 (+0.0022) | ADDS SOMETHING (marginal) |

  `rt_cos` alone reaches AUC 0.654 and evaporates under the length control — the predicted
  describability confound, now measured rather than argued.

  **Incidental and useful: reply length alone has AUC 0.306** — inverted, so *longer replies
  predict wrong answers*, and as a signed predictor it gives CV AUC **0.693** on its own. A token
  count beats every internal signal in the programme except N13's answer instability (0.869).

  **One secondary lead, surviving BH-FDR across the read classes:** `rt_cos` measured at the
  **`answer_line`** — LRT p = 2e-05, **q = 8e-05**, ΔCV AUC **+0.0267**. Faithfulness carries
  incremental information at the moment the answer is emitted, and none across the reasoning
  trace (`reasoning` stratum: p = 0.59, ΔCV AUC −0.0059). The BH correction is computed inside
  the script so the JSON cannot be quoted without it.

- **What worked / hypothesis verdict:**
  - **R1 ✓ measured, and it refutes yesterday's stated cause.** Floor: **0.85–0.90 per-item
    agreement on a fixed card**. Any paired per-item claim in this programme must carry it.
  - **R2 ✗ — the item-level framing of N10b is not supported.** Report it as a population
    separation. "This file scores 0.14, so it is probably malware" is not licensed.
  - **R3 ✗ — `rt_cos` adds nothing over a token count** in the pooled primary. `act_norm` is
    marginal (p = 0.027, ΔCV AUC +0.002) and not worth carrying either.

- **Observations:**
  - **The three results are the same result at three scales.** N10b's score ranks crowds but not
    files; `rt_cos` tracks how much text was produced rather than whether it was right; and the
    decoding pipeline itself does not reproduce an individual item. *Every* item-level claim in
    this programme has now been squeezed from a different direction, and each time the population
    level survives and the item level does not.
  - **P0.4's verdict is unaffected and its resolution is now measured.** Per-item churn inflates
    discordant counts roughly symmetrically, so it widens CIs rather than biasing the contrast;
    the primary's CI [−0.1833, +0.0500] already reflects it. What changes is that the limit is a
    number instead of a guess.
  - **B4's exact tie deserves re-reading.** "V1 0.550 = V3 0.550, 7-v-7, p = 1.00" was quoted as an
    exact identity. At a marginal resolution of ±0.02–0.03 and 10–15% item churn, the *exactness*
    is a coincidence. The conclusion (no separation) is unchanged and if anything better
    supported; the rhetorical force of "exactly" should go.
  - Writing the R2/R3 decision rules into the docstrings before running cost about ten minutes and
    made both verdicts mechanical. R3's `answer_line` lead would have been very easy to promote to
    a headline without the pre-declared primary stratum and the BH correction.

- **New questions / new hypotheses:**
  - **Can the floor be lowered?** `torch.use_deterministic_algorithms(True)`, a fixed SDPA backend,
    and `CUBLAS_WORKSPACE_CONFIG` would test whether this is autotuning or something structural.
    If a deterministic backend gets same-card agreement to 1.0, every future paired run should use
    it and the floor becomes a choice rather than a fact. **This is the cheapest high-value thing
    left** — one replicate pair, ~30 min.
  - **How much of the existing ledger is inside the floor?** N13's AUCs (0.491 / 0.442 / 0.408) and
    B5's CIs spanning zero were all read as nulls. A null that sits inside a 10–15% churn band is
    still a null, but the *bounds* quoted for it are too tight, and the six documented
    interim-to-full reversals now have a candidate mechanism that is not sample size.
  - **The `answer_line` lead is the only place faithfulness has ever paid.** If the readout paper
    needs a positive, that is where to look — and it is a *read*-side result, which is exactly the
    side P0.4 left untested.

- **Next Steps:** the determinism check above; then the per-layer dense probe (depth for a *read*).
  P0.3-ext still blocked on `adversarial_rename`, which exists only on csr-94608.
