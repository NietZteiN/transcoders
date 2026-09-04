### Target Date: 2026-09-03 (L3 results — one real predictor, one artifact, and a correction to today's L entry)

Result for the rule frozen in [`2026-09-03_g-variance-prereg.md`](2026-09-03_g-variance-prereg.md).
**This entry also corrects a headline claim in
[`2026-09-03_length-confound-results.md`](2026-09-03_length-confound-results.md)** — per protocol
that entry is left untouched and the correction lives here. No GPU.

- **Hypotheses / what we're testing:** **H-L3** — does any cheap, on-disk property of the stimulus
  predict the per-item obfuscation cost? CONFIRM if some predictor reaches q < 0.05 and
  |ρ_partial| ≥ 0.30.

- **Setup:** `nla/src/g_variance.py` (sha256 a90d493e06ed…), CPU, env `nla-mi`, seed 20260724, n = 60,
  10,000-resample bootstrap, 1,000-permutation p, BH-FDR across the six frozen predictors, each a
  partial Spearman controlling rank(|y_clean|). Output
  `data/nla/p0/trace_llr/gemma12b/g_variance.json`. The post-hoc section below re-ran the same six
  predictors against **G_sum = G × |y_clean|**, the un-normalised total cost in nats — this was
  **not** in the frozen rule and is labelled post-hoc throughout.

- **Results:**
  - **Frozen rule → `L3-SIGNAL`.** Two predictors clear q < 0.05 and |ρ| ≥ 0.30:

    | predictor | ρ (partial, \|y_clean\| controlled) | CI95 | q |
    |---|---|---|---|
    | `decoy_overlap` | **−0.352** | [−0.546, −0.085] | 0.036 |
    | `decoy_verbosity` | **+0.305** | [+0.035, +0.507] | 0.046 |
    | `snippet_size` | +0.297 | [−0.012, +0.522] | 0.046 |
    | `id_surface` | +0.252 | [−0.030, +0.475] | 0.081 |
    | `language_js` | −0.236 | [−0.454, +0.038] | 0.094 |
    | `l0_correct` | −0.037 | [−0.355, +0.294] | 0.796 |

  - **The control was doing enormous work, so I checked it.** Raw ρ(G, |y_clean|) = **−0.889**, and
    ρ(G, 1/|y_clean|) = **+0.889**. **G is very largely 1/reply-length.** The per-token normalisation
    *injects* item-level variance rather than removing it: CV(G) = **0.590** against
    CV(G_sum) = **0.353**. The total cost is the more stable quantity across items.
  - **Re-testing on the un-normalised total (post-hoc) separates the real from the artifact:**

    | predictor | ρ vs G_sum | CI95 | q | verdict |
    |---|---|---|---|---|
    | `snippet_size` | **+0.512** | [+0.283, +0.692] | 0.006 | strongest on total cost |
    | `decoy_overlap` | **−0.382** | [−0.600, −0.118] | 0.006 | **replicates** |
    | `id_surface` | +0.307 | [+0.023, +0.557] | 0.036 | replicates, weaker |
    | `language_js` | −0.252 | [−0.482, −0.001] | 0.072 | — |
    | `decoy_verbosity` | **+0.090** | [−0.173, +0.340] | 0.508 | **does not replicate → artifact** |

- **What worked / hypothesis verdict:**
  - **H-L3 ✓ SUPPORTED, but with one of its two survivors withdrawn.** `decoy_overlap` holds on
    **both** readouts and in the **pre-registered direction** (negative): items whose decoy terms
    overlap the true terms cost the model less. `decoy_verbosity` was significant on G only and is
    **an artifact of the per-token normalisation** — longer decoy names go with longer replies.
  - **My prediction for the best candidate (`id_surface`) was wrong**; it lands third on both
    readouts and clears q only post-hoc.

- **CORRECTION to [`2026-09-03_length-confound-results.md`](2026-09-03_length-confound-results.md).**
  That entry's headline — *"the amount of renaming does not predict what the obfuscation costs the
  model"*, and the inference *"L1b is not a dosage effect"* — is **WITHDRAWN**. It rests on
  ρ(G, Δ) = −0.166, which is an artifact of the same normalisation. On the un-normalised total:

  | test | on G (frozen) | on G_sum (post-hoc) |
  |---|---|---|
  | **H-L1** ρ(cost, Δ tokens) | −0.166 [−0.402, +0.087] | **+0.378 [+0.104, +0.606]** |
  | **H-L2** partial(cost, inflation \| n_renames) | −0.159 [−0.424, +0.140] | **+0.020 [−0.213, +0.241]**, p 0.86 |

  So **H-L1 was right as originally pre-registered** (+0.3…+0.5; measured +0.378, CI excluding zero)
  and the "failed prediction" I recorded was a failure of the *readout*, not of the prediction.
  Renaming **is** dose-dependent on total cost.
  **The decision is unaffected: H-L2 is null on both readouts, so `L-CLEAN` stands and H-T3
  (the length-matched corpus fork) stays retired.** The consequential conclusion of that entry
  survives; its headline claim does not.

- **Observations:**
  - **Scope of the normalisation problem — it does not touch R2.** R2 compares *arms within an item*,
    where every arm shares the same |y_clean| denominator, so it cancels in arm-vs-arm contrasts.
    The defect is specific to **item-level correlations** using a per-token quantity. `R2-GENERIC`,
    the mover table and the belief-arms-are-negative result are unaffected.
  - **Four times today a measurement definition, not the science, produced the wrong headline:**
    R's ratio denominator, S's computed-but-ungated veto, T's unreachable sanity threshold, and now
    G's per-token normalisation. Each was caught by a control or a diagnostic rather than by the
    result looking wrong. The lesson worth carrying: **check what the readout is mechanically
    coupled to before correlating anything item-level with it.**
  - The surviving substantive finding is the theory-relevant one. Decoy/true term overlap predicting
    *lower* cost is semantic displacement behaving as Papers 2–3 describe: the trap works to the
    extent the decoy names point somewhere **other** than the truth. Snippet size predicting higher
    total cost is unsurprising and mostly a scale effect.

- **New questions / new hypotheses:**
  - **H-L5:** `decoy_overlap` is a crude set-intersection over `terms_true`/`terms_decoy`. A graded
    plausibility/displacement measure should predict better. Now motivated enough to be worth an
    embedding or judge score — the thing L3's prereg deferred.
  - **H-L6:** re-run the item-level parts of this thread on **G_sum** as the primary readout, with
    per-token G kept only for arm-level contrasts. Cheap, and it is the correct default going forward.
  - **H-L4** (powered dose test on Dataset B) is now *more* interesting, not less: it should confirm
    a **positive** dose effect on accuracy, matching G_sum rather than G.

- **Next Steps:**
  1. Treat `decoy_overlap` as a lead, not a finding: it needs Dataset B's 250 snippets to be claimed.
  2. Adopt G_sum for item-level work (H-L6); no re-run of arm-level results is needed.
  3. Standing: 6/60 flippable gate; grow the corpus from Dataset B.
