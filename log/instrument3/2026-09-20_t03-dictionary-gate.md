### Target Date: 2026-09-20 (T0.3 — the dictionary gate: `DICT-MARGINAL`, but the gate's own question passes)

- **Hypotheses / what we're testing:** T0.3 from [`../../docs/EXPERIMENT_BACKLOG.md`](../../docs/EXPERIMENT_BACKLOG.md),
  the gate in front of E1/E2/E7. The Llama Scope dictionaries are trained on Llama-3.1-8B **base**, every
  E1/E2 claim reads on **Instruct**, and `CLAUDE.md` §4 names base→instruct transfer as a silent-failure
  mode alongside degenerate reconstruction and dead features. Thresholds set before the run:
  `DICT-HEALTHY` FVE ≥ 0.70 and L0 ∈ [10, 200] · `DICT-DEGENERATE` FVE < 0.50 or L0 < 1 ·
  `DICT-MARGINAL` otherwise.

- **Setup:** `src/t03_dictionary_smoke.py`, SAELens 6.47.0 official `llama_scope` loader (delegated on
  purpose: Llama Scope's dataset-wise normalisation plus JumpReLU is exactly the convention that, if
  hand-rolled wrong, manufactures a "degenerate" verdict out of my own arithmetic), `llama_scope_lxr_8x`,
  stimuli = this project's own HumanEval-X Java. Jobs 413607 / 413862 / 413871 / 413880 / 413882 / 413884.

- **Results:**

  **Provenance — our pinned copy IS the fetched dictionary.** SAELens resolves `fnlp/…`;
  `configs/dictionaries.yaml` pins `OpenMOSS-Team/…@8dbc1d85`. Best-fit scalars **10.1386** (encoder) and
  **0.09863** (decoder) — and √4096 / 6.3125 = 10.1386 exactly — with relative residual **~1e-14** and
  cosine **1.0000**. Same dictionary, differing only by the folded normalisation constant. **E1 may use
  the pinned revision and these measurements transfer to it.**

  **The gate's question: base→instruct transfer is NOT a problem.**

  | | FVE | L0 |
  |---|---|---|
  | base / java | 0.4934 | 23.4 |
  | base / english | 0.4364 | 30.2 |
  | instruct / java | 0.4446 | 21.7 |
  | instruct / english | 0.4205 | 28.7 |

  Checkpoint effect **+0.0488** (java) / **+0.0159** (english); domain effect **−0.0570** — Java is
  *better* than English.

  **Layer sweep (E1 targets 12/16/20; layer 8 was merely what was cached):**

  | layer | base | instruct |
  |---|---|---|
  | 8 | 0.4934 | 0.4446 |
  | 12 | 0.4643 | 0.4151 |
  | 16 | 0.5435 | 0.4916 |
  | **20** | **0.5693** | **0.5206** |

  L0 is **22–27 at every layer and both checkpoints**, against a trained `top_k` of 50.

  **VERDICT `DICT-MARGINAL`** — best cell 0.5206 (instruct, L20), below the 0.70 healthy bar, above the
  0.50 degenerate bar, L0 well inside range.

- **What worked / hypothesis verdict:**
  - **The gate passes on what it was built to ask.** Base→instruct costs ≤ 0.049. More usefully, the
    residual low FVE is **uniform across checkpoint, domain and layer**, so it cannot bias an
    **L0-vs-L1b contrast** — which is what E1's Semantic-Capture Score actually measures. A uniform
    reconstruction ceiling weakens absolute feature claims, not the differential.
  - **Three hypotheses tested, three refuted:** transfer (✗, ≤0.049), domain (✗, Java *better* by
    0.057), activation-scale mismatch (✗, our mean norm 5.8724 vs stored 6.3125 — 7 %).
  - **`DICT-MARGINAL` is the honest label and I declined to make it `HEALTHY`.** Scaling x by ~2× reaches
    FVE 0.571, but the measured norm already matches the stored constant and that scale drives L0 to 66
    against a trained 50. Passing the threshold by an unjustified fudge would have been the worst
    outcome available.

- **Observations:**
  - **The largest correction was BOS, and it would have poisoned every SAE number in this project.**
    Identical activations scored **FVE −2100.8** with position 0 included and **+0.444** without. Llama's
    attention-sink token carries a residual norm an order of magnitude above every other position and the
    SAE reconstructs it badly, so it dominates the sum of squares. Now excluded in the script, documented,
    and it applies to E1/E2/E7 equally.
  - **My first provenance check fired for the wrong reason** — it compared raw safetensors against a
    loader-transformed state dict and reported `max|diff| 4.57`, which reads as "different dictionary".
    Replaced with a proportionality test (best scalar, residual after scaling, cosine), which resolved it
    to 1e-14. Had I trusted the first output I would have gone looking for a nonexistent problem.
  - **L0 ≈ 22–27 against a trained 50 is unexplained and should be carried as an open caveat.** It is
    stable across layers and checkpoints, so it is systematic rather than noise; the scale test rules out
    the obvious cause.
  - **Recommendation for E1:** run at **L16/L20** (best FVE), report FVE and L0 beside every feature
    claim as `CLAUDE.md` §4 requires, and keep the SAE strictly as a **discovery** tool with the mandated
    dense/prompting baselines — a dictionary explaining ~half the variance is not a measurement
    instrument.

- **New questions / new hypotheses:**
  - **T0.4:** why is L0 half the trained `top_k` everywhere? Candidates: the JumpReLU threshold was tuned
    on their corpus; or `dataset_average_activation_norm` is layer-specific and the stored 6.3125 belongs
    to layer 8 only (the sweep reused one SAE's constant implicitly via each layer's own folded weights —
    worth checking each layer's `hyperparams.json`). Free, no GPU.
  - **T0.5:** compare against the **32×** width variant (`llama_scope_lxr_32x`) at L20. If FVE rises
    materially, E1 should use it; the charter already lists it as a width variant.

- **Next Steps:** E1 is unblocked. Run it at L16/L20 with FVE/L0 reported alongside.
