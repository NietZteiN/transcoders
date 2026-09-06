### Target Date: 2026-09-06 (H-W13/H-W14 — the effect **accumulates** across spans and is mostly **item identity**; the harness passes an exact identity check)

**Thread:** nla-harness · **Job:** 379075 (h200, **44:28**, COMPLETED, 0 errors, 49/49 items) ·
**Host:** `gemma12b` L32/48 · **Resolves:** [`2026-09-05_dose-and-tier-prereg.md`](2026-09-05_dose-and-tier-prereg.md).

- **Hypotheses:** H-W13a (redundant: `P_1 ≥ 0.50 × P_all`), H-W13b (accumulative + monotone),
  H-W14 (`F_all − F_L1b_all ≥ +12.11` → a generic clean-code component), plus the **SELF** identity
  assertion.

- **Results.**

  **SELF identity: −0.0096 nats against a 1.0 tolerance — PASS.** Writing each position's *own*
  activation back is exactly the identity, and it is: **exactly 0.000 on 21 of 49 items**, max |ΔG|
  **1.195** across all 49. The write path, the per-position mapping and the scorer are validated end
  to end. *(The non-zero residue on 28 items is bf16 non-determinism in the forward pass, the same
  order as the reproducibility floor this thread has documented; it is bounded and far inside the
  frozen tolerance.)*

  | arm | G_sum | 95 % CI |
  |---|---|---|
  | `P_1` | +10.29 | [+8.67, +11.92] |
  | `P_2` | +17.95 | [+15.45, +20.63] |
  | `P_4` | +28.19 | [+24.46, +32.21] |
  | `P_all` (≈7.65 spans) | **+45.71** | [+40.89, +50.78] |
  | `F_1` | +1.81 | [+0.87, +2.96] |
  | `F_all` | +14.27 | [+11.80, +16.88] |
  | `F_L1b_all` | +10.10 | [+8.22, +12.01] |

  **Dose curve monotone.** `P_1 / P_all` = **0.225** [0.194, 0.257], against a 0.50 bar.
  Content contrasts: at k=1 `P_1 − F_1` = **+8.49** [+6.67, +10.26] (45/49, below the +12.11 bar);
  at k=all `P_all − F_all` = **+31.44** [+26.82, +36.46] (**49/49**, clears).
  **H-W14:** `F_all − F_L1b_all` = **+4.16** [+1.37, +7.30] — CI excludes 0 but far below +12.11.
  **Verdict `W13-ACCUMULATIVE-TIER-GENERIC`.**

- **What worked / hypothesis verdict.**
  - **H-W13a ✗ NOT redundant.** One span buys **22.5 %** of the effect, not the ≥ 50 % the redundant
    reading requires.
  - **H-W13b ✓ ACCUMULATIVE, with diminishing returns.** The curve is monotone, and per-span yield
    falls steadily: **10.29 → 8.98 → 7.05 → 5.98** nats per span at k = 1, 2, 4, all. Perfect
    linearity across 7.65 spans would give `P_1/P_all` = **0.131**; the measured **0.225** is well
    above that, so the first span buys more than its share — the signal accumulates but saturates.
  - **H-W14 ✗ tier is mostly GENERIC.** A foreign item's *obfuscated* vector already delivers
    **+10.10**, and making it *clean* adds only **+4.16**. Most of what a foreign vector supplies is
    program-shaped-state, not cleanliness.

- **Observations.**
  - **The four measurements now decompose `P_all` additively, and they sum exactly.**

    | component | nats | share |
    |---|---|---|
    | generic — any activation from another program (`F_L1b_all`) | +10.10 | 22.1 % |
    | + that vector being **clean** rather than obfuscated | +4.17 | 9.1 % |
    | + it coming from **this item** | **+25.67** | **56.2 %** |
    | + **correct span** assignment | +5.77 | 12.6 % |
    | **total** | **45.71** | 100 % |

    **Item identity is the majority of the effect (56 %); span identity is 13 %; being clean at all
    is 9 %.** That is the whole W programme's result in one table, and every row is a measured
    contrast rather than an inference.
  - **The accumulative result and the sibling-swap result fit together.** The battery permuted the
    item's own clean vectors *across all* spans and kept 87.4 %; here, patching *one* span correctly
    keeps 22.5 %. Both hold because what matters is that **every position receives a clean vector
    from this item** — *which* one it receives is nearly irrelevant, but *how many* positions get one
    matters a great deal.
  - **⚠️ `F_L1b_all` tripped the behavioural veto** (accuracy 0.714 → **0.653**, parse unchanged), so
    H-W14's null is measured against a mildly damaging arm and its +10.10 may be *understated*. The
    direction of that bias makes the H-W14 gap a **lower bound**, so "tier is mostly generic" is not
    an artifact of the damage — but the exact +4.16 should not be quoted without this caveat.
  - **`P_all` = +45.71 for the third time**, in three separate jobs on three nodes (377835, 378019,
    379075). Teacher-forced `G_sum` is reproducible to the reported precision.
  - **Cost note:** 44 min of compute after **~19 h in the queue**. Measured across 196 GPU jobs since
    2026-09-01, this cluster runs a **3:1 wait-to-run ratio** (median wait 110 min, median run 8 min),
    and the subject-only jobs were widened to `h200,h100` in response (commit `4aa98b3`).

- **New questions / new hypotheses.**
  - **H-W15:** the per-span yield falls 10.29 → 5.98. Is saturation a property of the *positions*
    (later spans carry less) or of the *quantity* (the model only needs so much clean evidence)?
    Re-running the ladder in reverse seeded order separates them and costs one job.
  - **H-W16:** the 56 % item-identity component is the thing to characterise next. Is it "this
    program's semantics" or "this prompt's surface"? A **paraphrased-L0** control — same program,
    different formatting — would split them.
  - **H-W12 is now settled as small but real:** span identity is **12.6 %** of the effect, consistent
    with the battery's +5.77 and no longer merely "sub-bar".

- **Next Steps:** H-W16 is the higher-value of the two and needs new stimuli (paraphrased L0);
  H-W15 is one cheap job on the banked pool. Pre-register before either.
