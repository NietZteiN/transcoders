### Target Date: 2026-08-28 (P0.3-ext pre-registration — full grid on juno, three seeds, layer-matched controls)

**Status: PRE-REGISTRATION. Written before a single juno cell has been generated.** The only P0.3
numbers in existence are the eight csr-94608 cells reported in
[`2026-08-28_p0-triage-results.md`](2026-08-28_p0-triage-results.md), which are quoted below and
are **not** being renegotiated by this document.

- **Hypotheses / what we're testing:**

  P0.3 returned **SITE LIVE**: attention steering restricted to `[20,20]` moved P@1 by
  **−1.434 / −1.658** points against a baseline seed-to-seed spread of **0.794**, same direction at
  both seeds. That verdict is recorded and stands. It licenses exactly one claim — *layer 20 is not
  inert* — and it does so on **two** seeds, on a pipeline where **E5b watched a five-transform
  family average flip sign between draws**. Two questions follow, and neither is answerable from
  the existing table.

  - **H-ext1 (the one that matters — layer specificity).** SITE LIVE says a single-layer
    intervention at 20 costs accuracy. It does **not** say layer 20 is *special*: if perturbing any
    single layer costs about the same, then "site live" is a fact about single-layer attention
    steering in general and carries no information about the belief-injection site. Layer-matched
    controls at **[13,13]** and **[26,26]** decide it. 13 is not an arbitrary control — P0.1 found
    cross-item coherence of the task direction peaks there (0.543 vs L20's 0.459) — and 26 is a
    late-layer control matched to 20's depth regime.
  - **H-ext2 (replication at n = 3).** The E5b lesson was that **two seeds is the floor, not the
    target**. A third seed either confirms the [20,20] effect or reverses it.
  - **H-ext3 (the pre-registered design, finally closed).** The `[20,27]` cells were ~25% complete
    when the cluster migration stopped them. They are not needed for SITE LIVE — that branch reads
    only [20,20] versus baseline — but the frozen P0.3 design specified a four-point scope curve and
    it should exist.

- **Setup:**

  **The whole grid is re-run on juno, including the eight cells csr already finished.** Two reasons,
  both stated before any number is seen. (1) The per-run `score.json` tree that backs the csr cells
  **did not survive the migration**, so those eight cells cannot be re-scored, re-clustered, or
  given a different denominator here — they exist only as a restored summary. (2) Re-running them
  turns a provenance debt into a **cross-platform replication check** at no extra design cost. On a
  cluster with 26 × 2 H200s the marginal cost of the redundant cells is wall-clock we are not
  otherwise using.

  | | |
  |---|---|
  | grid | 7 bands × 3 seeds = **21 cells** |
  | bands | `base` (no steering) · `L20` [20,20] · `L2021` [20,21] · `L2023` [20,23] · `L2027` [20,27] · **`L13` [13,13]** · **`L26` [26,26]** |
  | seeds | 1000 · 2000 · **3000** |
  | model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
  | corpus | 164 HumanEval Java programs, `adversarial_rename`, 3 runs/snippet |
  | sampling | temperature 0.9, top-p 0.95, max-new-tokens 512 |
  | attention | `--prior slice --beta-post 0.8 --head-subset-mode none --n-bins 8` |
  | belief channel | **OFF** (α = 0) throughout — this is purely the positional lever |
  | scoring | `nla/src/p03_score.py`, P@1 = Σ`case_correct` / Σ`case_total` over every run in a cell |
  | platform | juno-l-01, SLURM `h200` partition, one GPU per cell, `nla/scripts/p03_sbatch.sh` |
  | env | `codesteer` rebuilt on juno (transformers 4.57.1 pinned — the attention backend imports 4.x decoder internals), JDK 21 |

  **csr-94608 reference values, quoted here so they are frozen before the juno runs land:**

  | cell | seed 1000 | seed 2000 |
  |---|---|---|
  | base | 64.974 | 64.180 |
  | L20 | 63.541 | 62.522 |
  | L2021 | 61.883 | 62.971 |
  | L2023 | 62.798 | 63.972 |
  | L2027 | *61.821 (partial)* | *61.777 (partial)* |

- **Frozen decision rules:**

  **G-juno — the replication gate. Nothing below is believed until this passes.**
  The juno re-runs of `base_s1000`, `base_s2000`, `L20_s1000`, `L20_s2000` must each land within
  **±1.00 P@1 point** of the csr value above.
  - *Why 1.00.* It is strictly smaller than the **1.434 / 1.658** effect the SITE LIVE verdict rests
    on. A pass therefore means cross-platform drift **cannot manufacture that effect**. It is larger
    than the 0.794 baseline seed spread, so the gate does not demand better-than-seed reproducibility
    across a different GPU architecture, CUDA version and rebuilt environment. Like the 0.50 in
    P0.1, the exact number is arbitrary and is frozen here **because** it is arbitrary.
  - **FAIL ⇒** the juno grid is reported as a **separate platform**, the csr-based SITE LIVE verdict
    stands exactly as published, and the discrepancy itself becomes the finding to chase. It does
    **not** silently replace the old numbers.

  **Primary — H-ext1, is layer 20 special?**
  Let `spread₃` = max − min of the three juno `base` cells. A band **MOVES** iff
  |Δ vs same-seed baseline| > `spread₃` at **all three** seeds, in the **same direction** at all
  three.
  - **LAYER-20-SPECIFIC** — `L20` moves and **neither** `L13` nor `L26` moves.
  - **GENERIC SINGLE-LAYER COST** — `L20` moves **and** at least one of `L13` / `L26` also moves in
    the same direction. Then SITE LIVE is a fact about single-layer attention steering, not about
    layer 20, and it must be re-worded wherever it appears.
  - **INCONCLUSIVE** — `L20` moves and the controls disagree with each other.

  **Secondary — H-ext2, does [20,20] survive a third seed?**
  - **CONFIRMED at n = 3** — `L20` moves under the rule above.
  - **REVERSED** — it does not. This would be the **seventh** interim-to-full reversal in this
    programme, and it is to be logged as a reversal in exactly those words, not as a "weakened"
    result. The 2-seed SITE LIVE entry is append-only and stays; the correction goes in a new entry.

  **Tertiary — H-ext3, the scope curve.** The four `20:N` bands are reported as **four points, not a
  trend**. No monotonicity claim unless the ordering of the four is identical at all three seeds.

  **Standing constraints, restated so they are not renegotiated:**
  - `recovered` / `damaged` reported separately, never netted.
  - Every cell records its seed; a cell that fails is re-run, never dropped, and
    `runs_unscorable` is reported rather than absorbed into the denominator.
  - This extension **cannot overturn P0.2**, which it does not touch, and therefore cannot by itself
    move Phase 0 out of licensing row 4. The strongest thing a GENERIC verdict does is weaken the
    *interpretation* of SITE LIVE from "layer 20 carries interventions" to "single-layer attention
    steering costs accuracy anywhere" — which still leaves the site non-inert and still leaves
    belief-shaped writes doing nothing.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** run `nla/scripts/p03_sbatch.sh`, then `nla/src/p03_score.py` and
  `nla/src/p0_verdict.py`; record results in a new dated entry.
