### Target Date: 2026-09-02 (PRE-REGISTRATION — does Llama's L2 cell survive replication?)

**Frozen before any replication draw exists.** The cell in question was found by exploration, on
data I had already inspected, and it is the only survivor in a 9-cell matrix. Everything below is
written so that finding is either confirmed or discarded by a rule set in advance.

- **Hypotheses / what we're testing:**

  The read-side battery on three hosts found the relational route beats reply length on Qwen
  (+0.112) and Llama (+0.103) but **not** Gemma (−0.057). The deflation controls then ran per host.
  Two of nine cells returned BEYOND STATIC COMPLEXITY, and both sat on a **degenerate combined
  baseline** — the combined ridge scoring worse out-of-fold than its own best component, by 0.118
  (Gemma L2) and 0.102 (Llama L2). Under a strict baseline of `max(length, static, combined)`:

  | cell | as run | strict |
  |---|---|---|
  | Gemma L2 | BEYOND (+0.105) | **not beyond (−0.013)** — artefact |
  | **Llama L2** | BEYOND (+0.253) | **BEYOND (+0.151)**, p = 0.0149 |

  So **Llama L2 is the one cell where the residual stream beats every baseline available**, and it
  clears even against reply length alone. Every other cell on every host reduces to length, static
  code shape, or — for the dispatcher-span probe, unanimously across all three hosts — text
  repetition.

  - **H-LL2 (primary).** Does Llama's L2 advantage over the strict baseline survive on independent
    draws?

- **Setup:**
  ```
  draws   five NEW draws (10-14) for Llama L2, same host, same regime, --max-new-gen 8192
  score   draws 10-14 ONLY; the discovery draws 0-9 are not pooled in
  stat    mean rho over all 32 layers minus max(length, static, combined), selection-free
  rows    non-terminating rows excluded and counted, per the 2026-09-02 category
  ```
  Llama L2 excluded 33 non-terminating rows in the discovery set, the second-highest cell in the
  matrix, so the exclusion count is reported with the verdict rather than buried.

- **Decision rules — frozen:**

  - **REPLICATED** iff beats-strict **≥ +0.075** on the new draws *and* the permutation p on the
    mean statistic is **< 0.05**. The discovery value is +0.1507; **+0.075 is half of it**, the
    standard replication convention, set here rather than after seeing the number.
  - **NOT REPLICATED** otherwise. Then the cell joins Gemma L2 as a discovery-set artefact, and
    the three-host conclusion is uniform: **nothing in the residual stream beats a surface
    statistic on any host at any tier.**

  **Secondary, reported either way:**
  - Llama L3 as the sibling control. It shares the relational route and did **not** survive
    (+0.0107 strict). If L3 now clears while L2 does not, the effect is not L2-specific and the
    original framing was wrong.
  - The **degeneracy diagnostic** (`combined_degraded_vs_best_single`) is reported for every cell.
    A replication resting on a degenerate baseline does not count, which is the whole reason the
    strict baseline exists.

  **Standing constraints:** full sample only; the strict baseline is the verdict and the combined
  baseline is never used alone; and this cannot revive Gemma L2, which the strict baseline already
  refused. **Both outcomes are publishable and are written down as such now** — a single surviving
  cell out of nine, on the weakest host, is at least as likely to be noise as signal, and this
  programme has retracted four positives that looked stronger.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** five draws on Llama L2 and L3, then `p1b_l2_mechanism.py` restricted to the new draws.
