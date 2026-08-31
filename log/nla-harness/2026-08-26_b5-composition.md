### Target Date: 2026-08-26 (B5 — the two levers do not measurably compose)

**Written 2026-08-31 to close an owed ledger entry** (`nla/continuation/03_LEDGER_DEBTS.md` §1).
B5 ran on 2026-08-26 and its results existed only in `data/nla/b5/cells_scored.json` and
`reports/2026-08-26_full-results/RESULTS.md`, never in the ledger, which
[`../../CLAUDE.md`](../../CLAUDE.md) §6 makes the source of truth. Every number below is quoted
from those two artifacts; nothing is re-derived, and the date is the run's, not the writing's.

- **Hypotheses / what we tested:**
  B4 (belief steering) is refuted and B0 (attention steering) is unreplicated. **B5 asks whether
  the two levers compose** — whether a belief direction written at layer 20 plus attention
  reallocation over layers 20–27 recovers accuracy that neither achieves alone.

- **Setup:**
  ```
  stimuli  164 Java programs, adversarial_rename, 1,930 cases per cell — identical across cells
  cells    none · attn · nla · both · rand · disjoint   (6 x 1,930)
  gate     alpha = 0 makes the belief channel a verified no-op, so the patched runner with
           attention steering ON must reproduce the banked obf_steer number
  data     data/nla/b5/cells_scored.json · gate_verdict.json
  ```

  **Validation gate PASSED**: 62.746 against a banked 61.710, |Δ| = **0.01036** against a 0.02
  tolerance. The integration did not change the measurement.

- **Results:**

  Raw cell accuracies (P@1): none **64.04** · attn 62.75 · nla 61.92 · both **64.56** ·
  rand 62.23 · disjoint **64.66**.

  | contrast | estimate | 95% CI (cluster bootstrap over programs) |
  |---|---|---|
  | attention only | −1.30 | [−6.31, +3.61] |
  | belief only | −2.12 | [−6.76, +2.39] |
  | both levers | +0.52 | [−3.37, +4.38] |
  | **both − attention** | +1.81 | [−2.61, +6.16] |
  | **both − random-direction control** | +2.33 | [−2.17, +6.74] |
  | **disjoint − both** *(collision check)* | +0.10 | [−3.93, +4.26] |

  **Not one interval excludes zero.** The point estimates read like a composition result — the
  pair beats either lever alone and beats a random direction by 2.33 — and every one dissolves
  against its interval.

  **The collision control did its job.** Attention steering occupies layers 20–27 and the
  autoencoder lives at layer 20, so "both on" could have been a *collision* rather than
  composition. Moving attention to layers 21–27 changed the result by **+0.10**. Not a collision,
  but also not an effect.

- **What worked / hypothesis verdict:**
  - **B5 → NULL.** The levers do not measurably compose on this corpus at this sample size.
  - The gate passed, so this is a null about the manipulation and not about the plumbing.

- **Observations:**
  - **Two bounds on this null, both recorded at the time:** it is **one seed**, and the V1
    directions were exported from `Qwen2.5-7B-Instruct` and injected into the Coder sibling that
    **B1 refuted transfer to** (rt_cos 0.694 vs 0.864). A null under a transfer the programme has
    already shown to be degraded is weaker evidence than a null under a validated one.
  - **The run persisted no per-case rows.** Nothing new appeared under
    `allocation_replication/results/runs/`, and its stage-4 aggregate regenerated the pre-existing
    RQ1 grid rather than writing B5's own. **The six cells cannot be re-analysed from disk** — not
    re-scored, not clustered differently, not given a second seed without re-running from scratch.
    P0.3 does not have this problem and persists per-run `score.json` keyed by run tag; **that
    property should be required of anything new.**

- **New questions / new hypotheses:**
  - A second seed would cost what the first did and is the single thing that would move this from
    "null at n = 1 seed" to a reportable null — E5b showed a five-transform family average flipping
    sign between draws on this exact pipeline.
  - If B5 is ever re-run, persist per-case rows first. The absence of them is the reason this entry
    can only restate the aggregate.

- **Next Steps:** *(historical entry; B5 was superseded by the Phase-0 triage, which asked why the
  levers fail rather than whether they compose)*
