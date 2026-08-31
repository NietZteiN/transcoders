# 03 — Owed fixes and unlogged results

All diagnosed, none applied. These are the things a reviewer — or a future you — will trip over.
Ordered by how much damage they do if left.

---

## 1. B5 ran on 2026-08-26 and has **no ledger entry at all**

The project's protocol (`transcoders/CLAUDE.md` §6) makes `log/` the source of truth and requires
a dated entry per working day per thread. B5's results exist **only** in
`data/nla/b5/cells_scored.json` and `reports/2026-08-26_full-results/RESULTS.md`.

**Worse: the run persisted no per-case rows.** Nothing new appeared under
`allocation_replication/results/runs/`, and its stage-4 aggregate regenerated the pre-existing RQ1
grid rather than writing B5's own. So **the six cells cannot be re-analysed from disk** — not
re-scored, not clustered differently, not given a second seed without re-running from scratch.

*(P0.3 does **not** have this problem — it persists per-run `score.json` keyed by run tag under
`artifact/artifacts/obfuscation/result/<model>/<snippet>/adversarial_rename/p03_<cell>/run_NNNN/`.
Keep that property in anything new.)*

**Owed:** `log/nla-harness/2026-08-26_b5-composition.md`, recording the gate pass
(62.746 vs banked 61.710, |Δ| = 0.0104 against a 0.02 tolerance), the six cells, the cluster-
bootstrap CIs, the collision control (+0.10), and the two bounds — one seed, and the model tension
(V1 directions exported from `Qwen2.5-7B-Instruct`, injected into the Coder sibling that **B1
refuted transfer to**).

---

## 2. Both log indexes are stale

`log/README.md` and `log/nla-harness/README.md` both still say **"Last updated: 2026-08-14"**, and
their Timeline / Entries tables stop before the 08-16, 08-17 and 08-26 entries. The index no longer
reflects the ledger it indexes.

**Owed:** refresh both tables, and add the Phase-0 pre-registration + results rows.

---

## 3. Phase-0 results are unlogged

P0.1 is complete (verdict HARD, argmax coherence L13) and 8 of 10 P0.3 cells finished, but no
results entry exists — only the pre-registration.

**Owed:** `log/nla-harness/2026-08-28_p0-triage-results.md` once the run completes. If it never
completes, log the partial state and say so; a stopped run recorded honestly is worth more than a
gap.

---

## 4. The 2026-08-17 programme report is superseded and does not say so

`reports/2026-08-17_believe-the-lie-programme/REPORT.md` presents B0's family contrast
(**+7.02 [+2.00, +10.09]**) as the paper's remaining spine and says it is "currently being extended
from one model to three." **E5b (2026-08-26) flipped that sign at a second seed.** The report needs
a pointer to the seed result; the FSE recommendation built on B0 needs revisiting.

**What survives of B0:** `arith_rewriting` is clearly negative at both seeds (−8.22, −5.65) — the
one cell CodeSteer itself flagged. A single-transform claim with two seeds is defensible where a
five-transform family average is not.

---

## 5. `CLAUDE_SCRATCHPAD.md` — now current, keep it that way

Protocol §0 requires it to track live task state. It had been stale since 2026-08-03 (Phase-0
scaffold text) and was rewritten 2026-08-28 with the current Phase-0 state. **Update it whenever
you start or stop a long GPU run** — it is the first thing a fresh session reads.

---

## 6. Wording errors to fix before external use

From the 2026-08-28 audit (detail in `00_STATE.md` §Corrections):

- **"identical activations"** — false, in the 2026-08-13 entry and both 08-14 reports. The two
  framing arms are separate captures; **0 of 14 read positions overlap**. Say "identical *code*,
  two prompts."
- **"64 tests"** — it is 86 Python tests plus 3 JS regression tests.
- **"twice in one month"** for interim reversals — it is **six**.
- **`configs/b4_steering.yaml`** claims the sweep covers "three injection positions"; the run
  manifest shows `--positions last_prompt` only. Trim to the alpha range, which is what the log
  entry correctly reports.

---

## 7. Cheap unrun tests that keep not getting run

Each answers something live, none needs much GPU:

| test | cost | what it settles |
|---|---|---|
| reply length alone vs length + faithfulness | **no GPU** | settles B6 judge-free; answers whether `rt_cos` adds anything at all |
| N10b per-item split-half reliability | **no GPU** | whether the one positive can rank a single sample or only separate populations — newly urgent since the score is known to lean on the generic term `payload` |
| verbalizer vs subject for the Python prior | ~1 GPU-h | whether the 37%-wrong-language prior lives in the **verbalizer** or the **subject's residual stream** — open since 2026-08-07 |
| unified-table sweep | no GPU | `rt_cos`/`act_norm` by read class × corpus with length as covariate, over all 14,619 rows at once |

---

## 8. Integrity work required regardless of which paper wins

- **B0's original four transforms at seed 2000** (~20 GPU-h) so the old-vs-new contrast exists at
  two seeds — currently it cannot be computed at the second seed at all.
- **Re-run the seed-1000 grid *seeded*** — it predates the `--seed` artifact fix, so those runs are
  uncontrolled draws at temperature 0.9 rather than a controlled seed.
- **Treat 2 seeds as the floor, not the target.** A measurement that flips a five-transform family
  average between draws needs 3+ before any sign-dependent claim.
- **Hand-validate the malware capability labels** (~100 samples, two labellers, ≥85% agreement)
  before any per-capability number is quoted again. `UNASSIGNED` is 51 of 183 and never means
  benign; 78% of the labelled files are `REMOTE_EXEC`.
