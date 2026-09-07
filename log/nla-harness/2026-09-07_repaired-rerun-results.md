### Target Date: 2026-09-07 (H-W23 — the meaning effect replicates at 3.6× the span set **and holds in JavaScript**; the Python-only bound is lifted)

**Thread:** nla-harness · **Job:** 380671 (h200, **55:20**, COMPLETED, 0 errors) ·
**Resolves:** [`2026-09-06_repaired-rerun-prereg.md`](2026-09-06_repaired-rerun-prereg.md) under
[`2026-09-06_hw17-language-confound.md`](2026-09-06_hw17-language-confound.md).

- **Coverage, after the repair.** `diag` = **471/476 spans anchored for both L0 and L1, 0 with no
  L0 anchor** (was 375 and 131, with 88 unanchored). The item set grows **49 → 60**: eleven items
  previously contributed nothing because every one of their spans was unanchorable.
  SELF identity **−0.0175** (tol 1.0) → **PASS**.

- **Results.**

  | arm | G_sum | 95 % CI |
  |---|---|---|
  | `T_L0_all` | +41.52 | [+36.84, +46.21] |
  | `T_L2_all` (true names, flattened) | +41.06 | [+36.51, +45.82] |
  | `T_L0_sub` | +41.52 | [+36.94, +46.42] |
  | `T_L1_sub` (**nonsense** names) | **+20.47** | [+17.31, +23.69] |

  **H-W23c ✓** — `T_L0_all − T_L2_all` = **+0.46** [+0.07, +0.88], does not clear. Structure is worth
  ~1 % of the effect, replicating job 379819's **+0.66** on a 26 % larger span set. The expanded set
  is comparable to the old one, which is what this arm was for.

  **H-W23a ✓ REPLICATED** — `T_L0_sub − T_L1_sub` = **+21.05** [+17.80, +24.28], clearing on
  **58 of 60 items**, against H-W17a's **+21.82** [+15.52, +28.26] on 131 spans / 20 Python items.
  A 3.6× expansion of the span set moves the point estimate by **0.77 nats** and tightens the CI.

  **H-W23b — the pre-registered language split:**

  | group | n | `T_L0_sub` | `T_L1_sub` | gap | 95 % CI | clears | L1/L0 |
  |---|---|---|---|---|---|---|---|
  | all | 60 | +41.52 | +20.47 | **+21.05** | [+17.80, +24.28] | ✓ | 0.493 |
  | python | 30 | +39.94 | +22.11 | **+17.83** | [+13.71, +22.33] | ✓ | 0.554 |
  | **javascript** | **30** | +43.11 | +18.83 | **+24.28** | [+19.57, +28.84] | **✓** | 0.437 |

  **Verdict `W16-MEANING-CARRIES-IT`.**

- **What this settles.**
  - **The Python-only bound recorded on 2026-09-06 is LIFTED.** JavaScript clears the bar on its own
    30 items, with a CI that excludes the threshold — and the JS gap (**+24.28**) is *larger* than
    Python's (+17.83), with overlapping CIs. The effect is not language-specific.
  - **Identifier meaning carries ~51 % of the item component and removal of the decoy ~49 %**
    (`L1/L0` = 0.493 overall). Both halves are large. The "removal suffices" reading — which I
    pre-registered as my expectation and which a confound briefly supported — is refuted on the
    clean contrast in **both** languages.
  - **Structure contributes ~1 %**, twice measured, on 375 and now 471 spans.

- **Observations.**
  - **The repair earned its keep beyond the language question.** It recovered **11 items** that were
    silently contributing nothing, and closed `no_L0_anchor` from 88 to 0. Every earlier W result
    — the fidelity ceiling, the null battery, the dose curve — ran on 375 spans and 49 items because
    of a metadata defect, not a scientific constraint.
  - **The replication is the reassurance the prereg asked for.** It stated that a large H-W23a/H-W17a
    discrepancy would be *"a warning about the repair, not a result"*, because the repair's 1.000
    validation accuracy was measured only on entries the pipeline could already pair. **+21.82 →
    +21.05 on a 3.6× larger, differently-composed span set** is about as good as that check gets.
  - **Item-level `T_L0_sub` = `T_L0_all` exactly** (+41.52 both), as it must once every anchored span
    is in the subset — a free internal consistency check that the two position sets really did merge.
  - Behavioural veto passed for both generated arms.

- **Updated decomposition of the item component** (all measured, all on this corpus):
  **meaning ≈ 51 % · decoy removal ≈ 49 % · control-flow structure ≈ 1 %.**

- **New questions / new hypotheses.**
  - **H-W25:** re-run the fidelity ceiling (H-W7) and null battery (H-W9/W12) on the repaired
    471-span, 60-item set. They are the two results the paper leans on hardest and both currently
    rest on the pre-repair 375/49. Expected to move little — but "expected" is what H-W16b taught me
    to distrust.
  - **H-W26:** the JS gap being *larger* than Python's is unexplained. With overlapping CIs it is
    not a finding, but the two languages differ in how much a nonsense name recovers (0.437 vs
    0.554) and that is worth a look before any cross-language claim is published.

- **Next Steps:** H-W25 — the two headline results deserve the repaired corpus.
