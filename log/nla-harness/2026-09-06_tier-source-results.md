### Target Date: 2026-09-06 (H-W15 ✓ quantity not position · H-W16a ✓ structure is worth ~nothing · **H-W16b's verdict is an artifact of my own fallback, and it flattered my stated prediction**)

**Thread:** nla-harness · **Job:** 379819 (h200, **45:24**, COMPLETED, 0 errors, 49/49 items) ·
**Resolves:** [`2026-09-06_tier-source-prereg.md`](2026-09-06_tier-source-prereg.md).

- **Results.** SELF identity **−0.0096** nats (tol 1.0) → **PASS**, identical to job 379075.

  | arm | G_sum | 95 % CI | positions |
  |---|---|---|---|
  | `T_L0_all` (anchor) | **+45.71** | [+40.82, +50.88] | 375 |
  | `T_L2_all` (true names, **flattened**) | **+45.05** | [+40.25, +50.12] | 375 |
  | `T_L0_sub` | +30.90 | [+24.98, +37.40] | 190 |
  | `T_L1_sub` (**nonsense** names) | +22.20 | [+18.92, +25.62] | 190 |
  | forward `P_1 / P_2 / P_4` | +10.29 / +17.95 / +28.19 | | |
  | reverse `R_1 / R_2 / R_4` | +11.40 / +18.93 / +30.59 | | |

  **H-W16a:** `T_L0_all − T_L2_all` = **+0.66** [+0.26, +1.11] → does not clear.
  **H-W16b:** `T_L0_sub − T_L1_sub` = **+8.70** [+5.14, +12.69] → does not clear.
  **H-W15:** R−P = +1.10 / +0.98 / +2.40, every CI containing 0 → **quantity, not position**.
  Frozen verdict **`W16-NEITHER-REMOVAL-SUFFICES`**.

- **What worked / hypothesis verdict.**
  - **H-W15 ✓ SUPPORTED — saturation is about QUANTITY, not which spans.** Forward and reverse
    ladders select disjoint span subsets at k = 1, 2 and yet agree at every k. The dose curve's
    shape is a property of *how many* positions carry a clean vector, not of *which*.
    *(Reverse runs consistently ~1–2 nats higher at all three k. Every CI contains zero so the
    frozen rule is satisfied, but the sign is consistent and is flagged, not buried.)*
  - **H-W16a ✓ REFUTED as pre-registered, and the number is striking.** Patching from **L2** —
    true identifiers, **control flow flattened** — recovers **98.6 %** of the clean-state effect
    (+45.05 vs +45.71). **Destroying the program's control-flow structure costs 0.66 nats of 45.71.**
    Whatever these identifier positions carry, it is essentially independent of the structure the
    program was written in. Coverage here is **375/375 spans**, all genuine, so this result is
    unaffected by the defect below.
  - **H-W16b ✗ — THE VERDICT IS AN ARTIFACT AND MUST NOT BE REPORTED AS A FINDING. See below.**

- **⚠️ Defect #7, and the worst kind: it produced the result I had predicted.**

  `tier_anchor()` falls back to the true term when a tier's `rename_map` has no entry — I added
  that deliberately, and recorded it in the prereg, because it lifted L1 coverage from 131 to 190
  spans. **It lifts coverage by adding spans that cannot express the contrast.** An "unrenamed"
  span is one L1 never renamed, so its L1 anchor **is the same identifier as its L0 anchor** — the
  arm is near-identical to `T_L0_sub` there by construction, and contributes ≈ 0 to the gap.

  How bad it is:

  | | items | anchored spans | of which null-by-construction |
  |---|---|---|---|
  | full L1 coverage | 20 | 131 renamed + 21 unrenamed | 14 % |
  | **partial coverage** | **29** | **0 renamed + 38 unrenamed** | **100 %** |

  **All 29 partial-coverage items contribute structural zeros** — visible in the run log as
  `sub L0=+17.8 L1=+17.8`, `+24.4/+24.4`, `+27.9/+27.9`. So the pooled **+8.70** averages 20 items
  carrying a real contrast against 29 carrying an exact null.

  Restricted to the 20 items whose spans are genuinely renamed:
  **`T_L0_sub` +51.70 vs `T_L1_sub` +30.22 — gap +21.48, ratio 0.584.** That **clears the +12.11
  bar**, in the opposite direction to the pooled verdict.

  **I pre-registered the prediction that H-W16b would fail — that nonsense would work as well as
  truth — and wrote that the failure "is the more interesting one".** The confound delivered exactly
  that prediction. Had I not checked why the sub-arms were printing identical values, I would have
  filed "the repair is removing the decoy, not restoring meaning" as the programme's headline, from
  an artifact, in agreement with my stated prior. **The pre-registration did not protect against
  this**, because the defect was inside an implementation choice the prereg described approvingly
  rather than inside the decision rule.

  **What stands and what does not:** `W16-NEITHER-REMOVAL-SUFFICES` is the frozen verdict and is
  recorded as fired, **but it is not evidence about meaning.** The restricted +21.48 is **post-hoc
  and on a self-selected subset** (items where L1's renaming happened to be thorough), so it is a
  **lead, not a finding**. Neither number may be quoted as settling H-W16b.

- **Observations.**
  - Provisionally, and pending the clean re-run: **structure ≈ 0, meaning ≈ 42 % of the item
    component, removal of the decoy ≈ 58 %.** Both halves are real; the earlier "removal suffices"
    framing is *not* supported.
  - `T_L0_all` = **+45.71 for the fourth time**, four jobs, four nodes.
  - Behavioural veto passed for both generated arms (`T_L2_all` acc 0.714 = baseline; `T_L1_sub`
    0.673).

- **New questions / new hypotheses.**
  - **H-W17 (the clean re-run, and it is required before any H-W16b claim):** restrict `T_L1_sub`
    and its matched `T_L0_sub` to the **131 `rename_map`-anchored spans only** — a *span-level*
    restriction, not the item-level selection used post-hoc here, so it does not inherit the
    selection bias. One job on the banked pool.
  - **H-W18:** if structure is worth 0.66 nats at identifier spans, is the L2/L3 relational-overload
    route simply invisible at these positions? That would predict a *different* set of causal
    positions for L2 — and it is E3's question, reached from the other side.

- **Next Steps:** pre-register and run **H-W17** before anything else in this thread. No statement
  about meaning-vs-removal leaves here until it lands.
