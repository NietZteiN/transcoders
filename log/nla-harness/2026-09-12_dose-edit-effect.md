# 2026-09-12 · H-C8 — the EDIT effect rises with data too, and shows NO saturation. Buying data buys the thing we want.

**Thread:** nla-harness · **Experiment:** H-C8, analysis of the H-C4 dose points · **Status:** **no new GPU**
Extends [`2026-09-12_dose-response-results.md`](2026-09-12_dose-response-results.md) (append-only: that entry
stands; its observation 5 said the edit's data-sensitivity was "the next thing to test, not to assume" — this
tests it, and the answer reverses the cautious reading).

- **Hypotheses / what we're testing:** H-C4 showed *transport* (`S_c3`) is data-limited. That raises the ceiling
  the edit works against; it is not the same as the **edit itself** improving. If the content-specific edit effect
  `SPEC = S_edit − S_foreign` were flat in volume, then H-C7 (buying 2× data, ~35 GPU-h) would buy a better
  ceiling and not the thing the programme is actually after. **H-C8:** does `SPEC` rise with training volume?
  The frozen **+1.46-nat** bar and the paired-per-item form are reused unchanged from
  [`2026-09-12_dose-response-prereg.md`](2026-09-12_dose-response-prereg.md) for comparability rather than
  re-derived.

- **Setup:** zero GPU. `S_edit` and `S_foreign` at L7 are standard arms of `nla_ml_gate.py` and were already
  written by job **391522** into `/scratch/juno/jvl210002/nla_ml_dose/f{025,050,100}/gate/gate_rows.jsonl`
  (plus the banked `data/nla/ml/gemma4b/gate/gate_rows.jsonl`). Same 60 items / 471 spans, seed 20260724,
  N_BOOT 10 000, cluster bootstrap over items, paired per item. Nothing was re-run and no arm was added for
  this question.

- **Results:**

  | dose | `edit` | `foreign` | `rt` | **`SPEC` = edit − foreign** | `edit − rt` | SPEC as % of the +73.15 ceiling |
  |---|---|---|---|---|---|---|
  | 25 % | +28.29 | +25.80 | +25.40 | **+2.49** | +2.89 | 3.4 % |
  | 50 % | +30.41 | +24.93 | +24.34 | **+5.49** | +6.07 | 7.5 % |
  | 100 % | +33.67 | +24.95 | +23.78 | **+8.72** | +9.89 | 11.9 % |
  | 100 % banked | +32.06 | +23.88 | +23.31 | +8.18 | +8.75 | 11.2 % |

  **Paired steps:**
  - `SPEC(100 %) − SPEC(50 %)` = **+3.23 [+1.01, +5.65]** — clears the +1.46 bar, CI excludes 0
  - `SPEC(50 %) − SPEC(25 %)` = **+3.00 [+0.90, +5.22]** — clears, CI excludes 0
  - `SPEC(100 %) − SPEC(25 %)` = **+6.23 [+3.20, +9.32]**
  - control `SPEC(100 % re-trained) − SPEC(banked)` = +0.53 [−1.95, +3.20] ✓ within noise

- **What worked / hypothesis verdict:**
  - **H-C8 → the edit effect IS data-limited, and both steps clear.** `SPEC` runs +2.49 → +5.49 → +8.72, and it is
    **not merely tracking the ceiling**: as a share of the fixed +73.15 ceiling it rises 3.4 % → 7.5 % → 11.9 %.
    `foreign` is essentially flat across the whole range (+25.80 / +24.93 / +24.95), so the growth is in the
    content-specific component, not in the non-specific decoy-removal part. `edit − rt` tells the same story
    (+2.89 → +6.07 → +9.89).
  - **The key asymmetry: transport saturates, the edit does not.** Increments per doubling —
    transport ratio: **+0.264 then +0.122** (halving, ratio 0.463) · `SPEC`: **+3.00 then +3.23** (flat, if
    anything rising). So on this range the edit effect is growing roughly *linearly in log volume with no visible
    saturation*, while transport is clearly bending toward its bound of 1.0.
  - **This reverses the cautious note in the H-C4 entry** (observation 5) and materially strengthens **H-C7**:
    buying 2× data is forecast to move `SPEC` from +8.72 to **≈ +12** and transport to ≈ 0.966 — i.e. it buys both
    the ceiling *and* the specific effect. Registered as a falsifiable forecast, not a result: two increments
    extrapolated one step.

- **Observations:**
  1. **This is the first thing in the whole programme that moves the edit step.** Every prior attempt was
     geometric — layer choice (H-S1), head targeting (H-S2/H-S3), layer count and the β write form
     (H-S5–H-S9) — and the specific effect stayed pinned near +8–9 nats, ~12–14 % of the ceiling, in all of them.
     Training volume moves it: **+2.49 → +8.72, a 3.5× change**, larger than anything the geometry produced.
  2. **It reframes the standing `M-NO-GAIN` result.** Phase B concluded the NLA edit delivers nothing above its
     nulls at scale, and the W family found the released 12B pair's edit was +19.47 against random +19.60 — no
     gain. H-C8 says our edit's weakness is at least partly a **budget** artifact on a measurable trajectory,
     rather than a fixed property of the read→edit→reconstruct channel. It does **not** show the released pair's
     null result is a budget artifact — that pair is trained at ~10× our volume *with* RL, so its flat edit is
     evidence against a pure volume story at the top end, and the two facts have to be reconciled by H-C7.
  3. **Caveat that keeps this honest:** the H-C4 entry's own observation 4 stands — a 25 % pair is not a live
     instrument (`fve` 0.1407 < 0.20), so the lowest point of this curve comes from a pair that would be excluded
     everywhere else in this thread. The 50 %→100 % step, which clears on its own, does not depend on it.
  4. Volume and optimiser steps are still confounded (`--train-frac` at 1 epoch varies both). With both H-C4 and
     H-C8 now `DATA-LIMITED`, the same-steps arm is worth running *before* H-C7, because it is cheap and it decides
     whether "more data" or "more optimisation" is being bought.

- **New questions / new hypotheses:**
  - **H-C10 (cheap, do before H-C7):** same-steps control — 50 % of the rows for 2 epochs vs 100 % for 1 epoch,
    equal optimiser steps. If 2× epochs on half the data recovers the 100 % `SPEC`, the lever is optimisation, not
    data, and H-C7's corpus regeneration is unnecessary. ~1 GPU-h at L7 (banked acts, banked vectors path).
  - **H-C7 (conditional on H-C10):** 2× volume at 4B L7. Forecast to freeze before running: transport ratio
    **≈ 0.966**, `SPEC` **≈ +12**. `DOSE-MODEL-HOLDS` within a band to be fixed in that prereg.
  - **H-C11:** is the released pair's flat edit (+19.47 vs random +19.60) reconcilable with a volume story at all?
    Its `SPEC`-equivalent should be recomputed on the repaired anchoring for a like-for-like comparison against our
    +8.72 before H-C7 is justified on "volume closes everything" grounds.

- **Next Steps:** run **H-C10** (~1 GPU-h) before committing the ~35 GPU-h of H-C7. Index this entry; update the
  thread README ledger and `CLAUDE_SCRATCHPAD.md`.
