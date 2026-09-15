### Target Date: 2026-09-13 (accuracy under NLA steering — the result the user asked for)

- **Hypotheses / what we're testing:** the user asked directly and repeatedly for this number — *"The result I
  want to see is if accuracy improved by steering the nla"* — and it runs although the H-S9 gate **did not fire**
  (best config `band` +9.85 vs best single layer +8.18, gap +1.67 < +3.00). That is recorded as the user's call,
  not as the gate being met. **Per the pre-registration
  ([`2026-09-12_accuracy-prereg.md`](2026-09-12_accuracy-prereg.md)) no verdict word was committed in advance and
  none is assigned here**; the entry reports the pre-specified readouts and the pre-declared bound. The bound,
  frozen before the run: *"acc_l0 0.567 → acc_l1b 0.533; 2/60 net headroom, 7 flippable; a PERFECT rescue of all 7
  against the measured 6–9/60 same-condition churn gives McNemar p = 0.092. No positive significance claim is
  possible at n = 60."*

- **Setup:** job **391968**, `sbatch nla/scripts/nla_accuracy_sbatch.sh`, node **g-08-05** (H200 NVL),
  `2026-09-13T04:37:40Z → 08:07:54Z`, **3:30:15 elapsed, rc=0, 3.50 GPU-h** (prereg estimated 3–5).
  sha256 (12): `nla/src/nla_accuracy.py` `2486fd57e2b5` · `nla/scripts/nla_accuracy_sbatch.sh` `e6255f1af35a`.
  Host Gemma-3-4B-it, same 60 items / 471 repaired spans (`select_pairs`, seed 20260724), banked vectors at
  `data/nla/ml/gemma4b/gate/vectors/`, `--deterministic` OFF, N_BOOT 10 000, `MAX_NEW_GEN` 1100,
  grading `steer_run.graded`. Arms exactly as pre-registered: `noop` · `c3_band` · `edit_band` · `foreign_band`
  (band = **L2–L13, β = 0.35**, the best specific-effect config from
  [`2026-09-12_beta-layerset-results.md`](2026-09-12_beta-layerset-results.md)) · `edit_single`
  (**L2, β = 1.0**, the banked `best_single_edit_layer`) · `prompt` (mandatory prompting baseline per CLAUDE.md §4).
  Sampling n = 8 via `num_return_sequences` at **batch 1, no padding** (left padding would break
  `PositionReplacer`'s absolute-position targeting). `n_prompt_skipped` **0**. Three stages ran: a 2-item smoke,
  a standalone **greedy** pass, and the **sampled** pass (which re-measures greedy alongside it). Outputs
  `data/nla/ml/gemma4b/gate/accuracy{,_greedy}/accuracy_stats.json`.

- **Results.** Sampled pass, n = 8 per item, 60 items — the powered readout:

  | arm | greedy acc | Wilson 95 % | pass rate | boot 95 % | vs `noop`: fixed / broke | McNemar p | paired Δ rate |
  |---|---|---|---|---|---|---|---|
  | `noop` | 0.533 | [0.409, 0.654] | 0.527 | [0.417, 0.635] | — | — | — |
  | `c3_band` | 0.533 | [0.409, 0.654] | 0.521 | [0.413, 0.631] | 5 / 5 | 1.000 | **−0.006 [−0.081, +0.069]** |
  | `edit_band` | 0.533 | [0.409, 0.654] | 0.535 | [0.427, 0.646] | 5 / 5 | 1.000 | **+0.008 [−0.038, +0.056]** |
  | `foreign_band` | 0.450 | [0.331, 0.575] | 0.460 | [0.352, 0.569] | 5 / 10 | 0.302 | −0.067 [−0.140, +0.004] |
  | `edit_single` | 0.400 | [0.286, 0.526] | 0.406 | [0.310, 0.508] | 4 / 12 | 0.077 | **−0.121 [−0.206, −0.035]** |
  | `prompt` | 0.533 | [0.409, 0.654] | 0.538 | [0.419, 0.654] | 3 / 3 | 1.000 | +0.010 [−0.023, +0.042] |

  Standalone greedy pass (same code, separate generations): `noop` **0.517** · `c3_band` 0.533 · `edit_band` 0.533
  · `foreign_band` 0.467 · `edit_single` **0.367** · `prompt` **0.567**. Greedy McNemar vs `noop`:
  `c3_band` 5 fixed / 4 broke p = 1.000 · `edit_band` 5/4 p = 1.000 · `foreign_band` 4/7 p = 0.549 ·
  `edit_single` **4/13 p = 0.049** · `prompt` **3/0 p = 0.250**.

  The 7 flippable items (L0-correct, L1b-wrong; `noop` = 0/7 by construction):

  | arm | greedy k/7 | sampled rate on the 7 | boot 95 % |
  |---|---|---|---|
  | `noop` | 0/7 | 0.232 | [0.071, 0.446] |
  | **`c3_band`** | **3/7 (0.429)** | **0.536** | [0.232, 0.857] |
  | `edit_band` | 2/7 (0.286) | 0.250 | [0.018, 0.518] |
  | `edit_single` | 2/7 (0.286) | 0.196 | [0.054, 0.357] |
  | `prompt` | 1/7 (0.143) | 0.339 | [0.089, 0.607] |
  | `foreign_band` | 1/7 (0.143) | 0.161 | [0.018, 0.411] |

- **What worked / hypothesis verdict:** no hypothesis was pre-registered with a verdict word for this stage and
  none is assigned. Reporting against the pre-specified readouts:
  - **Overall accuracy:** neither NLA arm separates from `noop` on the powered readout —
    `c3_band` **−0.006 [−0.081, +0.069]** and `edit_band` **+0.008 [−0.038, +0.056]**, both McNemar p = 1.000 with
    5 items fixed and 5 broken apiece. The prompting baseline is the same size (+0.010 [−0.023, +0.042]).
  - **The one interval that excludes 0 is a harm:** `edit_single` (single layer L2, **full** replacement β = 1.0)
    at **−0.121 [−0.206, −0.035]** paired rate, 4 fixed / 12 broken, greedy McNemar **p = 0.049** (13 broken
    there). The banked single-layer write that scores best on `G_sum` **damages task accuracy**, and the
    β = 0.35 band that H-S6 introduced to stop multi-layer writes damaging the host does not show that harm
    (+0.008). That is a new, independent argument for β from the accuracy side.
  - `foreign_band` — the content null — is also net-negative (−0.067 [−0.140, +0.004], 5 fixed / 10 broken),
    i.e. writing *another item's* true term costs more than it gives, as it should.
  - **On the 7 flippable items `c3_band` is the largest mover** (greedy 3/7, sampled 0.536 vs `noop` 0.232), and it
    is the only arm above the prompting baseline there. **This cannot be called an effect:** n = 7, the bootstrap
    CIs overlap heavily ([0.232, 0.857] vs [0.071, 0.446]), and the pre-declared bound says a *perfect* 7/7 rescue
    would still only reach p = 0.092. It is recorded as the shape the pre-registration said to look for, at a
    sample size that cannot certify it.
  - **The bound was not a formality.** `noop` itself came out 0.517 in the standalone greedy pass and 0.533 in the
    sampled pass's greedy — a 1-item swing in the *same job under the same condition* (`--deterministic` OFF),
    which is the 6–9/60 churn band made visible. Any greedy difference of 1–2 items in this table is noise.

- **Observations:**
  - **This lands inside a bound measured the same day and reported before it.** The tier ladder
    ([`2026-09-13_tier-ladder-results.md`](2026-09-13_tier-ladder-results.md), job 392094, `ERASURE-FLOOR`) found
    that *perfect* erasure in token space buys **−0.035 [−0.100, +0.027]** over the adversarial trap, while only
    reconstruction carries headroom (+0.083 [+0.015, +0.160]). The erasure-shaped arms here (`edit_band`,
    `edit_single`, `foreign_band`) were therefore bounded at ≈ 0 before this job reported, and they came in at
    ≈ 0 or worse. The one arm with headroom above it by that decomposition is `c3_band`, the meaning-installing
    write — and `c3_band` is exactly the arm that moves the flippable subset. **The two runs agree, and they were
    designed and frozen independently.**
  - **`c3_band` at −0.006 overall with 5 fixed and 5 broken is the informative shape of the null the prereg
    predicted:** the strongest NLA write this project has — ~91 % of the clean state's `G_sum` effect — reshuffles
    which items are right without moving the count. `G_sum` and task accuracy are not the same quantity, which is
    the programme's standing message stated in accuracy units for the first time.
  - **Where the prompting baseline stands** (CLAUDE.md §4 requires it): it broke **zero** items in the greedy pass
    (3 fixed / 0 broken) while every steering arm broke 4–13. On the flippable subset it is second to `c3_band`.
    Cheap, safe, and not beaten on overall accuracy by anything here.
  - No silent-failure signature: 0 prompt-arm skips, all 60 items in every arm, both stat files written.

- **New questions / new hypotheses:**
  - **H-S13:** is the `edit_single` harm a β effect or a layer effect? `edit_single` differs from `edit_band` in
    *both* (L2 alone vs L2–13, β 1.0 vs 0.35). One cell — L2 alone at β = 0.35 — separates them, ~0.6 GPU-h on
    banked vectors. Worth it: "full replacement damages the task while partial does not" is a claim the steering
    literature should care about, and right now it is confounded.
  - **H-S14:** the flippable-7 signal for `c3_band` is the only thing in this run pointing anywhere. The corpus
    doc (`docs/nla_flippable_corpus_scoping.md`) costs ~270 screened items for ~30 flippable at the measured 11 %
    yield; with 30 the perfect-rescue bound moves from p = 0.092 to well inside 0.05, so the question becomes
    answerable rather than bounded. **This is the decision the user's question actually turns on**, and it is a
    corpus purchase, not a method change.
  - **H-S15:** does the parse-rate story from the tier ladder apply here too — are the 12 items `edit_single`
    "broke" wrong answers or unparseable output? Free, from the rows on disk, and it decides whether full
    replacement destroys comprehension or just formatting.

- **Next Steps:** submit the held erasure-vector run
  ([`2026-09-12_erasure-vector-prereg.md`](2026-09-12_erasure-vector-prereg.md)) **as written** — its readout is
  `G_sum`, not accuracy, so the `ERASURE-FLOOR` bound does not void it, and the accuracy add-on contemplated in
  the plan is not triggered because `c3_band` did not move overall accuracy. Then H-A5 (0.5 GPU-h, gates
  `ROUTES-COMPOUND`), H-A4/H-S15 (free), and H-S13 before any further steering-geometry work.
