### Target Date: 2026-09-06 (H-W20 — audit of every banked W arm, and a guard that refuses to average an arm that wrote nothing)

Follows [`2026-09-06_hw17-results-and-two-bugs.md`](2026-09-06_hw17-results-and-two-bugs.md).
**No GPU.** Commit `a90ccfc`.

- **Hypotheses / what we're testing:** not a hypothesis — a harness fix and an audit of whether
  bugs #8 and #9 touched anything already reported.

- **The audit. Only one run is affected; every other W number stands.**

  | run | items | arms | zero-position items | arms scoring exactly 0.0 |
  |---|---|---|---|---|
  | writeback (377399) | 60 | 4 | none | none |
  | fidelity (377835) | 49 | 3 | none | none |
  | nulls (378019) | 49 | 4 | none | none |
  | dose (379075) | 49 | 8 | none | none |
  | tiers (379819) | 49 | 11 | none | none |
  | **meaning (379908)** | 49 | 11 | **`n_positions_sub` = 0 on 29/49** | **`T_L2_all` 49, `T_L0_sub` 29, `T_L1_sub` 29** |

  So `W-GENERIC-PERTURBATION`, `W7-CHANNEL-STRONG-AND-FAITHFUL`, `W9-ITEM-SPECIFIC`,
  `W13-ACCUMULATIVE-TIER-GENERIC` and job 379819's structure/ladder results are **untouched**.

- **The guard (`nla/src/arm_guard.py`), and why one detector was not enough.**
  - **Detector 1 — `n_pos_<arm> == 0`.** `record_positions()` stamps a per-arm written-position
    count into every row. Exact, and available only for rows written from now on.
  - **Detector 2 — `dG` exactly `0.0`.** Necessary, and I found that out by testing rather than
    assuming: **re-scoring the 379908 rows through detector 1 alone reproduced the bad verdict**,
    because those rows carry no position field and the fallback scored every row. A teacher-forced
    log-probability difference does not land on precisely 0.0 when an intervention was applied; it
    lands there when none was.
  - **`allow_exact_zero` is opt-in per arm.** SELF writes each position's own activation back and
    *should* score 0 — it did so exactly on 21 of 49 items. Making the exception per-arm keeps the
    detector strict everywhere else rather than weakening it globally for one honest case.
  - **`paired()` also refuses mismatched position counts** between two arms on the same item. Equal
    counts are what make `A − B` a contrast about *content* rather than about how hard each arm
    pushed — an assumption every null in this family has silently relied on.
  - An arm that never wrote **raises** rather than returning a flag: it is an absent experiment, not
    a null result, and it must not reach a verdict table.

  **Verified:** 379908 now scores **`W16-ARM-ABSENT`**, naming `T_L2_all`; 379819 re-scores
  **unchanged** (+0.66 structure, +8.70 meaning). 10 regression tests, both bugs encoded directly.

- **What this does NOT fix, stated so it is not assumed.** Bug **#7** — the `tier_anchor` fallback
  dilution — is invisible to this guard. Those spans wrote **real** positions with vectors that
  merely happened to be the same identifier, so `dG` was not zero and no count was wrong. It was a
  **semantic** confound and needed the `--strict-anchor` fix. Three bugs, two mechanisms: a
  position-accounting guard and a semantic-validity check are different things, and only the first
  is now automated.

- **Observations.**
  - The audit was possible only because every run persists per-item rows. The **H-W4 correction**
    two days ago argued that this project discards vectors and keeps scalars, and that the cost is
    always a metric question that should take minutes taking GPU-hours. Here the argument paid
    directly: correcting H-W17's headline from +8.91 to **+21.82** needed **no GPU at all**.
  - **Nine defects in this family, and the split is stable:** the *measurements* have been sound
    every time; what fails is the layer between a measurement and a verdict. Six were caught by a
    control, a smoke test or an assertion; three by noticing a number that looked impossible
    (identical arm values, a CI of exactly [0,0], reads printing 22/22 tag-less).

- **New questions / new hypotheses.**
  - **H-W21:** the other four scorers (`nla_writeback`, `nla_fidelity`, `nla_nulls`, `nla_dose`)
    still average raw columns. They are **clean today** by the audit above, but they carry the same
    latent defect and should adopt `arm_guard` when next touched. Recorded rather than done, because
    editing five scorers whose results are banked is a bigger change than it looks.
  - **H-W19** (cross-language L1 comparability) remains the highest-value open item and now blocks
    any cross-language claim about the L1 tier anywhere in the study.

- **Next Steps:** H-W19 — an hour of CPU on the stimulus builder to determine whether the missing
  JavaScript `rename_map` is a pipeline asymmetry or a real property.
