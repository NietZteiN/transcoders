### Target Date: 2026-09-03 (The rescue ceiling was the noise floor)

- **Hypotheses / what we're testing:** Re-analysis of banked rows, no new GPU. **H-D1:** the
  number of items that a steering rescue *could* fix — L0-correct and L1b-incorrect at baseline —
  is small enough that a perfect rescue lands inside the ±0.075 reproducibility floor. If so, every
  accuracy-level null filed today was **underpowered by construction**, independent of direction,
  depth or coverage. REFUTED if the flippable set is ≥ 12 (a +0.20 ceiling, detectable).
- **Setup:** `data/nla/p0/steerv2/gemma12b/run/baseline.jsonl` (60 items, L0 and L1b greedy
  replies per item, host `gemma12b`), joined against every steering row in
  `data/nla/p0/steerv2/gemma12b/run/`, `data/nla/p0/nla_steer/gemma12b/`,
  `data/nla/p0/coverage/gemma12b/` (1,920 rows). Per arm: greedy accuracy, items **rescued**
  (flippable → correct) and items **broken** (L1b-correct → incorrect). Python one-liner, no seed
  involved (deterministic re-read).
- **Results:**
  - Baseline cells: L0 ✓/L1b ✓ **38** · L0 ✓/L1b ✗ **6** · L0 ✗/L1b ✗ **16** · L0 ✗/L1b ✓ 0.
    12 of 60 L1b replies are **unparsed**. L0 acc 0.733, L1b acc 0.633.
  - **Flippable set = 6/60.** `humaneval-x-javascript/JavaScript/1`, `.../JavaScript/104`,
    `humaneval-x-python/Python/1`, `leetcodedataset/adjacent-increasing-subarrays-detection-{i,ii}`,
    `leetcodedataset/assign-elements-to-groups-with-constraints`.
  - Perfect-rescue ceiling = **+0.100**. Reproducibility floor = ±0.075. Rule threshold for a
    supported hypothesis all day = **+0.10**. The maximum achievable effect equals the threshold.
  - Rescued / broken per arm (all 60-item arms, `last_prompt` unless stated):
    `V1_gloss` **0/6** at every α (0.25, 1.0, 4.0) and at both coverages incl. α=8 · `V3_taskvec`
    1/6 (α 0.25/1.0/4.0), 0/6 at id_spans · `V4_oracle` 2/6@0.1, 0/6@1.0 · `V5_replace` 0/6 ·
    `A_antipodal` 2/6 (broke 2) · `P_prompt` 1–2/6 (broke 1) · `F_foreign` 1/6 · `R_random`
    0–1/6, broke 2→27 as α rises. The 22 items that are L0-wrong or L1b-unparsed can never
    contribute to a rescue and were 37% of every denominator.
- **What worked / hypothesis verdict:** **H-D1 SUPPORTED.** Six flippable items; +0.10 ceiling;
  floor ±0.075. Every "no arm recovered >2 of 60" statement today is literally true and
  **uninformative about rescue**: 2 of 6 is a 33% rescue rate, and the antipodal control and the
  prompt baseline both hit it, so 2/6 is churn, not signal — but the design could not have told
  a 4/6 rescue (+0.067) from noise either.
- **Observations:** This is a scope error, not a measurement error. The 60-item set was the smoke
  scale that became the main scale, chosen when the host was Qwen, whose L1b penalty was larger.
  On Gemma the trap breaks 6 items. The KV-bypass UNINFORMATIVE verdict, the N-0 replication and
  the C-0 verdict are all correct *as filed* under their frozen rules; what the rules did not
  anticipate is that the ceiling would sit on the threshold. V1's 0/6 in **seven** independent
  arms is the one per-item number that resists this excuse — the NLA direction never flipped a
  flippable item once — but with n=6 that is still only a bounded negative.
- **New questions / new hypotheses:** (1) Is the answer decided anywhere our writes reach? —
  needs a causal map, not more directions. (2) Is *any* vector at L32/last-token able to move a
  continuous readout? — needs an optimised-vector test with held-out items. (3) A per-item
  continuous metric (likelihood of the model's own clean trace under the steered corrupt prompt)
  would sidestep the 6-item denominator entirely. All three are pre-registered in
  [`2026-09-03_site-alive-prereg.md`](2026-09-03_site-alive-prereg.md).
- **Next Steps:** Add rescue/broken columns to `steer_stats.py` so no future arm reports accuracy
  without them. Add a **power gate** to the family rule: refuse to run a rescue experiment whose
  flippable ceiling is below 2× the floor. Grow the corpus from Dataset B (250) before any further
  accuracy-level claim on Gemma.
