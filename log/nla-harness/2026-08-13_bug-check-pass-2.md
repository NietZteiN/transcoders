# 2026-08-13 (pass 2) — Second debug pass: crash-on-empty in the unattended scorer

*Follows `2026-08-13_autonomy-and-bug-check.md`. That pass audited by reading; this one audited
by **running the unattended stages against adversarial inputs**, which is what found the rest.*

**Method.** Rather than re-read the code, execute the two stages that will run without
supervision — `select_coupling_set.py` on the partial screen, and `deception_stats.py` on a
deliberately degenerate capture (C2 rows only, which is what a truncated or filtered run looks
like).

**Found.**

1. **The scorer crashed on any empty cell — the worst possible failure for an unattended chain.**
   `statistics.mean([])` raises, and there were four such sites (`parse_rate`, item-level
   `rate`, the foreign null's per-permutation mean, and `quality.rt_cos_median`). A capture that
   produced only one condition — or any filtered subset — would land after hours of GPU time and
   then die in stage 4, leaving the data unanalysed while the autopilot reported a stage failure.
   Now every aggregate goes through a `_mean` returning `None` on empty, the report carries an
   explicit `warnings` list, and the degenerate input scores cleanly:
   `warnings: ["no captures for condition C0", "no captures for condition C1", ...]`,
   `parse_rate: {C0: null, C1: null, C2: 0.7}`, paired `delta: null`.
   **A missing cell should be visible, not fatal.**

2. **`unmatched_hits` was reported per algorithm, not per hit.** The check asked whether *any*
   selected control shared the hit's injected algorithm. With two hits on one algorithm and only
   one control available, both were reported as matched — so the report would have claimed full
   matching while a hit went uncontrolled. Now tracked per `(dataset, snippet)`.

**Regression checks.** The real N11 numbers are byte-identical after both changes — primary
C0 3.0% / C1 2.0% / C2 9.0%, McNemar p = 0.039, C2 foreign-null CI [0.02, 0.10], `warnings: []`.
Full suite **64 passing**.

**A selection-composition finding worth carrying into the write-up.** At 62% of the screen,
13 deceived items: **`hashing` accounts for 7 of 13 (54%)**, with fibonacci 2, palindrome 2,
matrix_multiply 1, prime_check 1. So the coupling result will be dominated by one injected
algorithm, and the obvious alternative reading is not that the model is deceived by hash-like
names but that hash vocabulary ("digest", "checksum", "computes a hash") is simply easy to emit
about many programs. The matched-control design already absorbs this — each hashing hit is paired
with a hashing **non**-hit, so the algorithm's base rate is held fixed within the contrast — but
the imbalance must be reported, and a coupling effect carried entirely by `hashing` should be
stated as such rather than generalized.

**Next steps.** Unchanged: screen → select → capture → score, unattended on GPU 0. Expect ~20
deceived items at full screen, above the 6-item floor, so the capture will fire (~40 snippets ×
3 conditions ≈ 120 captures).
