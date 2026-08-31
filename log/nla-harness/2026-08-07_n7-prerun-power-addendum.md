### Target Date: 2026-08-07 (N7 pre-run addendum — HT13 power, arm declaration, and the HT14 re-spec)

**This entry is written BEFORE the judge runs.** It exists because HT12 taught the lesson the hard
way: the frozen rule in [`2026-08-06_n6-n8-preregistration.md`](2026-08-06_n6-n8-preregistration.md)
specified an *estimator* and a *test* without first checking that the corpus's group structure could
carry them. That cost a singular fit and a declared deviation. So the same checks are now run for
HT13/HT14 up front, and their outcome is recorded here rather than discovered mid-analysis.
**Nothing below relaxes a pre-registered threshold** — the confirm/refute rules for HT13 and HT14 are
unchanged. What is added is the analyzable-n, the power behind it, and a split into a primary and a
pre-declared sensitivity arm.

- **What was checked (CPU, seed 20260724, no GPU):**

  1. **HT13's analyzable n.** The `after_error` split needs a first-error location *and* reads on both
     sides of it inside the dense corpus. Of 41 localized cases with a `u_rel`: 30 are in the dense
     corpus, and **23 have ≥3 reads both before and after the error** (median 7 before / 15 after).
     Only **8** of those 23 rest on a *trusted* detector — D0 (slice set-diff, deterministic, no LLM)
     or D2-call (known-answer verified). The other 15 are **D2-state**, which the 2026-08-06
     validation showed agrees with the D3 judge only 39% of the time (r = +0.10).
  2. **Power at those counts.** Simulation at the real per-case read counts, with the read-level
     within-case ICC estimated from `rt_cos` as a stand-in (**ICC = 0.159**), testing the
     pre-registered `correct × after_error` interaction via GEE (cluster = case):

     | n cases | δ = 0.2 SD | 0.4 SD | 0.6 SD | 0.8 SD |
     |---|---|---|---|---|
     | **23 (primary)** | 0.38 | **0.90** | 0.99 | 1.00 |
     | 8 (trusted-only) | 0.24 | 0.54 | 0.85 | 0.98 |

- **Declarations made now, before any judge output exists:**
  - **HT13 primary arm = all 23** analyzable cases. It is adequately powered (0.90) for a **0.4-SD**
    post-error alignment drop, which is the size worth caring about; it is **not** powered for a
    0.2-SD effect (0.38), so a null must be reported as "no *moderate-or-larger* effect", never as
    "no effect".
  - **HT13 sensitivity arm = the 8 trusted-localization cases**, declared here as **underpowered
    (0.54 at 0.4 SD)** and reportable only as agree/disagree in direction with the primary. It is not
    a second bite at significance.
  - **Localization noise attenuates toward the null.** 15 of the 23 primary cases carry a detector
    whose boundary is unreliable, and a misplaced boundary dilutes the before/after contrast. So the
    *true* minimum detectable effect is worse than the table shows, and a **null on HT13 is
    ambiguous between "no post-error drop" and "localization too noisy to see one"** — that ambiguity
    is declared now so it cannot be spun later as a clean negative. A confirm, by contrast, is
    unaffected: attenuation cannot manufacture an interaction.
  - **The instrument gate comes first.** Per the plan, if the shuffled-pair AUC < 0.65, or the judge
    fails to beat the AR-space baseline, or κ(Llama, Phi-3.5) < 0.40, then N7 is reported as an
    **instrument null**, HT13 is not adjudicated at all, and the artifact's alignment sort **stays
    disabled**. The judge must be shown to measure something before its measurements mean anything.

- **HT14 re-specification (the trap flagged in Observation 2 of [`2026-08-07_n6-ht12-refuted.md`](2026-08-07_n6-ht12-refuted.md)):**
  answer onset is **one number per case**, so `correct` is again constant within case and a `(1|case)`
  random intercept would absorb it exactly as it did in HT12. The frozen HT14 rule is fortunately
  already immune: its primary is **Kaplan–Meier + tier-stratified log-rank on the censored onset**,
  which is a **case-level** survival test with no per-case random effect. **No change of rule is
  needed — only a change of implementation:** N8 must run KM/log-rank at case level as written, and
  must not "upgrade" to a read-level mixed model. Recorded so the temptation is closed off in advance.

- **Unchanged:** BH-FDR across {HT12, HT13, HT14} applied once all three report. HT12 is already
  refuted by a trigger rather than a p-value, so it does not consume family error rate.

- **Next:** build `sanitize_windows.py` (blinding boundary) → `judge_align.py` (Llama-3.1-8B,
  constrained verdicts, shuffled + distant nulls, AR baseline) → κ cross-check vs Phi-3.5 → gate →
  only then adjudicate HT13.
