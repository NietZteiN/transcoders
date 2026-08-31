# 2026-08-15 — N13 addendum: the answer normalizer inflated entropy, and it was carrying the one marginal positive

**References** [`2026-08-15_n13-torn-null.md`](2026-08-15_n13-torn-null.md). Append-only correction
per the log protocol: that entry's `mean_instability` result is **superseded** by this one. Its
nulls, gates and the describability finding are unchanged.

**What prompted it.** Pulling concrete examples for a write-up, two of the three "committed wrong"
items turned out to be grading artifacts rather than model behaviour:

| item | truth | sampled answers | problem |
|---|---|---|---|
| `cruxeval-x-python/102:L1` | `[]` | `{'**[]': 4, '[]': 4}` | markdown emphasis split ONE answer into two |
| `cruxeval-x-python/1:L1b` | `{1:None,…,4:None}` | `{1:none,3:none,4:none,2:none}` vs `{1:none,2:none,3:none,4:none}` | dict **key order**; semantically identical |

Stage 1 counted distinct answers with the harness normalizer
(`re.sub(r"[\s'\"`]", "", s).lower()`). That is exactly right for reproducing the banked greedy
labels — and it is what made the 8/8 greedy tripwire meaningful — but it was never designed to
decide whether two answers are *the same answer*. **36.4% of items (120/330) contain a `*` in at
least one sampled answer.**

This matters more than a tidiness complaint: entropy is the **outcome variable** of the entire
experiment. Spurious distinct answers inflate it, and noise in an outcome **attenuates**
associations — so a null measured against a noisy label can be an artifact of the label rather
than a fact about the model. Re-analysis was mandatory before reporting, not optional.

**Setup.** `nla/src/reharden_entropy.py` recomputes per-item entropy/modal/correctness from the
banked `answer_counts` under a hardened normalizer: strip markdown emphasis, strip leading
"Output:/Answer:/Result:", then the harness rule; and canonicalize mapping literals `{k:v,…}` by
sorting their comma-separated items (order-independent for dicts — **not** applied to lists, which
are ordered). Banked Stage-1 output left untouched as the record; hardened copy written alongside.
Entropy changed on **96/330** items, correctness label flipped on **15**. Stage 4 re-run unchanged
otherwise, seed 20260724.

**Results — the one marginal positive does not survive.**

| statistic | harness norm | **hardened norm** |
|---|---|---|
| mean entropy | 0.4328 | 0.3788 |
| frac perfectly consistent | 0.188 | 0.270 |
| horse race p_perm **`mean_instability`** | **.029** | **.234** |
| horse race p_perm `mean_rt_cos` | .402 | .606 |
| horse race p_perm `mean_act_norm` | .983 | .409 |
| baseline R² (`log_reply` alone) | 0.3511 | 0.3443 |
| 2×2 AUC rt_cos / act_norm / instability | .491 / .442 / .408 | .470 / .432 / .424 |
| Gate 1b, L1b − L0 | −0.0005 [−0.082, +0.082] | −0.0113 [−0.098, +0.071] |
| **entropy AUC wrong vs right** | 0.843 | **0.869** |

**Verdict — the conclusion gets simpler and stronger, and one claim is withdrawn.**

1. **Withdrawn:** read instability beats the length baseline (ΔR² .017, p = .029). Under a cleaner
   outcome label it is **p = .234**. That edge was partly a formatting artifact. **All three
   internal signals are now null**, with no marginal case to caveat.
2. **Unchanged:** every gate and every 2×2 (all AUCs remain at chance); Gate 1b still shows the
   L1b trap does not induce torn-ness; length still dominates (R² ≈ 0.34).
3. **Strengthened:** answer entropy predicts correctness at **AUC 0.869** (was 0.843). A signal
   getting *sharper* when label noise is removed is what a real effect does — the opposite of what
   the instability result did.

**Observations.** The direction of every change is diagnostic. Removing label noise moved the one
borderline internal result toward null and the one strong behavioural result away from it. Had the
artifact been cleaned only after seeing which way it pushed a favoured hypothesis this would be
worthless; it was found by reading raw examples for a write-up, before any of these numbers
existed.

Worth noting for reuse: the harness normalizer is **still correct where it is used** — reproducing
banked greedy labels, where 8/8 agreement was verified. The bug is one of scope, importing a
grading rule into a *distinctness* question it was never built for. Any future work sampling K
answers should use `reharden_entropy.harden`, not `grade`'s normalization.

**Next steps.** Unchanged from the parent entry. The hardened file
(`data/nla/n13/answer_entropy_hardened.jsonl`) is the one to use for any further entropy analysis;
`answer_entropy.jsonl` is retained as the raw record and is the correct input for anything
comparing against banked greedy labels.
