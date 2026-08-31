# 2026-08-15 — N13 follow-on: the mid-trace faithfulness dip is not length and not magnitude, but the two-measure test disagrees

**References** [`2026-08-15_n13-torn-null.md`](2026-08-15_n13-torn-null.md) and the original
positional finding in [`2026-08-06_faithfulness-by-token-type.md`](2026-08-06_faithfulness-by-token-type.md).
Exploratory; prompted by the question "does N13 explain the start-high / dip / recover pattern?"

**Question.** The 2026-08-06 analysis found `rt_cos` across the reasoning trace goes
0.941 → 0.859 → 0.843 → 0.827 → 0.871: a **dip, not a slide**. Its stated interpretation was
describability (openings restate the problem, mid-trace holds bookkeeping, the end converges on a
statable answer) but it had one measurement and no controls. N13 banked two things that can test it:
`act_norm` at all 5,090 positions with `u_rel` (Stage 2), and read instability at 400 positions
(Stage 3).

**Setup.** Restricted to **`cls == 'cot'`** reads only (n = 1,980 over 330 cases) — an earlier pass
over all in-reply reads produced a spurious U-shape in `act_norm` that was pure **composition**, the
80–100% bin being inflated by `answer`-class reads. Case-clustered bootstrap, 2,000 resamples,
seed 20260724. Read length = character length of the banked verbalization.

**Results.** Within `cot` only:

| position | `rt_cos` [95% CI] | `act_norm` [95% CI] | read len |
|---|---|---|---|
| 0–20% | 0.9411 [0.9399, 0.9422] | 115.37 [114.90, 115.87] | 682.9 |
| 20–40% | 0.8591 [0.8537, 0.8644] | 107.88 [106.47, 109.24] | 654.6 |
| 40–60% | 0.8428 [0.8375, 0.8477] | 106.78 [105.72, 107.78] | 648.4 |
| 60–80% | 0.8332 [0.8282, 0.8387] | 105.41 [104.40, 106.38] | 654.3 |
| 80–100% | **0.8710** [0.8672, 0.8745] | **102.51** [101.90, 103.10] | 633.1 |

Read instability on the same class (n = 8 / 34 / 8 positions):

| | instability [95% CI] | lexical instability | `rt_cos` |
|---|---|---|---|
| early 0–20% | 0.0091 [0.0076, 0.0105] | 0.6030 | 0.9383 |
| middle 20–80% | 0.0341 [0.0286, 0.0401] | 0.6123 | 0.8325 |
| late 80–100% | 0.0204 [0.0166, 0.0254] | 0.6006 | 0.8594 |

Correlations among in-reply reads: `rt_cos`×read-length **+0.173**, `rt_cos`×`act_norm` **+0.542**.

**Verdict — the dip is real and is neither a length nor a magnitude effect, but the corroboration is
weaker than it first appears.**

1. **Not read length.** This is the notable one: read length is the variable that explained away
   HT12, and it does **not** work here. It declines monotonically (683 → 633) while `rt_cos` is
   U-shaped, and correlates only +0.173 with it in this subset.
2. **Not state magnitude.** `act_norm` declines monotonically and never recovers. The final bin has
   the **lowest** norm of all five and the **second-highest** faithfulness — describability and
   magnitude dissociate. Consistent with a compact, low-magnitude but semantically crisp
   end-of-trace state.
3. **The arc reproduces on an independent measure** — 5-sample read instability shows dip and
   recovery with non-overlapping CIs at both transitions.
4. **But the judge-free measure is FLAT** (0.603 / 0.612 / 0.601). `rt_cos` and AR-space instability
   both route through the same AR reconstructor and correlate at −0.96 overall, so their agreement
   here is close to guaranteed and is weak evidence. **We cannot separate "mid-trace states are less
   describable" from "the AR reconstructor handles mid-trace states worse"** — and the lexical null
   is consistent with the latter.

**Observations.** The parent entry's finding (instability ≈ describability, r = −0.96) cuts both
ways: it makes read instability useless as an independent check on anything `rt_cos` already says.
Any future corroboration of an `rt_cos` result must come from a measure that does not use the AR.

**Caveat.** The `rt_cos`/`act_norm` rows rest on 1,980 reads and are solid; the instability arc rests
on 8/34/8 positions and is suggestive only. The earlier composition artifact is a reminder to hold
`cls` fixed before reading any positional profile.

**Next steps.** (1) The positional profile at other layers — if the dip is a property of the model
it should move or persist with depth in an interpretable way; if it is a reconstructor artifact it
should not track layer content. (2) A describability proxy that bypasses the AR entirely (e.g.
next-token entropy of the subject model at the read position), which would break the shared-machinery
confound in a way neither measure here can.
