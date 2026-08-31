# 2026-08-16 — one table for every reading: 14,619 rows across three passes, and the first cross-corpus contrast

**Goal.** Three capture passes had produced readings in three different row shapes, which kept them
un-comparable. Unify them so cross-corpus questions become askable at all.

| pass | cases | readings | shape |
|---|---|---|---|
| overnight | 380 | 5,090 | token classes (`cls`), `anchor` |
| dense | 182 sides | 4,653 | `kind`, `tok`, `u_rel` |
| malware | 366 arms | 5,108 | `locus`, `mm`, native `act_norm` |

**Setup.** `nla/src/unify_corpora.py` → `data/nla/unified/{readings,cases}.jsonl` + manifest.
One normalized `read_class` vocabulary across the three naming schemes; `corpus`, `pass`,
`case_id`, `position`, `token`, `read`, `rt_cos`, `act_norm`, `u_rel`, `malice_margin`, plus the
label columns (`tier`, `correct`, `malicious`, `capability`). Seed 20260724.

**Two things fixed at the unification boundary.**
1. **No double counting.** The dense pass deliberately re-used **232** readings from the overnight
   pass. They carry `reused: true` and are dropped, so 4,653 dense rows contribute **4,421** new
   ones. 5,090 + 4,421 = **9,511**, which reconciles exactly with the results-browser count.
2. **Malware rows are redacted here, not downstream.** Derived data inherits quarantine status, so
   a raw unified table would itself be a quarantined file and useless for ordinary analysis.
   Redacting at this boundary means the table carries no live indicator. Verified: 0 residual.

**Result — the table.** **14,619 readings / 928 cases.** `act_norm` present on **10,198** of them
(native for malware, recovered for the overnight positions by N13 stage 2; the dense pass never
stored it).

**First cross-corpus contrast — reasoning readings, obfuscated vs malware.**

| | rt_cos [95% CI, case-clustered] | mean read length |
|---|---|---|
| obfuscated (n=6,401, 330 cases) | 0.8539 [0.8508, 0.8569] | 658 |
| malware (n=2,184, 182 cases) | **0.8928** [0.8901, 0.8952] | 719 |

Raw gap **+0.0389**. But `rt_cos` correlates **+0.309** with read length here (consistent with the
+0.277 measured on the banked corpus), and the malware readings are 61 chars longer — the exact
confound that dismantled HT12. Stratifying by pooled read-length decile and re-weighting:

**length-adjusted gap +0.0152, CI [+0.0073, +0.0232]** — **61% of the raw gap was length**, and the
remainder survives. The per-decile pattern is not uniform: the shortest decile *reverses*
(−0.0503, n=34 malware) while all nine others are positive.

**Two more, from columns that could not previously be compared.**
- **`act_norm` is higher on malware**: 113.32 [112.91, 113.72] vs 109.53 [109.22, 109.87],
  non-overlapping. Only askable because N13 stage 2 recovered `act_norm` for the obfuscation
  positions; before that it existed on one corpus only.
- **Framing changes what a reading says, not how faithful it is.** The security and neutral arms sit
  at rt_cos 0.8700 [0.8670, 0.8728] and 0.8730 [0.8699, 0.8759] — indistinguishable — while the
  capability-naming rate between the same two arms is 0.874 vs 0.705. A clean dissociation:
  the preamble moves the *content* of the readings without moving the instrument's fidelity.

**Interpretation.** All three are consistent with the describability account N13 landed on. Malware
is semantically vivid — "this steals credentials and posts them to a webhook" is an easy sentence.
Obfuscated benign code is semantically thin — dispatcher bookkeeping is a hard sentence. Faithfulness
tracks how much there is to say, and it says so again across a corpus boundary.

**Caveats.** The two corpora differ in prompt, task and length distribution, so this is an
observational contrast, not a controlled one; the length stratification removes the one confound
with a documented history in this programme, not all of them. The benign corpus retains one
confabulated URL (a HumanEval-X problem *about* URL parsing) — it is not malware-derived and is not
subject to the quarantine filter.

**Next steps.** The unified table makes the obvious sweep cheap: `rt_cos` and `act_norm` by
`read_class` × `corpus` with length as a covariate, over all 14,619 rows at once, instead of three
separate analyses that cannot be pooled.
