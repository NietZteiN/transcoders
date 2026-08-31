# 2026-08-13 — N11 deception: the effect is lexical echo; the echo-free residue is not established

**Goal / hypothesis (B3).** Do adversarially misleading identifiers induce a *false internal
belief* about what a program computes, or do reads merely re-read the misleading name? Three
conditions per snippet, differing only in identifiers: **C0** original · **C1**
`identifier_renaming` (neutral opaque names — the confabulation floor) · **C2**
`adversarial_rename` (names of a plausible wrong algorithm, chosen by us and therefore ground
truth by construction).

**Setup.** Qwen2.5-7B-Instruct L20, released NLA pair, seed 20260724. 100 HumanEval snippets ×
3 conditions = **300 captures, 6,171 reads, 7.02 h**, GPUs 0+1, tmux `nla-deception`, 0 errors.
21 reads/item at four loci sited by character span: `CODE_ID` (on renamed identifiers),
`CODE_FAR` (code tokens away from them), `COT`, `ANSWER` (the `Algorithm:` commitment line).
Task ends in a mechanically graded line, so no case packs and no "true algorithm" label are
needed. Scoring: `nla/src/deception_stats.py`.

**Quality.** `rt_cos` median **0.870** (band 0.70–0.96), mostly-CJK reads **0 / 6,171**.
The instrument was not the limiting factor.

**Results.**

Read-level hit rate (a read names the injected algorithm), by locus × condition:

| locus | C0 | C1 (floor) | C2 |
|---|---|---|---|
| `CODE_ID` | 0.9% | 0.8% | **30.9%** |
| `CODE_FAR` | 1.6% | 1.8% | 3.0% |
| `COT` | 1.4% | 2.2% | 3.6% |
| `ANSWER` | 0.3% | 0.4% | 3.0% |

C2 distance gradient (chars from the nearest misleading identifier):

| 0 (on it) | 1–50 | 50–150 | 150–400 | 400+ |
|---|---|---|---|---|
| **30.9%** | 3.7% | 2.5% | 5.4% | 0.0% |

Item-level, four variants (the first is the pre-declared primary):

| variant | C0 | C1 | C2 | Δ(C2−C1) | discordant | McNemar p |
|---|---|---|---|---|---|---|
| **primary — echo-free, k=2** | 3.0% | 2.0% | **9.0%** | **+7.0pt** | 8 vs 1 | **0.039** |
| echo-free, any read (k=1) | 11.0% | 17.0% | 18.0% | +1.0pt | 9 vs 8 | 1.00 |
| all loci, k=2 | 5.0% | 6.0% | 74.0% | +68.0pt | 68 vs 0 | <1e-15 |
| all loci, any read — echo-inflated | 14.0% | 19.0% | 90.0% | +71.0pt | 71 vs 0 | <1e-15 |

Foreign-read null, attached to the **primary** statistic:

| | own | null mean | null 95% CI | verdict |
|---|---|---|---|---|
| C0 | 0.030 | 0.052 | [0.02, 0.09] | within |
| C1 | 0.020 | 0.040 | [0.01, 0.07] | within |
| **C2** | **0.090** | 0.057 | **[0.02, 0.10]** | **within — does not clear** |

Behavioural coupling: `stated_hit` (the model *says* the injected algorithm) is **0/98 (C0),
0/82 (C1), 3/77 (C2)**. Among those 3, the recurrent read flag fired **3/3**; among C2 items
where it was not stated, 8.1%.

**Verdict — B3 NOT ESTABLISHED.**

1. **The headline is echo, and the design caught it.** The raw item-level rate is 90% in C2
   against a 19% floor — and 30.9% of `CODE_ID` reads hit against 0.8% at the same locus in C1.
   Off the identifier the rate collapses to ~3% and stays there regardless of distance. That is
   the signature of a local lexical echo, not a belief propagating through the representation.
   Reporting the 90% would have been reporting that the NLA can read a token.
2. **The echo-free residue is real-looking but does not survive its own null.** The primary
   contrast is +7.0 points (9.0% vs 2.0%) with McNemar p = 0.039 on 8-vs-1 discordant pairs —
   nominally significant, uncorrected, and one of four variants examined. But the foreign-read
   null for that same statistic spans [0.02, **0.10**] and the observed C2 rate is 0.090. Shuffling
   read *content* between items, while holding each item's read positions fixed, reproduces the
   effect. **The paired test and the null disagree, and the null is the one attached to the
   deciding statistic.** Eight items out of a hundred is what this rests on.
3. **The one suggestive cell is behavioural coupling, at n=3.** Every C2 item where the model
   *stated* the wrong algorithm also carried a recurrent internal read (3/3) against 8.1% where
   it did not. That is the predicted direction and exactly the dissociation structure the
   programme is after — but three items is an anecdote, and the model is almost never
   behaviourally deceived (3/77), which is itself consistent with the papers' finding that
   adversarial renaming does not reliably hurt accuracy.
4. **The `ANSWER` locus behaved as designed but is too thin.** 3.0% in C2 against 0.3%/0.4%
   floors — the largest *ratio* in the table, at the locus furthest from the identifiers. But
   it is 7 reads out of 231, and the denominator is itself compromised (below).

**Confound found, not designed around: the parse rate declines monotonically.** C0 **98%** →
C1 **82%** → C2 **77%** of captures produce the `Algorithm:` line. So the harder the identifiers,
the less often the model commits at all — a real behavioural effect of renaming, but it also
shrinks the `ANSWER` denominator precisely in the condition of interest, and it means every
statistic conditioned on a parsed answer (coupling included) is computed on a
condition-dependent subsample. This was not pre-declared and should be treated as a limitation,
not folded into a claim.

**Two bugs in my own scoring, both found before the numbers were reported.**
(a) The foreign-read null stored each read's hit flag relative to *its own* item's injected
algorithm and reused it after reassignment — so it asked "does read X name X's algorithm",
which is the own rate resampled, not a null. It came out *above* the own rate because taking a
max over resampled reads saturates. Fixed to store the set of algorithms each read names and
test membership of the **target** item's algorithm.
(b) Having fixed it, the null was still attached to the k=1 statistic while k=2 carried the
verdict — **the exact HT14 error**, where a foreign-answer null guarded a descriptive
trajectory while the log-rank ran unguarded. Now computed on the primary, preserving each
item's read positions so the recurrence rule's adjacency structure is unchanged.

**Limitations.** 100 snippets, HumanEval only, one model, one layer, greedy, single seed. The
vocabulary bounds recall from below: a read expressing the injected algorithm in words outside
the closed set scores as a miss. `k=2` was chosen from the affordability measurement (median 2
gradeable reads/item) and fixed before scoring, but it is weaker than the NLA paper's own
recurrence heuristic.

**Next steps.** The honest read is that adversarial renaming moves the *reading of the name*
and not, detectably, the *belief about the program*. Before any larger run, the one cell worth
chasing is behavioural coupling — it needs items where the model is actually deceived, and at
3/77 this corpus supplies almost none. Selecting or constructing snippets where renaming does
flip the stated answer would give that test a denominator; without it, B3 has no behavioural
variance to couple to and the internal measure stands alone against a null it does not clear.
