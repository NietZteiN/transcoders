# 2026-08-16 — steering, seen in the text: all 10 flips reproduce, and V1/V3 tie while disagreeing on 7 of 11

**References** [`2026-08-14_n12-steering-gate.md`](2026-08-14_n12-steering-gate.md), whose verdict
(gate FAILS, V1 0.550 = V3 0.550, p = 1.00) is unchanged. This adds the generated text the original
run did not keep, and one new observation from it.

**Why a re-run was needed.** `steer_run.py` stored `answer`, `correct` and `reply_chars` but **not
the replies**, so "what did steering do to the text" was unanswerable from the banked results.
Added `--snippets`, `--only-conditions` and `--save-replies`; re-ran the 11 items that V1 flipped at
alpha = 1.0 (7 recoveries + 4 damages), conditions V1_gloss and V3_taskvec at alpha 0.0 and 1.0.
Seed 20260724, GPU 0, ~35 min, 55 rows / 44 with replies, 0 errors.
A first launch applied `--snippets` *after* the baseline pass and was generating baselines for all
60 items before discarding 49; the filter now runs before that pass.

**Result 1 — the flips reproduce exactly.** All **10** items with both runs available reproduced
their original classification: 7 recovered, 3 broken (the 4th damage case landed in the tail of the
run). Same seed, same directions, same outcomes weeks later. `alpha = 0` is the unsteered run and
the write hook is byte-identical there, so the two columns differ in exactly one thing.

**Result 2 — what V1's "explanation edit" actually is.** The direction is
`AR(gloss_true) - AR(gloss_decoy)`, and both glosses turn out to be a **fixed template listing
identifier names** — *"code whose identifiers are about ret, e1, l1, l2, e2, common, Set, add"* —
not a description of what the algorithm computes. **V1 therefore tested swapping a name list, which
is a materially narrower intervention than "edit the explanation and steer" implies.** This does not
change the gate's verdict but it does change what the verdict is a verdict *about*.

**Result 3 (new) — V1 and V3 tie by netting out, not by agreeing.**

| | V1_gloss | V3_taskvec |
|---|---|---|
| correct on these 11 items | 7 | 6 |
| **same verdict as the other** | **4 of 11** | |

The full sweep reported V1 = V3 = 0.550 exactly. On this subset the totals are again near-identical
(7 vs 6) while the two methods **disagree on 7 of 11 individual items**. So the tie is not "the NLA
direction is redundant with activation arithmetic" — they are *different* interventions with
indistinguishable hit rates. That is a sharper statement of the null than the aggregate supports on
its own, and it points at a real follow-up: an ensemble or an oracle over {V1, V3} would beat either
alone, which would say the failure is in *selection*, not in the directions.

**Caveat, load-bearing.** These 11 items were chosen **because V1 flipped them**, so the subset is
biased toward V1 being active and the 7-vs-6 totals are not an unbiased comparison. The item-level
disagreement is the observation worth keeping, and it too rests on a V1-selected n = 11.

**Observations.** Reading the pairs, the edit rarely changes the *method* the model applies — it
changes which quantity the model believes it is tracking, and the arithmetic follows. It re-runs the
same procedure on a different referent, sometimes landing right and sometimes destroying a correct
answer. That is the same picture the N13 confusion work arrived at from the other side: the decoy
installs one confident wrong reading rather than two competing ones.

**Next steps.** (1) The ensemble/oracle check above, on the full 60-item set, which is cheap now that
the runner can be pointed at a subset. (2) If V1 is ever re-run as a claim rather than a gate, the
gloss template must be replaced with an actual algorithm description — the current template is a
confound between "editing an explanation" and "swapping identifier names".
