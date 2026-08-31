# 2026-08-12 — Grader validated against 5,090 real reads: one confound killed, one rule unaffordable

*Follows `2026-08-12_belief-steering-pipeline.md`, which built `belief_grade.py` against
synthetic strings only. This entry is the validation that entry named as the next free step.*

**Goal.** `belief_grade.py` had never seen an NLA read. Before any deception number is quoted,
measure on the banked corpus: does the vocabulary fire often enough to grade, is it degenerate,
and does any label track the *condition* rather than the content? CPU only; no GPU used (the
N10 malware capture holds two).

**Setup.** Env `nla-mi`, `LD_LIBRARY_PATH=$ENV/lib`, seed 20260724.
Corpus `data/nla/overnight/2026-08-04/captures.jsonl` — 380 items, **5,090 reads**, 5 tiers,
Python + JavaScript, with read-site class (`orig` / `adversarial` / `cot` / `answer` / …).

**Results.**

| statistic | value | reading |
|---|---|---|
| fire rate (any algorithm claim) | **0.205** | ~1 read in 5 is gradeable |
| distinct labels | 19 | broad |
| max label share | **0.278** | **not degenerate** — N7's failed rater sat at 0.91 |
| gradeable reads/item | median **2**, mean 2.74 | 14 reads/item design |
| items with ≥1 / ≥2 / ≥3 gradeable | 82.1% / 63.4% / **43.4%** | — |
| items with zero gradeable reads | **17.9%** | — |

Fire rate by read site — the split the design rests on:

| site | fire rate |
|---|---|
| `fn_orig` / `orig` (code tokens, clean) | 46.6% / **45.9%** |
| `target` | 32.9% |
| `cot` | **19.6%** |
| `adversarial` (code tokens, renamed) | **18.3%** |
| `answer` | **12.7%** |
| `l1_neutral` | 14.0% |

**Observations.**

1. **`encoding` was an obfuscation-surface detector, not an algorithm label — and it was the
   only one.** It was the single label whose rate *rose* with obfuscation tier (L0 1.68% →
   L1b 3.04% → L3 3.69%) while every genuine content label fell (`generic_sort` 8.86% → 2.89%).
   The cause was the bare pattern `\bencod\w*\b` matching reads that describe the *obfuscation*:
   "encoded variable names", "a cipher or log entry", "ASCII encoding context". It was 10.7% of
   all fires and would have loaded directly onto the C1/C2 contrast it was meant to measure —
   the same shape as the obfuscation confound that gates the malware corpus. Narrowed to require
   an object (`decodes the string from base64`) rather than a bare verb; it now contributes ~0
   at every tier. **Direction, not magnitude, is what exposed it**: a content label cannot fire
   more often as the code gets harder to read.
2. **The pre-registered `k ≥ 3` recurrence rule is unaffordable and must change.** Only 43.4% of
   items carry three gradeable reads and 17.9% carry none, so the statistic would be undefined
   or automatically negative for most of the corpus — the same failure shape as HT14, where 92%
   of cases never reached onset and the median was unmeasurable. Options, in order of
   preference: (a) **k = 2**, which 63.4% of items support; (b) raise reads/item from 14 to ~21
   for the deception run, which moves the median to ~3 at ~50% more capture cost; (c) keep k = 3
   and pre-register the reduced denominator honestly. Recommend (a) + (b) together, decided
   *before* the run rather than after seeing rates.
3. **Code-locus reads are 2.4× more gradeable than CoT reads** (45.9% vs 19.6%) and 3.6× more
   than answer-locus reads (12.7%). This supports the CODE-vs-VERDICT locus split, and warns
   that the verdict locus — where a concealment claim must live — is exactly where the grader is
   most data-starved. Consistent with N9's finding that reasoning prose reads are far noisier
   than code-token reads (46% vs 12% wrong-language).
4. **Adversarial renaming halves the fire rate at the very locus B3 depends on** (18.3% vs
   45.9% clean). Two readings, and they are not separable from this run alone: renaming may
   destroy the semantic content the read would express, or the read may still carry a belief the
   vocabulary cannot see. Either way B3's effective *n* is roughly half the naive estimate.
5. **Two bugs in my own diagnostic, both caught by tests I wrote against it.** The
   `min_base_rate` guard was applied to the *baseline* cell, which exempted the strongest
   possible signal — a label firing 0% on clean code and often under obfuscation has ratio ∞ and
   baseline 0, and sailed through. Moving the floor to the elevated cell fixed that but then
   flagged five rare labels, e.g. `hashing` at 4 reads in 1,082 (0.4%, ratio ∞). A rate floor
   alone cannot see that four reads move nothing, so the gate now requires **both** a minimum
   rate (1%) and a minimum count (20). With both floors: **zero suspect labels remain**.
6. The suspect threshold is 1.5, not 1.0. A bare `>1` test flagged `generic_sort` at 1.01 and
   `fibonacci` at 1.07, all driven by **L2**, where control-flow flattening genuinely adds
   dispatch loops and accumulator structure — the banked L2 dispatcher reads say "state machine",
   "while count < threshold". That is real content; a hair-trigger threshold condemns good labels.

**New tooling** (`belief_grade.py`): `surface_tracking_report()` — per-label fire rate by
condition with a ratio against a clean baseline, flagging any label that fires *more* under
obfuscation; and `gradeable_reads_per_item()` — what k the design can actually afford. Both are
gates to run **before** freezing a vocabulary. 63 tests pass, including regressions built from
the exact real read strings that produced the `encoding` false positives.

**Limitations.** Fire rate is measured against this vocabulary, so it bounds recall from below
only — `ALGO_NONE` still means "nothing listed fired", never "the read is contentless". The
vocabulary's true recall against a hand-labelled sample is still unmeasured, and the C1-vs-C2
comparison has not been run at all (no C2 corpus exists yet for Java; the transform is built and
verified but the corpus is not).

**Next steps.** Decide k and reads/item and freeze both before the deception run. Measure
vocabulary recall on ~100 hand-labelled reads to convert the 0.205 fire rate into a recall
estimate. Then B2, whose gate is unchanged: belief signal must clear the C1 confabulation floor
before B3 and B4 are worth running.
