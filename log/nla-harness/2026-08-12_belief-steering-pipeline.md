# 2026-08-12 — Belief-steering pipeline built and unit-tested (CPU only, no GPU)

**Goal.** Build the code pipeline for the belief-readout / belief-steering programme
(plan v0.3, `~/.claude/plans/`) while the GPUs are occupied by the N10 malware capture. No
experiment is run here; the deliverable is working, tested machinery and the bugs found in it.

**Setup.** Env `nla-mi` (torch 2.11, CPU), seed 20260724, commit `b54b7ee` + uncommitted.
`LD_LIBRARY_PATH=$ENV/lib` is required — without it `scipy` fails to import against the system
`libstdc++` (`CXXABI_1.3.15 not found`). The driver scripts already set this; anyone running
pytest by hand must too.

```
ENV=/data/jvl210002/conda_envs/nla-mi
LD_LIBRARY_PATH="$ENV/lib" CUDA_VISIBLE_DEVICES="" $ENV/bin/python -m pytest nla/tests -q
```

**Results.** 45 tests pass across three new modules.

| module | what it is | tests |
|---|---|---|
| `nla/src/steer.py` | the activation write hook — the one mechanism the NLA release does not ship | 10 |
| `nla/src/steer_vectors.py` | the V1–V5 direction ladder + control battery | 15 |
| `nla/src/belief_grade.py` | judge-free grading of what a read claims a program computes | 20 |

`steer.py` implements the paper's protocol, `h <- h + alpha*||h||*delta/||delta||` at chosen
absolute token positions at L20. Tests assert the properties that matter rather than proxies:
**alpha=0 reproduces the unsteered generation byte-identically**; the edit lands at the target
position with the exact expected value and every other position is bit-unchanged; alpha scales
the edit linearly; a reply-side position is reachable only if the cursor survives the
prefill/decode boundary; `reset()` rewinds. Tested on Qwen2.5-0.5B-Instruct, the same Qwen2
decoder architecture as the 7B subject, so the code path under test is identical.

`steer_vectors.py` arranges the directions as a **ladder with a pre-registered bar**, not a
menu: V1 `AR(edited) − AR(original)`, V2 the same from a single-word edit, V3 a contrastive
task vector `mean(h_clean − h_obf)` that uses **no NLA at all**, V4 the item's own oracle
difference, plus random / foreign-item / shuffled-text / antipodal controls. The leakage guard
is in the type signature — `TaskVectorBank.direction_for` requires `exclude` and raises without
it — because fitting a direction on the item it is applied to would inflate every steering
number in the paper and is invisible afterwards.

`belief_grade.py` replaces the graded rubric with a closed-set regex over an algorithm
vocabulary, plus the recurrence (k≥3 non-adjacent), distance, and behavioural-coupling
defences, and a `vocabulary_coverage` diagnostic that surfaces a degenerate labeller before a
run rather than after.

**Observations.**

1. **Three real bugs in the grader vocabulary, all caught by tests, all of the same kind.**
   `prime` fired on "primer"; the bubble-sort paraphrase was written as the literal "swapping
   adjacent" and so missed the far more natural "swaps adjacent". Every pattern is now anchored
   with `\b` on both ends and verb forms go through `\w*`. Writing the tests from the *shape of
   real reads* — hedged, descriptive, full of confabulated specifics — is what surfaced these;
   a test written from clean sentences naming an algorithm would have passed throughout.
2. **A fourth bug the tests could not have caught, found only by integrating.** Reads quote
   source identifiers verbatim (the N10 malware reads name `browser_cookie3` and
   `custom_function` straight out of the code), and no word-boundary pattern can see inside
   camelCase: `\bprime\b` does not match `isPrime`, because `s` and `P` are both word
   characters. The grader now scores both the raw text and an identifier-split form. This only
   appeared when the C2 renamer was wired to the grader end to end — the unit tests on either
   side were green.
3. **One test was wrong, not the code.** `primary_claim` was asserted to return `ALGO_NONE` on
   "quicksort pivot or binary search midpoint", but that is not a tie — `quicksort` and `pivot`
   are two distinct hits against one. The tie-break rule was behaving correctly and the test
   was rewritten to use a genuine tie. Worth recording because the reflex is to patch the code.
4. **The bag-of-words stub cannot test the shuffled-text control.** Under a purely lexical
   encoder, shuffling is a no-op by construction, so that test asserts equality and documents
   why. It becomes a real control only against the actual AR, which reads word order.

**Limitations.** Nothing here has touched a real activation. `steer.py` is verified on a 0.5B
model at layer 8, not on Qwen2.5-7B at L20; the AR is stubbed everywhere in
`test_steer_vectors.py`; the algorithm vocabulary's recall against real reads is unmeasured and
must be sampled once before any deception number is quoted. `ALGO_NONE` means "nothing in the
vocabulary fired", never "the read is contentless".

**Next steps.** Measure vocabulary recall on the banked reads (free, no GPU). Then, when GPUs
free up: the belief-readout gate (B2) — belief signal must clear the C1 confabulation floor
before B3/B4 are worth running, because "which algorithm is this" is a *specific* claim, and
specifics are the class N9 showed confabulate at 37%.
