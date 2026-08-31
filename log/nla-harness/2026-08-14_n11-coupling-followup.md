# 2026-08-14 — N11 coupling follow-up: the 3/3 was small-sample optimism (3/10 with a denominator)

*Follows `2026-08-13_n11-deception-results.md`, whose one live cell was behavioural coupling at
n=3. This run was built solely to give that cell a denominator.*

**Goal / hypothesis.** N11 found that every C2 item where the model *stated* the injected wrong
algorithm also carried a recurrent internal read (3/3), against 8.1% where it did not. Right
direction, right structure, three items. Prediction: with a real denominator the coupling
survives — read-hit rate should be materially higher among behaviourally deceived items.

**Setup.** Two stages, both unattended on **one GPU** (`autopilot.sh`):

1. **Behavioural screen** — 862 snippets × {C1, C2} = **1,724 batched generations**, ~43 min at
   41/min. Generation only: no NLA, no AV server, no activations, so it can batch (the read path
   cannot — batching was measured to break AV determinism). Marginal stated-hit rate on parsed
   C2: **2.6%**, closely matching N11's 3/77 — the base rate was not a small-sample artifact.
2. **Targeted capture** — 19 deceived items + 19 controls matched on **injected algorithm** and
   nearest **code length**, × 3 conditions = 114 captures, **113 ok / 1 error**, 2,319 reads,
   5.8 h. AV server + runner co-located on GPU 0, AR on CPU.

Quality: `rt_cos` median **0.860**, CJK **0/2,319**, no warnings. Parse rate C0 91.9% / C1 79.0% /
C2 89.5%.

**Results.**

**Coupling — the question this run was built to answer:**

| | read-hit | rate |
|---|---|---|
| model **stated** the wrong algorithm | 3/10 | **30.0%** |
| model did **not** state it | 2/24 | 8.3% |

Fisher exact **p = 0.138**, odds ratio 4.7.

Primary (item-level, echo-free, k=2): C0 2.7% · C1 2.6% · **C2 13.2%**, Δ +10.5 pt, discordant
**4 v 0**, McNemar **p = 0.125**. Foreign-read null for that statistic: C2 own 0.132 vs null
0.121, CI [0.026, 0.211] — **own sits inside the null CI**.

Read-level by locus:

| locus | C0 | C1 | C2 |
|---|---|---|---|
| CODE_ID | 0.028 | 0.010 | **0.378** |
| CODE_FAR | 0.054 | 0.063 | **0.063** |
| COT | 0.005 | 0.021 | 0.053 |
| ANSWER | 0.010 | 0.000 | 0.128 |

**Verdict — B3 remains NOT ESTABLISHED, and the follow-up is what shows why.**

1. **The 3/3 was small-sample optimism.** With a proper denominator it is **3/10**. The
   direction survives (30% vs 8.3%, OR 4.7) but p = 0.138, and the effect rests on three
   items in the numerator. This is the single most useful thing the run produced: N11's most
   promising cell, given the denominator it lacked, does not hold up. Getting that for ~6 GPU-h
   on a targeted set — rather than by scaling the read capture 20× to reach the same 10 deceived
   items — is what the cheap screen bought.
2. **`CODE_FAR` is completely flat: 0.054 / 0.063 / 0.063.** There is **no elevation whatsoever**
   away from the misleading identifier, in an item set deliberately enriched for deception. That
   is the cleanest statement of the whole line: the effect lives on the identifier token and
   nowhere else in the code. Combined with N11's distance gradient (collapsing to ~3% beyond
   50 chars), the echo account is now the better-supported one twice over.
3. **The `ANSWER`-locus elevation is confounded by the enrichment, and must not be read as
   dissociation.** C2 0.128 vs C1 0.000 looks striking, but half this set was *selected* because
   the model stated the wrong algorithm — so a read at the answer line naming it is reading the
   model's own committed output, not a hidden belief. In N11's unselected sample the same cell
   was 3.0%. Any verdict-locus claim needs a set not selected on the stated answer.
4. **The primary still does not clear its foreign-read null** (0.132 inside [0.026, 0.211]), now
   for the second time on independent data.

**Selection composition, stated because it bounds the claim.** `hashing` is **8 of 19** deceived
items (42%); palindrome 5, fibonacci 2, matrix_multiply 2, bubble_sort 1, prime_check 1. So the
deceived population is dominated by one injected algorithm, and the alternative reading is not
that the model is deceived by hash-like names but that hash vocabulary ("digest", "checksum",
"computes a hash") is easy to emit about many programs. The matched controls hold that base rate
fixed *within* the contrast — each hashing hit is paired with a hashing non-hit — so the coupling
comparison is not driven by it, but a coupling effect carried mostly by `hashing` should be
reported as such, not generalized.

**Limitations.** 10 deceived items is the binding constraint and no amount of read budget fixes
it — the base rate is 2.6% and the corpus is 862 snippets, so the ceiling is ~22 deceived items
without a new corpus. One capture lost to a transient
`RemoteProtocolError('Server disconnected')` (1/114); the run is keyed and idempotent so a rerun
would recover it. Stratified selection: read rates are reported within stratum and support no
unconditional deception rate.

**Next steps.** The honest position after two independent runs is that adversarial renaming moves
the *reading of the name* and not, detectably, the *belief about the program*, and that the
coupling cell — the last thing pointing the other way — did not survive its denominator. Raising
n further needs a corpus where the model is deceived far more often than 2.6%, which is a corpus
design problem, not a measurement one. Recommend closing B3 and moving read budget to the
steering arm (B4), whose gate (V1 vs the no-NLA contrastive V3) is untested and cheap.
