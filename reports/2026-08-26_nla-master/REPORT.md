# Reading the Residual Stream

**The complete NLA programme — Instrument 2 — from first principles to current status.**

*Master report · 2026-08-26 · written to be read cold, no prior familiarity assumed.*
*Supersedes nothing; consolidates `reports/2026-08-05_nla-out-of-box`, `2026-08-14_nla-full`,
`2026-08-14_nla-programme`, `2026-08-15_n13-torn`, `2026-08-17_believe-the-lie-programme`,
`2026-08-26_full-results`, and all 30 entries in `log/nla-harness/`.*

| | |
|---|---|
| Period | 2026-08-03 → 2026-08-26 (24 days) |
| Last material change | **2026-08-26** — B5 ran (§13, null); N10b decomposed (§9) |
| Subject model | `Qwen/Qwen2.5-7B-Instruct`, layer-20 residual stream (d = 3584) |
| Instrument | `kitft/nla-qwen2.5-7b-L20-{av,ar}`, released checkpoints, **frozen — nothing trained here** |
| Seed | 20260724 throughout (except the B0 grids — see §14) |
| Environments | `nla-mi` (transformers 5.12.1) · `transcoders-mi` (scoring) · `codesteer` (transformers 4.57.1) |
| Verbalized state reads | **~20,250** |
| Graded generations | **~14,592** |
| Injection failures | **0** |
| Experiments with a frozen decision rule | 8 pre-registered, 4 exploratory, 3 gates |
| Clear positives | **1** — population-level, and trimmed 2026-08-26 (§9) |

---

## Contents

- [0. The answer in one paragraph](#0-the-answer-in-one-paragraph)

**Part I — What is actually being measured**

- [1. The residual stream, and why it is unreadable](#1-the-residual-stream-and-why-it-is-unreadable)
- [2. The instrument: two models that translate](#2-the-instrument-two-models-that-translate)
  - [What a "read" is — and what it is not](#what-a-read-is--and-what-it-is-not)
  - [rt_cos — the instrument's self-check](#rt_cos--the-instruments-self-check)
  - [The write hook](#the-write-hook)
  - [What a read costs](#what-a-read-costs)
- [3. The stimulus ladder](#3-the-stimulus-ladder)

**Part II — The rules of evidence**

- [4. Why the nulls in this report mean something](#4-why-the-nulls-in-this-report-mean-something)
- [5. Verdict vocabulary](#5-verdict-vocabulary)

**Part III — The chronology**

- [6. Phase 0 — Does the released instrument even work here? (Aug 3–5)](#6-phase-0--does-the-released-instrument-even-work-here-aug-35)
- [7. Phase 1 — The confirmatory faithfulness family (Aug 6–7) · all three negative](#7-phase-1--the-confirmatory-faithfulness-family-aug-67--all-three-negative)
- [8. Phase 2 — Confabulation, measured (Aug 7) · exploratory](#8-phase-2--confabulation-measured-aug-7--exploratory)
- [9. Phase 3 — Safety readout: can a read tell you the code is malicious? (Aug 12–14)](#9-phase-3--safety-readout-can-a-read-tell-you-the-code-is-malicious-aug-1214)
- [10. Phase 4 — Deception: does a misleading name create a false belief? (Aug 13–14)](#10-phase-4--deception-does-a-misleading-name-create-a-false-belief-aug-1314)
- [11. Phase 5 — Torn-ness: can we detect confusion? (Aug 15–16) · N13, null](#11-phase-5--torn-ness-can-we-detect-confusion-aug-1516--n13-null)
- [12. Phase 6 — Steering: the causal arm, end to end (Aug 12 → Aug 26)](#12-phase-6--steering-the-causal-arm-end-to-end-aug-12--aug-26)
  - [12.1 The mechanism, built before any GPU touched it](#121-the-mechanism-built-before-any-gpu-touched-it)
  - [12.2 The ladder — five directions, arranged so a claim can fail](#122-the-ladder--five-directions-arranged-so-a-claim-can-fail)
  - [12.3 The bar, and why it is not zero](#123-the-bar-and-why-it-is-not-zero)
  - [12.4 The gate — B4, and it fails](#124-the-gate--b4-and-it-fails)
  - [12.5 The sweep — and it fails everywhere](#125-the-sweep--and-it-fails-everywhere)
  - [12.6 The one surviving lead, labelled as such](#126-the-one-surviving-lead-labelled-as-such)
  - [12.7 Why it fails — the directions point different ways](#127-why-it-fails--the-directions-point-different-ways)
  - [12.8 V5 — the composition rung](#128-v5--the-composition-rung)
  - [12.9 What the causal arm establishes](#129-what-the-causal-arm-establishes)
- [13. Phase 7 — Believe the Lie: the paper-shaped programme (Aug 15–17)](#13-phase-7--believe-the-lie-the-paper-shaped-programme-aug-1517)
- [14. Phase 8 — Unification, and the seed shock (Aug 16, Aug 26)](#14-phase-8--unification-and-the-seed-shock-aug-16-aug-26)
  - [The seed shock (Aug 26) — B0 does not replicate](#the-seed-shock-aug-26--b0-does-not-replicate)
- [15. Phase 9 — The steering triage and the read-side programme (Aug 27–31)](#15-phase-9--the-steering-triage-and-the-read-side-programme-aug-2731)
  - [15.1 A naming collision, resolved](#151-a-naming-collision-resolved)
  - [15.2 The triage — P0.1 to P0.4](#152-the-triage--p01-to-p04)
  - [15.3 The reproducibility floor, and what it costs](#153-the-reproducibility-floor-and-what-it-costs)
  - [15.4 The read side — the prompt is worth a token count](#154-the-read-side--the-prompt-is-worth-a-token-count)
  - [15.5 The ladder, and three deflations](#155-the-ladder-and-three-deflations)

**Part IV — What it all means**

- [16. The ledger](#16-the-ledger)
- [17. The single pattern behind every result](#17-the-single-pattern-behind-every-result)
- [18. Nineteen ways a result died here](#18-nineteen-ways-a-result-died-here)
- [19. What survives, and is worth keeping](#19-what-survives-and-is-worth-keeping)
- [20. Where it stands, and what to do next](#20-where-it-stands-and-what-to-do-next)
- [21. Caveats every headline number inherits](#21-caveats-every-headline-number-inherits)
- [22. Map — where everything lives](#22-map--where-everything-lives)

---

## 0. The answer in one paragraph

A Natural Language Autoencoder turns one number-vector of a model's internal state into English
and back. We pointed it at a 7B model reading deliberately obfuscated and malicious code, across
~20,250 reads and 24 days. **The instrument worked every single time** — round-trip fidelity stayed
in its validated band, zero reads failed. What it produces is **theme-level** content: *"this is
security-flavoured code"*, *"this mentions sorting"*, *"this is bookkeeping in a dispatcher loop"*.
It reliably separates **populations**. It does not resolve **individual items** — which capability
this malware has, which algorithm this program implements, whether this particular model run has
been misled. Nine attempts to extract item-level signal each produced an effect, and each effect
was reproduced by a matched control. The one positive is population-level (malicious vs benign)
— and a later decomposition showed most of even that rides on one generic word, leaving
**AUC ≈ 0.65** on vocabulary that cannot be dismissed as web-programming boilerplate. The causal test — write a corrected belief back in — **tied exactly** with a
contrastive difference vector that needs no autoencoder at all. And on 2026-08-26 the fallback
result the programme had retreated to (B0, a boundary condition on attention steering) **failed to
replicate at a second seed**, with the family average flipping sign.

Five days in late August closed the remaining question and sharpened the conclusion. A
pre-registered triage established that the causal failure is **not** the injection channel, **not**
a dead site, and **not** the wrong depth — the oracle, written at the most coherent depth in the
network, nets **exactly 0.0000: five items recovered, five damaged**. The read side then explained
the whole ledger of item-level nulls in one sentence: **at the position every one of them was
measured, a linear probe on the raw residual stream ties a token count** (+0.010 over reply
length). One route-specific effect survived pre-registered replication and then deflated three
times under its own controls, ending as *a signal about dispatcher count largely predictable from
how repetitive the text is*. Along the way the measurement itself was found to be unreliable:
**greedy decoding does not reproduce per item on a fixed GPU** (0.85–0.90 agreement), which
produced a false finding and then caught it. Reads, by contrast, are bit-exact.

---

# Part I — What is actually being measured

## 1. The residual stream, and why it is unreadable

A transformer processes text through stacked blocks. After block *k*, every token position holds a
vector — for this model, **3,584 floating-point numbers**. That vector is the model's entire working
state at that point in the text: everything it has accumulated so far that will influence what it
says next. It is called the **residual stream**, and it is not human-readable. There is no
dictionary that maps those 3,584 numbers to concepts.

Every experiment in this programme reads the residual stream at **block 20** of 28 — deep enough to
carry semantics, shallow enough that the state is not yet collapsed onto the next token.

## 2. The instrument: two models that translate

A **Natural Language Autoencoder** (NLA) is a *pair* of fine-tuned models that make that vector
readable in both directions:

| half | direction | what it does |
|---|---|---|
| **AV** — activation verbalizer | vector → English | takes one 3,584-number activation, emits 2–3 English sentences describing it |
| **AR** — activation reconstructor | English → vector | takes a description, emits a 3,584-number vector |

Both were fine-tuned from the same `Qwen2.5-7B-Instruct` we are reading, and released publicly
(method paper: *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*,
transformer-circuits.pub, 2026; checkpoints on Hugging Face under `kitft/`). **We trained nothing.**
Every checkpoint is frozen. This matters for interpretation: results here bound what *this* NLA at
*this* layer on *this* model can do, not what NLAs can do in general.

### What a "read" is — and what it is not

Point the AV at the model while it is reading a program, and at any token position you get an
English paragraph. We call one of these a **read**.

> **A read is not a gloss of the token it sits on.** It is a description of the model's *whole
> accumulated state* at that position. A read taken at the token `bubbleSort` describes everything
> the model is holding at that moment — which happens to include the fact that it just saw the word
> `bubbleSort`. Distinguishing "the model believes this program sorts" from "the model just read the
> letters b-u-b-b-l-e" is the single hardest measurement problem in this programme, and §10 is the
> story of how it was eventually settled.

### `rt_cos` — the instrument's self-check

Because the pair goes both ways, every read carries a free quality score:

```
v  →  AV  →  "this describes a loop accumulating a running total"  →  AR  →  v̂
rt_cos = cosine(v̂, v)
```

If the English captured what was in the vector, `v̂` lands near `v`. **`rt_cos` is the round-trip
cosine**, and it is the instrument's self-reported faithfulness. The validated working band is
**0.70–0.96**. Across the entire programme, observed medians ran **0.860–0.886**. It was never the
limiting factor — every negative result below is a real negative, not a broken pipeline.

**`rt_cos` turned out to be a *describability* meter, not a comprehension meter.** This was
established the hard way (§7, §11): it correlates **+0.28 to +0.31 with how much text the verbalizer
produced**, it is highest where there is something vivid to say, and it does *not* predict whether
the model's answer is right.

### The write hook

The AR runs backwards, so in principle you can *edit* a belief: reconstruct a corrected description
into a vector, and add it to the model's live state during generation.

```
h  ←  h + α · ‖h‖ · Δ̂          (Δ̂ = normalized direction, α = strength)
```

`nla/src/steer.py` implements this. It is verified byte-identical to unsteered generation at α = 0,
the edit lands at exactly the target position with neighbours bit-unchanged, and the cursor survives
the prefill/decode cache boundary.

> **One consequence that killed a planned remedy.** Injection rescales every activation to a **fixed
> L2 norm** before the verbalizer sees it. Magnitude is discarded entirely. Any plan of the form
> "re-derive the injection scale to fix a norm mismatch" is a provable no-op — which is exactly what
> B1 later ran into (§13).

### What a read costs

Sequential AV inference is **~1.5–4 s per read**, and it cannot be batched: an async client was
6.3× faster but only **15% byte-identical** to the sequential reference (`2026-08-06_s1-async-gate-failed`).
The AV is deterministic *sequentially* (60/60) — batching is the cause. Option B (sequential, default
config) was adopted to keep the banked corpus comparable, and it is why read budget is the binding
constraint on every design in this programme. Generations, by contrast, *can* be batched at ~41/min —
a fact that reshaped the N11 follow-up (§10).

## 3. The stimulus ladder

The parent project studies how models read deliberately obscured code. Every program exists at five
tiers, all **execution-equivalent** (verified):

| tier | transformation |
|---|---|
| **L0** | the original program |
| **L1** | identifiers renamed to meaningless strings (`var_bd90`) |
| **L1b** | identifiers renamed to *misleading* ones — a decoy asserting a **wrong** algorithm |
| **L2** | control flow flattened into a dispatcher loop |
| **L3** | everything at once |

**L1b is the interesting tier.** It is a Stroop trap: the code does one thing, the names claim
another. If a model forms a *false belief* from the names, you might be able to correct it. If it is
merely distracted, you cannot. That distinction is the programme's central question.

---

# Part II — The rules of evidence

## 4. Why the nulls in this report mean something

Nine of the results below are negative. A negative result is only informative if the test could have
come out the other way and the criterion was fixed in advance. Four devices enforce that here.

**Pre-registration with frozen decision rules.** HT12/HT13/HT14 (2026-08-06) and B4 (2026-08-16) had
their decision rules, refute clauses, and multiplicity corrections written into a dated log entry
*before* any data existed. HT12's refute clause (`|β| < 0.005`) is why its negative required no
renegotiation when the numbers landed.

**A null attached to the deciding statistic.** The signature move of this programme is the
**foreign-read null**: recompute the headline statistic using reads drawn from *other* items, holding
positions fixed. If borrowed reads score as well as an item's own, the statistic carries no
item-specific information. Variants: foreign-answer null (N8), permuted-label null (N10),
length-matched controls (N6, N10b), norm-matched random directions (B4).

> **The lesson that cost the most:** HT14's null guarded a *descriptive trajectory* while the
> deciding **log-rank test** ran unguarded — and a confounded p = 0.002 initially looked like
> support. A null must guard the statistic that carries the verdict.

**Judge-free grading wherever possible.** An 8B LLM judge separated real from shuffled read/text
pairs at **AUC 0.757** and could not rank a single item (**κ = 0.049** against a second rater,
ρ = 0.032 against a free baseline). It was retired. Everything since is graded by regex against a
ground-truth field the verbalizer never sees, or by closed-vocabulary matching where the ground truth
is exact *by construction*.

> **The most trustworthy measurement in the programme needed no model.** N7 spent two GPUs and 5,653
> judged comparisons on a score that cannot rank one item. A regex against a known ground-truth field
> labelled all 9,511 readings for free (N9). Look for a mechanically checkable claim *inside* the
> output before reaching for a judge.

**The V3 bar.** For any causal claim, beating "no steering" is worthless — any large enough
perturbation changes outputs (measured: a random norm-matched direction gains **+0.083** at high α).
The pre-registered bar is **V3**, a contrastive difference vector: average the internal-state
difference between clean and obfuscated versions of *other* programs and push along it. V3 uses no
verbalizer, no reconstructor, no English. **If V3 matches the NLA direction, the autoencoder is
buying interpretability but not capability.**

## 5. Verdict vocabulary

| verdict | meaning |
|---|---|
| **supported** | the pre-registered criterion was met, and the effect cleared its own null |
| **not established** | a real effect appeared, but a matched control reproduced it — it is not what it looked like |
| **refuted** | the criterion was missed, with a null informative enough to exclude the hoped-for effect size |
| **instrument null** | the measuring device failed its validity gate; the hypothesis was never adjudicated |
| **unmeasurable / not run** | found undefined on this data before it could be tested, or never executed |

---

# Part III — The chronology

## 6. Phase 0 — Does the released instrument even work here? (Aug 3–5)

**G0, the replication gate.** Stage A reproduces the reference implementation exactly (**0 / 101 over
tolerance**) after two real fixes: a `return_dict` incompatibility with transformers 5.12, and a
missing `<|im_end|>` in the reply layout. Stage B round-trips against the reference at median
|Δcos| **0.0037**. Mostly-CJK reads (the signature of injection failure) 0–3 / 101. **PASS.**

**The 44-read code probe — feasibility confirmed.** Out of the box:
L0 reads verbalize the true recurrence · L1b reads show a **decoy-vs-true mixture** (the first
DRM_AR signal, moving the predicted direction: −0.055 vs L0's −0.102, i.e. the decoy pulls ~+0.05) ·
L2 reads say "state machine" · CoT reads track the reasoning structure. Caveats recorded on day one
and still true: **specifics confabulate**, and **digit-piece tokens verbalize as numerology**
(`state_1395` → `1`,`3`,`9`,`5`) and must be excluded from any quantitative read set.

**Answer-line reads.** The NLA reads the *act of committing to a formatted answer* exactly; digits
come out as lossy variants. One read echoes the model's own **wrong** answer — the instrument is
faithful to the computation, not to the world.

**The consecutive-position scan — the one place structure beat the surface.** Reading 44 *adjacent*
tokens of the L1b prompt shows content accumulating token by token: generic "code block follows" at
`const`, decoy adoption as `smoothArea = (_lastNSecs` builds — and then, **at the exact position
where `== 0 || == 1` enters the prefix, "Fibonacci" appears in a read for the first time**, ~25
tokens before the recursion itself is visible. The base-case *shape* was a strong enough signature
to override the decoy *name*. Set this against B3's echo result (§10): the misleading name dominates
reads taken **on** it, while a structural cue elsewhere can override it — the two findings are
consistent, and together they bracket what the trap actually does.

**The flagship failure, diagnosed from internals.** On the L1b flagship the model answers 653 against
a truth of 927. Trap-identifier reads are a mixed decoy/generic soup — but **all six chain-of-thought
reads describe correct Fibonacci-recurrence evaluation with worked examples**. The diagnosis is
readable, and the accuracy bit alone cannot make it: *the decoy did not fool comprehension; arithmetic
execution failed.* (Squaring this with N4's "0 of 140 wrong traces contains internally false
arithmetic" is a live question — a trace can be locally self-consistent and still drop or skip a step.)

**An incidental reproduction.** One read reported the model's internal conclusion phrased in Chinese
(*"因此最终结果是：f(10) = 5…"*) — the NLA paper's own language-switching observation, reproduced in
the wild on a code task, and the thing that triggered the false-positive abort below.

**The overnight capture.** 380 tasks / **5,043 reads** / 0 errors / ~6.5 h unattended, after one
aborted run (a CJK monitor false positive — benign reads quoting Chinese, including one read
reporting that Qwen had internally concluded *in Chinese*; the detector was made fraction-based).

Behavioural payoff, out of the box: **tier accuracy is non-monotone**, on a model that was never in
the original panel.

| tier | Dataset A (n=10) | Dataset B (n=50) |
|---|---|---|
| L0 original | .70 | .64 |
| L1 neutral rename | .50 | .56 |
| L1b adversarial rename | **.80** | .60 |
| L2 flattened | .60 | .64 |
| L3 stacked | .50 | **.54** |

Adversarial renaming does not reliably hurt; **stacked obfuscation is the floor in both datasets** —
the papers' published pattern, reproduced with no tuning. Two things sit unresolved: **L1 ≤ L1b in
both datasets**, which nobody predicted and which single greedy runs cannot decide, and slice
prediction at **0.33** (the model over-includes the print line and drops initializer lines). Pooled
DRM_AR at trap identifiers ran L1b −0.0146 (333 reads) and L3 −0.0349 (172) — true-leaning under
crude template glosses, and reported as shape rather than as result.

## 7. Phase 1 — The confirmatory faithfulness family (Aug 6–7) · all three negative

Three hypotheses, rules frozen 2026-08-06 as one BH-FDR family, before any of N6–N8 ran.

**HT12 — does faithfulness predict correctness? ✗ REFUTED.**
An earlier analysis found correct runs more describable than wrong ones *in every tier*, all CIs
excluding zero. It was a **reply-length artifact**: wrong traces run ~1.6× longer and `rt_cos` falls
with length. Length-matched: **β = −0.0002, CI [−0.0051, +0.0047], p = .93** — an *informative* null
that excludes the banked effect size. Restricting the *same banked reads* to matched cases already
collapses Δ from 0.0145 to 0.0035.

**HT13 — does read↔reasoning alignment drop after the first error? INSTRUMENT NULL.**
The validity gate vetoed the test. G1 (shuffled AUC 0.757) ✅, G2 (beats the free AR baseline,
0.757 vs 0.733) ✅, **G3 κ = 0.049 ❌** — the second rater (Phi-3.5) answered "2" on 91% of items.
The score separates populations but is not a stable per-item property, so it cannot rank reads.
The artifact's alignment sort stays permanently disabled. *(Exploratorily there was no support
anyway: interaction +0.023, p = 0.22, wrong sign — a global late-trace decline, not a post-error one.)*

**HT14 — does the answer surface earlier in correct runs? ✗ REFUTED.**
**92% of cases never reach onset**, so the pre-registered median criterion is undefined. Own-answer
hit rate **2.8%** sits *below* the **3.6%** foreign-answer null. The significant log-rank (p = .002)
is an **answer-commonness artifact**: wrong answers are rarer and longer (median 0 vs 5 shared cases;
16 vs 7 chars; p = 1.3e-14).

> **BH-FDR across the family was vacuous, and is recorded as such.** No p-value entered it: HT12 died
> on a pre-registered trigger, HT14 on an unmeasurable criterion, HT13 was never adjudicated. The
> absence of an FDR table is a result, not an oversight.

Supporting instruments built in the same window: the **first-error oracle** (flagship known-answer
gate 4/4; coverage 41/117) — and the discovery that **0 of 140 wrong traces contains internally false
arithmetic**. The model computes *the wrong thing* correctly. Ground-truth execution is the only
mechanical error detector. Precise per-case error localization was declared **unsolved** (D2-state vs
D3 judge agree 39%; D3 self-agrees across seeds only 46%), and N5 was redesigned around a fixed
0.70–0.90 position strip instead of per-case error bursts. **N5 dense capture:** 91/91 length-matched
pairs, **4,653 reads**, ~4× read density, 0 errors, no instrument drift (232 re-used reads match
*exactly*).

## 8. Phase 2 — Confabulation, measured (Aug 7) · exploratory

The project's oldest qualitative claim — *"themes reliable, specifics confabulated"* — became a
number, using a regex and no GPU.

- **91%** of 9,511 readings name a programming language. **37% name the wrong one.**
- It is a **prior**, not noise: **79.9% wrong on JavaScript** (the reads say "Python" 3,162× vs
  "JavaScript" 798×) against **1.0% wrong on Python**.
- Structured by what is being read: **code tokens 12% wrong** vs **reasoning prose 46% wrong**.
- **Faithfulness barely catches it**: the wrong-language rate falls only 43.9% → 31.4% across `rt_cos`
  quintiles. This is the sharpest measured proof in the programme that **recovery ≠ truth**.
- A look-ahead/planning story was found and **killed by a foreign-reading null within ten minutes**:
  own readings anticipate 0.548 words vs a 0.532 null, CI [−0.022, +0.053].

Open and unrun: is the Python prior a property of the **verbalizer** or of the **subject's residual
stream**? One generation run separates them. (GPUs were busy; it has not been run since.)

## 9. Phase 3 — Safety readout: can a read tell you the code is malicious? (Aug 12–14)

**Corpus.** 600 DataDog `malicious_intent` PyPI samples fetched; **59% were near-duplicates**
(typosquat families publishing near-identical code) and were removed, leaving **183**, median 1,044
chars. Quarantine protocol: archives stay encrypted, extraction is in-memory only, and a static lint
proves no execution primitive can reach a sample. A second corpus (Sleeper Agents CWE safe/vulnerable
pairs, 850 pairs) passed its surface gate at AUC 0.638 after stripping a `#vulnerability` label leak
present on 43% of vulnerable completions.

**N10 — which capability does this sample have? NULL, and a framing confound.**
Two arms differing *only* in the preamble: `security` ("you are a security engineer reviewing a
package…") vs `neutral` (same task, every threat word removed).

| | security-framed | neutral |
|---|---|---|
| generic malice vocabulary | **17.5%** | **6.9%** |
| specific capability named | 15.5% | 11.4% |
| item-level any-capability | 87.4% | 70.5% |

**On identical code, the preamble more than doubles generic threat vocabulary.** A single-arm
design would have reported that as a fact about the residual stream.

> **A correction to how this has been stated.** The 2026-08-13 entry and both 08-14 reports say
> the effect holds on *"identical code and identical activations"*. The second half is wrong. The
> two arms are **separate captures** — a different preamble changes the prefix, so the state at
> every position differs and the model's reasoning trace differs outright; the malware-browser
> builder verifies that **0 of 14 read positions overlap** between arms. What was shown is that
> changing the prompt changes the whole pipeline's output, which is a real and important confound
> for single-arm designs. What was *not* shown — and what "identical activations" implies — is the
> much stronger claim that one fixed vector gets read differently under a different preamble.
> The unified corpus's "framing moves content, not fidelity" contrast inherits the same caveat:
> it compares two populations of reads, not two readings of one vector.

And the item-level statistic fails its own null:

| arm | own | foreign-read null | 95% CI |
|---|---|---|---|
| security | 0.874 | **0.903** | [0.858, 0.945] |
| neutral | 0.705 | **0.813** | [0.754, 0.869] |

87% looks strong until reads borrowed from *other* samples do slightly better. With a ~15% per-read
rate over 14 reads, P(at least one fires) ≈ 0.90 by base rate alone. The sharper "does it name the
**right** capability" test clears a permuted null by only +0.055 — against a null inflated to
0.53–0.61 because **78% of labelled samples are `REMOTE_EXEC`**.

**N10b — malicious vs benign at all? ✓ SUPPORTED — the programme's one positive, and the one
that has since been trimmed.**
183 malware vs **179 length- and bucket-matched benign controls**, 77.7% of them *hard* negatives
(benign code with a similar API surface), both framing arms, 0 errors.

| measure | value |
|---|---|
| item-level AUC, neutral arm | **0.764** [0.717, 0.808] |
| **AUC on the hard-negative stratum alone** | **0.757** |
| AUC, malice-margin measure | 0.856 |
| length confound AUC (matching check) | 0.477 |
| framing inflation on the read-fraction measure | +0.014 |

Holding at 0.757 on hard negatives means it is **not** an API-surface detector. This reconciles
cleanly with N10: reads carry **theme-level** malice ("something hostile is happening") without
**item-level** resolution ("which capability, exactly").

**Decomposition (2026-08-26) — most of the 0.764 is one generic word.** Splitting the vocabulary:

| vocabulary | AUC | 95% CI |
|---|---|---|
| full, as originally reported | **0.764** | [0.717, 0.808] |
| without `payload` | 0.684 | [0.638, 0.731] |
| **`payload` alone** | **0.708** | [0.665, 0.751] |
| **unambiguous-malice terms only** | **0.648** | [0.614, 0.685] |

`payload` alone nearly reproduces the full number, and removing it costs 8 points. It is **46% of
all malware vocabulary hits** (251 of 543) — malware does more HTTP POSTing than library code, and
"payload" rides along. This is the same shape as the `encoding` confound the grader validation
caught earlier: a label that looks like content and is partly detecting surface. *(Some terms are
clean: `webhook` fires 65× on malware and **0×** on benign.)*

**And the AUC hides the per-item error rates:**

| | |
|---|---|
| malware samples with **zero** malice reads (missed) | **54/183 = 29.5%** |
| benign samples with ≥1 malice read (false positive) | **41/179 = 22.9%** |
| worst benign false positive | the **`openai`** package, score 0.36 |

> **The honest statement.** Reads discriminate malicious from matched-benign code at **AUC ≈ 0.65**
> on vocabulary that cannot be dismissed as generic, rising to 0.76 only when a generic
> web-programming term is included. The signal is real, and it is population-level and
> theme-shaped — it says "something hostile is going on here", not which sample, not which
> capability, and it misses roughly three in ten malware samples outright.

**Grader validation** (on 5,090 real reads, before any of the above was believed): fire rate 0.205,
**not degenerate** (max label share 0.278, against N7's failed 0.91). It killed one confound —
`encoding` was the only label whose rate *rose* with obfuscation tier (1.68% → 3.69%), because
`\bencod\w*\b` matches "encoded variable names": **a label describing the obfuscation, not the
program**, and 10.7% of all fires. It also found the pre-registered `k ≥ 3` recurrence rule
**unaffordable**: median 2 gradeable reads per item, only 43.4% reach three, 17.9% have none.

## 10. Phase 4 — Deception: does a misleading name create a false belief? (Aug 13–14)

**Design.** Three conditions per Java snippet, differing *only* in identifiers.
**C0** original · **C1** neutral opaque rename (the confabulation floor) · **C2** adversarial rename
to a plausible **wrong** algorithm (`hasCloseElements` → `bubbleSort`, `Solution` → `BubbleSorter`,
`threshold` → `swapped`). Because *we* choose the injected algorithm from the grader's own closed
vocabulary, ground truth is exact **by construction** and scoring needs no judge. The injected label
is near-uniform across 10 algorithms (max class 11.4%) — deliberately avoiding N10's class imbalance.
C1 and C2 differ in exactly one respect: `AdversarialRenamer` subclasses the neutral renamer and
overrides only the name generator, so scope, collision and harness analysis are shared. If they
differed anywhere else, C1 would stop being a valid floor.

**The raw effect is enormous: 90% of programs vs a 19% floor, McNemar p < 1e-15.** Then you ask
*where* those reads are:

| locus | C0 | C1 (floor) | C2 |
|---|---|---|---|
| **on the misleading identifier** | 0.9% | 0.8% | **30.9%** |
| code, away from it | 1.6% | 1.8% | 3.0% |
| chain of thought | 1.4% | 2.2% | 3.6% |
| answer line | 0.3% | 0.4% | 3.0% |

By distance from the nearest misleading identifier, within C2:

| on it | 1–50 chars | 50–150 | 150–400 | 400+ |
|---|---|---|---|---|
| **30.9%** | 3.7% | 2.5% | 5.4% | **0.0%** |

> **A belief propagates through a representation; an echo is local. This is an echo.** The reads are
> reading the *word*, not a belief about the program. Reporting the 90% would have been reporting
> that the NLA can read a token.

**The echo-free residue does not clear its null.** Excluding the identifier locus and requiring the
claim to recur at ≥2 non-adjacent positions: C0 3.0% · C1 2.0% · **C2 9.0%**, discordant 8-vs-1,
**McNemar p = 0.039** — but the foreign-read null for *that same statistic* spans **[0.02, 0.10]**
and 0.090 sits inside it. **Verdict: NOT ESTABLISHED.**

*(k = 2, not the plan's k = 3, because the grader validation above found k ≥ 3 undefined for most of
the corpus. A confound found rather than designed around: the parse rate declines monotonically
C0 98% → C1 82% → C2 77% — renaming makes the model less likely to commit at all, shrinking the
answer-locus denominator exactly where it matters.)*

**N11-c — the coupling follow-up: 3/3 becomes 3/10.**
N11's one live cell was behavioural coupling: every C2 item where the model *stated* the injected
wrong algorithm also carried a recurrent internal read (3/3), against 8.1% where it did not.

The fix was not a bigger read capture. **Screen wide and cheap, then spend read budget only where
behavioural signal exists**: 862 snippets × {C1, C2} = 1,724 *batched* generations, ~43 min. Marginal
deception rate on parsed C2: **2.6%** — matching N11's 3/77, so the base rate was not a small-sample
artifact. Then 19 deceived + 19 controls **matched on injected algorithm and code length**, × 3
conditions.

| | read-hit | rate |
|---|---|---|
| model **stated** the wrong algorithm | 3/10 | **30.0%** |
| did **not** state it | 2/24 | 8.3% |

Fisher exact **p = 0.138**, OR 4.7. The direction survives; the significance does not.

**The cleaner result is a flat line.** In this set *deliberately enriched for deception*, reads away
from the identifier show **C0 0.054 · C1 0.063 · C2 0.063** — no elevation whatsoever — while the
identifier locus sits at 0.378. B3 was recommended closed.

## 11. Phase 5 — Torn-ness: can we detect confusion? (Aug 15–16) · N13, null

Can any internal signal show the model is **torn** — holding two incompatible readings at once, the
way a person hearing *"a plus b squared"* is split between `a + b²` and `(a+b)²`?

**Why this needed asking at all.** "Wrong" hides two opposite internal states, and no
correctness-tracking metric can separate them:

| | same answer every time | different answers |
|---|---|---|
| **right** | confidently right | lucky |
| **wrong** | **confidently wrong** — one reading, firmly held, mistaken | **torn** — two readings, unresolved |

Nothing in the project could see this distinction, for a mundane reason: **every stored run was a
single greedy generation.** One sample cannot tell you whether the model would answer the same way
ten times. The missing axis had to be built — K = 8 sampled answers per item, and K = 5
verbalizations of each *same* vector. (Greedy regeneration reproduced the stored answers 8/8; the
`act_norm` recovery landed on the recorded token 5,090/5,090.)

Three candidate signals, all **null** on the matched contrast:

| signal | AUC | permutation p |
|---|---|---|
| `rt_cos` (faithfulness) | 0.491 | .402 |
| `act_norm` (activation magnitude) | 0.442 | .983 |
| read instability across samples | 0.408 | — |

Trap tier does **not** induce torn-ness (Gate 1b −0.0005 [−0.082, +0.082]). Read length alone
explains **35%** of answer-entropy variance.

**The reason is circularity, and it is the most instructive finding of the phase.** Read instability
turned out to be **describability measured a second time**: r = **−0.757** per read, **−0.958 by
class** — the two columns are the same ordering, inverted. Its marginal ΔR² = .017 (p = .029) is
not a torn-ness detector.

> **A statistical bug that would have been the headline.** The horse race originally tested each
> signal by bootstrapping its ΔR² and asking whether the interval excluded zero. **ΔR² is
> non-negative by construction** — adding any column to a regression cannot lower in-sample R² — so
> that interval essentially always excludes zero. On synthetic data with three predictors *known*
> to be pure noise, all three came back "significant". Replaced with a permutation test, the same
> data gave p = .51–.58.

**One claim was withdrawn within the phase, by our own hardening.** The answer normalizer counted
`**[]` and `[]` as different answers (36.4% of items contain a `*`), and dict key order as different
answers — inflating the outcome variable. Under a hardened normalizer, the marginal instability
result moves **p .029 → .234** and all three internal signals are null. The same hardening
*strengthened* the free positive below.

**Not superposition — emptiness.** The design separated the two ways instability can arise by
contrasting decoy renames against meaningless ones (same token shape; they differ only in whether the
new name asserts a confident wrong meaning). **Decoy − meaningless = −0.002, CI [−0.006, +0.003]** —
the point estimate is *negative*. Instability is nearer an **emptiness** meter than a superposition
meter, and the worked examples say the same thing: the most stable read in the corpus sits on the
function name `unique` (five different sentences, one consistent meaning), while the least stable
sits on a **whitespace token**, where five verbalizations agree on the *form* and invent a different
domain each time — a DataFrame, an FSM, a finance model, a game state. That is not a model torn
between two readings. That is a decoder confabulating because the vector gave it nothing to say.

**So the trap makes the model wrong, not torn.** Adversarial renaming does not install a competing
interpretation the model wavers between; it installs a confident wrong one. This is the same fact the
faithfulness-by-token-type analysis had already implied — **misleading names read exactly as well as
real ones** (0.879 vs 0.878, n.s.), while *meaningless* renames read worst (0.847) and real function
names best (0.934). A decoy injects semantics that are wrong but perfectly describable.

**"Confidently wrong" is also rare here:** only 6 of 107 wrong items gave the same answer all eight
times, against 83 of 223 right items. On this corpus, being wrong and being unstable nearly coincide.

**Free positive, from a place we were not looking: how often the model changes its answer predicts
whether it is right, at AUC 0.843 → 0.869 after hardening.** That is better than any internal signal
in the programme, and it is purely behavioural.

> **Corrected 2026-08-31 (§15.4).** Both figures scored entropy against `modal_correct` — the
> plurality of the *same* eight samples the entropy was computed from. A held-out label was always
> available (`banked_correct`, from a separate greedy run) and gives **0.8400 → 0.8428**. The
> positive stands at ≈ **0.84**; what does not survive is the *sharpening* claim, which is +0.003
> held out rather than +0.026.

**Follow-on — the mid-trace dip is real and is not length.** `rt_cos` is U-shaped across the trace
while read length declines *monotonically* (683 → 633) and `act_norm` declines monotonically too —
the final bin is the lowest-magnitude yet the second-most-faithful. **Describability and magnitude
dissociate.** The arc reproduces on 5-sample instability, but the judge-free lexical measure is flat,
and both instability and `rt_cos` share AR machinery, so corroboration is weak. A composition
artifact was caught and controlled (hold token class fixed).

## 12. Phase 6 — Steering: the causal arm, end to end (Aug 12 → Aug 26)

Everything in Phases 0–5 is **observational**: read the representation, correlate it with behaviour.
This phase is the only one that intervenes, and it is what separates a *descriptive* instrument from
a *useful* one. Can you write a corrected belief back into the model and change what it says?

The whole arm — the write hook, the direction ladder, the gate, the sweep, and the composition rung —
is collected here rather than split across the programmes it was scheduled under, because the pieces
only make sense against each other.

### 12.1 The mechanism, built before any GPU touched it

`steer.py` adds a scaled direction to the model's live residual stream at chosen positions:
`h ← h + α · ‖h‖ · Δ̂`. It was written and unit-tested on **CPU first** (45 tests, later 86), because
a write hook that is subtly wrong contaminates every downstream number without ever erroring:

| property | verified |
|---|---|
| α = 0 is byte-identical to unsteered generation | ✅ |
| the edit lands at the target position, neighbours bit-unchanged | ✅ |
| the cursor survives the prefill/decode cache boundary | ✅ |

**Where the write actually lands, and how little of the model it touches.** This is the scope of the
entire causal claim, and it is smaller than the phrase "steering the model" suggests:

| | |
|---|---|
| **layers written** | **one** — a single forward hook on `layers[20]`, of **28** decoder blocks |
| **why that layer** | the AR is trained at **layer 20 only**, so V1 and V2 are *undefined* anywhere else |
| **token positions written** | **one**, by default — `last_prompt`, i.e. the final prompt token (`{prompt_len − 1}`) |
| **what is added** | `α · ‖h‖ · Δ̂` — a single 3,584-dimensional vector |
| **when it fires** | once on prefill, then once per decode step; a cursor tracks absolute position across the KV-cache boundary |

The hook is installed on **the same site `extract.ActivationExtractor` reads from**, resolved through
the same path-probing helper, so a read and a write can never drift onto different tensors. Other
position specs exist (`all_reply` writes every generated token, `all` writes everything) but the
pre-registered primary is the single final prompt token, matching the source protocol: *"at one token
position, at the layer the NLA was trained on."*

> **So the headline negative is precise about its own scope.** What failed is a **one-vector edit, at
> one position, at one layer**. That is exactly the intervention the source method specifies — but it
> is not evidence that no residual-stream intervention can move this behaviour, and the report should
> not be read as saying so.

The subtle part is position arithmetic under a KV cache: the hook fires once on prefill with
`seq_len == len(prompt)` and once per decode step with `seq_len == 1`, so a hook targeting absolute
position *p* cannot use the local index. Getting that wrong is **silent** — the edit lands on the
wrong token and the result reads as "steering doesn't work" rather than as a bug.

The judge-free grader (`belief_grade.py`) was built alongside it, and testing found four bugs —
three word-boundary/paraphrase misses caught by unit tests, and **one camelCase blind spot
(`\bprime\b` cannot match `isPrime`) caught only by end-to-end integration**, which is the reason
the suite has integration tests at all.

### 12.2 The ladder — five directions, arranged so a claim can fail

Each rung strips something away, so a positive result can be attributed to something:

| | direction | what it is | what it tests |
|---|---|---|---|
| **V1** | `nla_edit` | AR(edited explanation) − AR(original explanation) | the source method: edit the belief *in English* |
| **V2** | `word_edit` | the same, but **exactly one word differs** | does a single lexical item carry the effect |
| **V3** | `task_vector` | mean `h_clean − h_obf` over **other** matched pairs | **needs no autoencoder at all** |
| **V4** | `item_oracle` | this item's own `h_clean − h_obf` | upper bound — **not deployable**, it sees the answer |
| **V5** | `combined` | V1 stacked on attention steering | do the two channels compose (§12.8) |

Plus a control battery: a **random** norm-matched direction, a **foreign** direction borrowed from
another item, the **antipodal** (V1 reversed), and **prompt-only** — just adding a sentence.

> **Two design guards worth copying.** V3 is fit on pairs and applied to items, so if the applied
> item sits inside the fitting set the vector has already seen its answer. `direction_for` therefore
> *requires* an `exclude` argument and raises without it — **the leakage guard is in the type
> signature, not in a comment**. And V4 is labelled an oracle in every table, because reporting it
> beside V1 and V3 unlabelled would be straightforwardly misleading.

### 12.3 The bar, and why it is not zero

**V1 must beat V3.** Beating "no steering" proves nothing — any large enough perturbation changes
outputs. V3 is a plain contrastive difference of activations: no verbalizer, no reconstructor, no
English. **If V3 matches V1, the autoencoder is buying interpretability but not capability** — a
publishable finding, but a different paper, and one that had to be discovered here rather than by a
reviewer.

### 12.4 The gate — B4, and it fails

**All seven conditions at the pre-registered strength**, so the whole control battery is visible at
once (60 programs, 420 generations):

| condition | what it actually does | Δaccuracy | 95% CI |
|---|---|---|---|
| **prompt-only** | No steering at all. Append one sentence to the prompt warning that the identifiers may be misleading. The cheapest possible intervention, and the honest competitor to all of it. | **+0.100** | [−0.017, +0.233] |
| **V4 · oracle ceiling** | Push along *this* program's own `h_clean − h_obf`. It has seen the un-obfuscated version, so it already knows the answer — **not deployable**, included only to show how much headroom exists. | +0.083 | [−0.033, +0.200] |
| **V1 · NLA direction** | The method under test. Verbalize the state, edit the English explanation to say the true thing, reconstruct both, subtract, inject the difference. | **+0.050** | [−0.083, +0.183] |
| **V3 · difference vector** | Average `h_clean − h_obf` over *other* matched programs and push along that. **No verbalizer, no reconstructor, no English** — the bar V1 has to clear to have earned its autoencoder. | **+0.050** | [−0.050, +0.167] |
| **antipodal** | V1 **negated**. If the direction carries real signal, pushing the opposite way should drive the model *deeper* into the false belief and hurt. | +0.033 | [−0.083, +0.150] |
| **random** | A random vector rescaled to the same norm as V1. Tests whether *any* perturbation of this size nudges accuracy, independent of meaning. | +0.033 | [−0.067, +0.133] |
| **foreign** | V1 built for a **different** program, injected into this one. Tests whether the effect is item-specific or just a generic shove. | **−0.017** | [−0.117, +0.083] |

*(All seven inject at **layer 20 only**, at the **single final prompt token** — see §12.1. Only
prompt-only touches no activations at all.)*

**How to read it.** Every interval spans zero, so no condition is individually significant — the
comparison that carries the verdict is V1 against V3, and they are identical to three decimals. The
ordering is still informative: **foreign is the only negative**, which is the control that *should*
be worst, so the battery is behaving. **Antipodal lands exactly on random**, which is the sign
failure. And **prompting beats the oracle** — the cheapest intervention in the table outscores the
condition that gets to read the clean program.

**The pre-registered gate, at α = 1.0:**

| | V1 (NLA) | V3 (no NLA) | Δ | discordant | McNemar p |
|---|---|---|---|---|---|
| α = 1.0 | **0.550** | **0.550** | **0.000** | 7 v 7 | **1.00** |

### 12.5 The sweep — and it fails everywhere

**At every other strength** (2,160 generations, 0 errors, a 16× range of α):

| α | V1 | V3 | Δ | p | q (BH) |
|---|---|---|---|---|---|
| 0.25 | 0.583 | 0.567 | +0.017 | 1.000 | 1.00 |
| 0.5 | 0.517 | 0.550 | −0.033 | 0.727 | 1.00 |
| 1.0 | 0.550 | 0.550 | 0.000 | 1.000 | *(primary)* |
| 2.0 | 0.633 | 0.617 | +0.017 | 1.000 | 1.00 |
| 4.0 | 0.550 | 0.550 | 0.000 | 1.000 | 1.00 |

Three details make this more than an underpowered shrug:

1. **Reversing the NLA direction performs identically to a random one.** If the direction carried
   signal, reversing it should hurt. It does not — the direction has **no consistent sign**, which is
   not a magnitude problem, and no strength fixes it. A geometry check found no bug:
   cos(V3, V4) = +0.418, and V1 is near-orthogonal to both (cos ≈ 0.02).
2. **Adding one sentence to the prompt beat every steering condition** (+0.100) — including the
   oracle that gets to see the clean program.
3. **A random, norm-matched direction gains +0.083 at the strongest setting.** Any sufficiently large
   perturbation nudges accuracy on this task. This is the single most important line for anyone
   reading a raw Δaccuracy column, and it is why the bar is V3 and not zero.

> **The interim reversed.** At n = 34 this looked exactly like the pre-registered success. At n = 60
> it tied exactly. That is the second result in the same month to reverse at full sample size.

### 12.6 The one surviving lead, labelled as such

**V2** — the *minimal single-word edit* rather than a
whole-gloss swap — is the only condition with a monotone dose-response (Δacc 0.000 → 0.033 → 0.117 →
0.117 → **0.183**), and at α = 4 it separates from V3: **0.683 vs 0.550**, discordant 12 v 4,
**p = 0.077**. It is exploratory, does not clear BH (q ≈ 0.385), and is **one cell out of 25**. It is
theoretically sensible — V2 is the *more* NLA-dependent intervention, needing the reconstructor to
render one specific lexical substitution, while V1's whole-gloss swap could differ in any of up to
eight terms. **It cannot be confirmed on this corpus**, which is hard-capped at 70 programs, 60 of
them already spent.

> **One dimension was deliberately not swept, and the reason is worth keeping.** The AR is trained at
> layer 20 only, so V1 and V2 are *undefined* anywhere else. A layer sweep would therefore sweep only
> the no-NLA baselines and **structurally favour V3**. It was recorded as a limitation rather than
> run as a courtesy. The pre-registered **injection-position** sweep (`all_reply` alongside
> `last_prompt`) is a different matter: the config's preamble describes the result as holding "at
> three injection positions", and the run manifest shows `--positions last_prompt` only. That claim
> should be trimmed to the alpha range, which is what the log entry actually reports.

> **V2 was silently broken until the sweep session.** Its decoy→true pairing *zipped* two
> identifier-term lists that differ in length on **52 of 60 pairs**, so only **8** directions were
> being built. Had this not been caught, the single most interesting cell in the sweep would have
> been reported on 8 items.

### 12.7 Why it fails — the directions point different ways

The negative has a mechanical explanation, and it is measured rather than inferred. On 164 Java
programs:

| pair | cosine |
|---|---|
| **V1 (NLA language-space) ↔ clean-minus-obfuscated** | **−0.001** [IQR −0.019, +0.021] |
| a *random* vector ↔ the same | −0.000 |
| V3 ↔ V4 — as they must, both being activation differences | **+0.702** |

**The belief direction is orthogonal to the direction that matters.** That is why reversing V1
behaves exactly like reversing noise, why no injection strength rescues it, and why the tie with V3
is a coincidence of outcomes rather than two names for one vector. It reproduces independently on
the Python/JS corpus, where the same cosine came out ≈ +0.02.

### 12.8 V5 — the composition rung

V5 is the rung that asks whether the **positional** lever (attention reweighting, which redistributes
focus across words already in the prompt) and the **semantic** lever (the NLA residual write) add,
interfere, or are simply independent. It is a different question from B4's: not *which lever is
better*, but *do they combine*. It sat unbuilt for two weeks for an infrastructure reason.

**Ran 2026-08-26 — NULL.** It needed one process driving both attention reweighting and the NLA
residual write — and they live in
different conda environments (`codesteer` on transformers 4.57.1, `nla-mi` on 5.12.1), with the
attention backend importing transformers-4.x decoder internals. The plan therefore moved the NLA hook
*to* the attention environment, resting on an **asserted** claim — "`steer.py` has no transformers
coupling" — that had never been tested there. **2026-08-17: the claim holds.** Under the `codesteer`
env, α = 0 is byte-identical, an edit at position 5 changes exactly `[5]`, neighbours bit-identical.

*A false alarm worth recording:* the first attempt reported positions `[5, 17]`. **The defect was in
the test, not the hook** — `ActivationSteerer.reset()` rewinds the position cursor but does *not*
uninstall the spec, so the "clean" reference pass was itself still steered. This is the project's own
standing lesson (*when a test contradicts a fix, suspect the test*) and it argues for an API change.

**The run, 2026-08-26.** Six conditions on 164 Java programs, **1,930 cases per cell**, unattended
across the day. The hard validation gate ran first: with the belief channel at α = 0 — a unit-tested
no-op — and attention steering ON, the patched runner had to reproduce the banked attention-only
number. **62.746 vs banked 61.710 on identical cases, |Δ| = 0.0104 against a 0.02 tolerance: PASSED.**
The integration does not change the measurement.

| contrast | estimate | 95% CI (cluster bootstrap over programs) |
|---|---|---|
| attention only | −1.30 | [−6.31, +3.61] |
| belief only | −2.12 | [−6.76, +2.39] |
| both levers | +0.52 | [−3.37, +4.38] |
| **both − attention** | +1.81 | [−2.61, +6.16] |
| **both − random-direction control** | +2.33 | [−2.17, +6.74] |
| **disjoint − both** *(collision check)* | +0.10 | [−3.93, +4.26] |

Raw cell accuracies: none 64.04 · attn 62.75 · nla 61.92 · both 64.56 · rand 62.23 · disjoint 64.66.

**Not one interval excludes zero.** The point estimates read like a composition result — the pair
beats either lever alone and beats a random direction by 2.33 — and every one of them dissolves
against its interval. **The layer arrangement, explicitly.** The NLA write is one hook at **layer 20**. Attention steering
is applied to the **last 8 layers — 20 through 27** (`--steer-last-n-layers 8`). They therefore
*overlap at layer 20*, which is the whole reason the collision control exists: the disjoint cell
reruns attention steering over **7 layers, 21–27** (`--steer-last-n-layers 7`), leaving layer 20 to
the belief channel alone.

**The one control that did its job:** attention steering occupies layers 20–27
and the autoencoder lives at layer 20, so "both on" could have been a *collision* rather than
composition. Moving attention to layers 21–27 changed the result by **+0.10** — not a collision, but
also not an effect.

> **Two things bound this run.** It is **one seed** (1000, temperature 0.9) — and E5b, logged the
> same week, measured per-transform swings of 2.6–7.8 points between seeds on this exact pipeline,
> which is larger than every contrast in the table. Within B5 itself the unsteered `none` cell sits
> at 64.04 against a banked 62.28 for the same nominal condition, a 1.76-point gap on nothing.
> Second, the subject is `Qwen2.5-Coder-7B-Instruct` (the only model where attention steering was
> replicated), while the V1 directions were exported from `Qwen2.5-7B-Instruct` — the transfer **B1
> refuted**. Composition forced a single model and there was no choice that leaves both arms clean.

### 12.9 What the causal arm establishes

- **NLA-derived steering does not beat a difference vector that uses no NLA** — at the pre-registered
  strength, across a 16× range of α, against seven controls.
- **Neither beats adding a sentence to the prompt**, which was the best condition in the battery and
  sat above the oracle.
- **The direction has no consistent sign**, and the geometry in §12.7 says why.
- **Composing the two levers changes nothing measurable** — on one seed, with every interval spanning
  zero, and with the collision control confirming the null is not an artifact of both levers
  occupying layer 20.
- **One lead survives**: V2, the minimal single-word edit — the *more* NLA-dependent intervention,
  since it needs the reconstructor to render one specific substitution — and it is unconfirmable on a
  70-program corpus.

## 13. Phase 7 — Believe the Lie: the paper-shaped programme (Aug 15–17)

Seven research questions, B0–B6, written as a paper outline: *read the belief, show it predicts
failure, correct it causally, and report honestly how much of that correction actually needs the
autoencoder.*

**B1 — can the released NLA read a *different* model's state? ✗ REFUTED (and the failure is
interesting).** `Qwen2.5-Coder-7B-Instruct` shares the architecture, the `tokenizer.json` blob, the
chat template and every dimension, so the checkpoint loads. 164 Java programs, 820 reads per model,
read at **identical positions** — the only variable is the weights.

| gate | verdict | detail |
|---|---|---|
| median `rt_cos` ≥ 0.70 | **FAIL** | **0.694** (host 0.864) |
| within 0.05 of the host | **FAIL** | gap **+0.170** |
| own vs foreign activation | PASS | Δ +0.355 [+0.344, +0.366] |
| activation-norm band | PASS | 154.0 (host 111.7) |
| mostly-CJK reads | PASS | 0 / 820 |

**It is not a scale artifact, and that is the point.** The sibling's activations are 38% larger — but
injection normalizes magnitude away entirely, so **the loss is directional**: the sibling's layer-20
directions have diverged far enough that a frozen verbalizer reads them measurably worse. This also
proves the plan's "re-derive the injection scale" remedy is a no-op. Degraded, not broken:
own-vs-foreign +0.355 means the reads still carry substantial item-specific information.
**Consequence:** the subject model stays `Qwen2.5-7B-Instruct`. No downstream re-work.

**B2 — does the stated belief beat a confabulation floor? UNMEASURABLE — and it was the *gate*.**
Scoring needs a ground-truth "true algorithm" per program, and only about **11 of 70** stimuli have
one the closed vocabulary can score (these are programs like "find the closest two elements", not
textbook named algorithms). Widening the vocabulary is not a fix — the surface-tracking diagnostic
built for exactly this purpose caught a label that fired *more* as code got harder to read.

> **B3 and B4 ran anyway, without their gate, and both then failed.** That ordering is a genuine
> weakness in the programme and is recorded as one rather than explained away.

**B5 — do the positional and semantic levers compose? RAN 2026-08-26 — NULL.** The composition rung
is the ladder's V5, so it is reported with the rest of the steering arm in **§12.8**.

**B6 — does read↔reasoning agreement predict correctness? NOT ADJUDICATED.** Same instrument null as
HT13. A judge-free version — predict correctness from reply length alone versus length plus
faithfulness — is nearly free and **has still not been run.**

## 14. Phase 8 — Unification, and the seed shock (Aug 16, Aug 26)

**The unified reading table (Aug 16).** Three capture passes had produced readings in three different
row shapes, which kept them un-comparable. `unify_corpora.py` merges them into one schema:
**14,619 readings / 928 cases**, with `act_norm` present on 10,198. Two things were fixed *at the
boundary*: the dense pass deliberately re-used **232** readings, which are dropped so nothing is
double-counted (5,090 + 4,421 = 9,511, reconciling exactly with the results browser); and **malware
rows are redacted at the unification boundary, not downstream**, because derived data inherits
quarantine status.

First cross-corpus contrast — reasoning readings, obfuscated vs malware:

| | `rt_cos` [95% CI, case-clustered] | mean read length |
|---|---|---|
| obfuscated (n = 6,401, 330 cases) | 0.8539 [0.8508, 0.8569] | 658 |
| malware (n = 2,184, 182 cases) | **0.8928** [0.8901, 0.8952] | 719 |

Raw gap +0.0389 — but `rt_cos` correlates +0.309 with read length, and the malware readings are 61
chars longer. **Length-adjusted: +0.0152 [+0.0073, +0.0232] — 61% of the raw gap was length, and the
remainder survives.** Two more, newly askable: `act_norm` is higher on malware (113.32 vs 109.53,
non-overlapping), and **framing moves the *content* of a reading without moving its fidelity** —
the security and neutral arms sit at `rt_cos` 0.8700 and 0.8730 (indistinguishable) while their
capability-naming rates are 0.874 vs 0.705. A clean dissociation, and all of it consistent with the
describability account: malware is semantically vivid ("this steals credentials and posts them to a
webhook" is an easy sentence); obfuscated benign code is semantically thin (dispatcher bookkeeping is
a hard sentence).

### The seed shock (Aug 26) — B0 does not replicate

By 2026-08-17 the programme had a fallback: **B0**, a powered boundary condition on CodeSteer's
attention steering, measured on 90 units / 12,432 generations across nine transforms. Its headline was
a *family contrast*: CodeSteer's own four transforms **+3.89**, the four they never tested **−3.14**,
difference **+7.02, 95% CI [+2.00, +10.09]**. "Attention steering helps where its authors tested and
hurts where they did not."

**A second seed (108 units / 15,018 generations / 0 failures) flips the sign.**

| transform | seed 1000 | seed 2000 | swing |
|---|---|---|---|
| arith_rewriting | **−8.22** | **−5.65** | 2.6 |
| branch_inversion | −2.21 | +0.66 | 2.9 |
| opaque_predicates | −3.54 | **+4.22** | **7.8** |
| loop_transformation | −0.87 | +2.92 | 3.8 |
| adversarial_rename | −0.57 | +3.16 | 3.7 |
| **family of 5** | **−1.24** [−4.41, +1.41] | **+4.46** [+0.85, +7.09] | **sign flip** |

Four of five transforms move from negative to positive. **The published contrast is therefore not
established** — it was a one-draw result reported with a CI that described sampling *within* the draw
rather than variability *across* draws. Note that every individual per-transform CI already spanned
zero and was reported that way; the family contrast was the only powered statistic, and it is the one
that does not survive.

Two honesty notes recorded with it. (i) The seed-2000 replicate omits CodeSteer's original four
transforms (their seed-1000 anchors existed under a different experiment id), so the *full* old-vs-new
contrast cannot be computed at the second seed at all — closing that needs ~20 GPU-h. (ii) The
seed-1000 arm **predates the `--seed` artifact fix**, so those runs are uncontrolled draws at
temperature 0.9. That does not rescue the finding: it means run-to-run variability is large enough to
flip the sign of a five-transform family average, which is the substantive problem either way.

**What survives:** `arith_rewriting` is clearly negative at both seeds (−8.22, −5.65) — the one cell
CodeSteer itself flagged. **A single-transform claim with two seeds is defensible where a
five-transform family average is not.**

---

## 15. Phase 9 — The steering triage and the read-side programme (Aug 27–31)

Everything above was written on 2026-08-26. What follows is five days of work that closed the
causal arm's open question, discovered a measurement floor that invalidated one of its own
findings, and then walked a promising positive down three levels of deflation. It is the only
phase in this report where the *method* is the main output.

### 15.1 A naming collision, resolved

The ledger calls this work **"Phase 0"** — a triage numbered P0.1–P0.4. This report already used
"Phase 0" for §6, *does the released instrument work here*. They are unrelated. In this report the
triage is Phase 9; in `log/nla-harness/` it is Phase 0. Both names are load-bearing in their own
documents, so neither is renamed — but a reader moving between them needs to know.

### 15.2 The triage — P0.1 to P0.4

**Why it existed.** B4 and B5's negatives were being read as *"there is no item-level belief to
edit"*. That reading was **not licensed**, because a second explanation predicts identical tables:
the injection channel cannot deliver anything, whatever it carries. The evidence sat inside B4's
own battery — **V4, the oracle that has seen the clean program, gained only +0.083, less than
adding one sentence to the prompt**. If ground truth cannot clear a prompt sentence through this
channel, no direction was going to, and the gate measured *delivery* rather than *belief*.

Four experiments, each pre-registered with a frozen decision rule before any run.

| stage | question | verdict | the number |
|---|---|---|---|
| **P0.1** | is the single-layer constraint hard? | **HARD** | min cos(Δ₂₀, Δ_ℓ) over ℓ ∈ [21,27] = **0.277** vs a frozen 0.50 |
| **P0.2** | is the channel the bottleneck? | **NOT CHANNEL-LIMITED** | V4 goes **+0.0833 → −0.3333** when the write widens to every reply position; contrast **−0.4166** against a frozen +0.10 |
| **P0.3** | is layer 20 a live site for *any* intervention? | **SITE LIVE** | [20,20] attention steering moves P@1 by **−1.434 / −1.658** against a **0.794** baseline seed spread |
| **P0.4** | is the instrument at the wrong *depth*? | **NOT DEPTH-LIMITED** | V4 at layer 13 − layer 20 = **−0.0667** [−0.1833, **+0.0500**] against a frozen +0.10 |

**P0.2's negative is stronger than "no delivery".** Widening the write drove V1's parse rate to
**0.100** (from 0.917) and V4's to 0.383. The channel does not under-deliver — it **over**-delivers
and destroys generation before it can carry content.

**P0.4 is the sharpest number in the triage.** P0.1 had found cross-item coherence peaking at
**layer 13 (0.543)** while magnitude peaks at layer 20 (0.459), the two directions nearly
orthogonal at cos 0.21 — so "the instrument is at the wrong depth" was a live explanation.
Writing the **oracle** at layer 13 nets **exactly 0.0000: five items recovered, five damaged**,
with parse rate 0.833 intact. Perfect information about the true semantics, written at the most
coherent depth in the network, produces a pure shuffle. Unlike P0.2 this is *indifference*, not
destruction — and the CI's upper bound of +0.05 sits below the frozen +0.10, so the effect is
**excluded**, not merely unfound.

A gate ran first that the codebase had asserted in comments since `steer.py` was written and never
tested: that `ActivationExtractor` and `ActivationSteerer` hook the same decoder block. Residual
**8.5e-08 / 8.8e-08 / 4.3e-08** at layers 6/13/20, one position written, nothing else moved.

**What the triage establishes.** Three pre-registered ways of blaming the *apparatus* are now
closed: the channel is not the bottleneck, the site is not inert, the depth is not wrong. What
remains is the reading the programme had been circling — **there is no item-level belief at this
site to edit** — now bought with 2,880 rows and a frozen rule rather than an argument.

### 15.3 The reproducibility floor, and what it costs

An unplanned finding, and it invalidated one of this phase's own results before the week was out.

**Greedy bf16 decoding does not reproduce per item.** Two runs of an identical command, on the
**same physical GPU** (UUID-verified), same seed, disagree on **6 of 60** items — agreement
**0.90** unsteered, **0.85** steered — while cross-node agreement (0.95) is *no worse*. The cause
is run-to-run, not card-to-card. The seed never touches this path: greedy decoding draws no RNG.

It survives three interventions:

| intervention | same-card agreement |
|---|---|
| default | 0.900 / 0.850 |
| `use_deterministic_algorithms` + pinned cuBLAS/cuDNN/SDPA | 0.8333 / 0.9000 |
| generation budget 1100 → 2048 | 0.8833 / 0.8833 |

Deterministic kernels applied cleanly — zero fallback warnings — and moved nothing. The cause
**remains unidentified**, and after three failed explanations the honest position is to quote
0.85–0.90 and stop looking.

**Two things bound the damage.** Reads are **bit-exact** — 120/120 identical, max |Δ| = 0.0, within
and across processes — so `rt_cos`, `act_norm`, the P0.1 layer curves and the 14,619-row unified
table carry **no** reproducibility caveat. The floor applies only to graded generations: B4's
McNemar gate, B5's cells, P0.3's P@1 tables, P0.4's own primary. And raising the budget to 2048
took parse rate **0.8833 → 1.0000** at unchanged accuracy, so **12% of items in every banked
accuracy were scored wrong for not finishing rather than for being wrong** — a separate problem
from reproducibility, and one worth fixing in future runs.

**The floor immediately produced a false finding and then caught it.** A per-layer probe reported
correctness decodability rising monotonically to the final layer. A second label draw — differing
on 8 of 60 items, exactly the floor — produced the *opposite* curve, argmax L27 versus L13 at
near-identical mean AUC. **The level replicates; the shape does not.** The fix was free: **ten
cheap label draws beat one careful grade**, turning an argmax that flipped between runs into curves
correlating 0.79–0.94 across independent halves.

### 15.4 The read side — the prompt is worth a token count

With the causal arm closed, the question becomes what a *read* is worth, measured against the dense
baseline this project's charter requires.

**The headline is a negative and it is the most useful result of the phase.** At `last_prompt` —
the site of **every** item-level read in this programme, B4's vectors, P0.1, P0.2, P0.4 included —
a linear probe on the raw residual stream predicts correctness at ρ = **+0.3730** against a
reply-length baseline of **+0.3626**. An advantage of **+0.010**. Measured per tier with each
tier's own baseline, the residual stream adds **+0.04 to +0.06**, clean or obfuscated.

*Every item-level read in this programme was taken at a position where a linear decoder ties a
token count.* That single sentence explains B4, B5, N11, N13 and P0.4 without invoking beliefs.

Three further results, all pre-registered:

- **`rt_cos` adds nothing over reply length** (LRT p = 0.27, ΔCV-AUC −0.0027, n = 512), settling
  B6 judge-free. Its solo AUC of 0.654 is the describability confound, now measured. **Reply length
  alone reaches AUC 0.306 — inverted** — beating every internal signal but answer entropy.
- **N10b is population-only.** Split-half reliability of the 14-read instrument is
  **0.581 / 0.613 / 0.523** against a pre-stated 0.70, falling to 0.477 on hard negatives. The
  programme's one positive separates crowds and **cannot rank a single file** — and the best
  population discriminator (`mm_mean`) is the *least* reliable per item.
- **The signal near the answer is readout, not prediction.** `rt_cos` at the `answer_line` does beat
  length (q = 8e-05) and correctness is decodable there at ρ = 0.5702 — but a pre-registered
  sceptical branch decided it: **|len(answer) − len(truth)|**, knowable only if you know what the
  model will emit, reads **+0.3650 at the prompt — *below* its own +0.3767 truth-length baseline** —
  and +0.5299 at the answer line. The prompt carries no information about what will actually be
  emitted, so the answer-line signal is a corollary of the answer having been decided.
- **N13's standing baseline was re-estimated and survives.** A flag raised mid-week — that its
  answer-entropy positive shared a same-sample dependency — was **half wrong and is withdrawn**.
  The reported 0.843/0.869 do use `modal_correct`, the plurality of the same eight samples; but
  entropy comes from eight sampled generations while `banked_correct` comes from an independent
  greedy run, so a held-out estimate always existed: **0.8400 / 0.8428**. Inflation is **0.026**,
  not the 0.18 that a five-shared-draw instability measure suffered. What does fail is the claim
  that hardening *strengthened* the signal — held out it moves +0.003, so most of the gain lived in
  the label.
- **Mid-reasoning is empty**, on five independent measures across two tiers. And the explanation is
  mundane: probing the mid-reply region for a target *known* to be encoded at the prompt
  (`n_dispatcher_spans`, ρ = +0.8879) gives **+0.1773**. Activations at reply position *t* are about
  the reply; the KV cache holds the prompt without re-representing it.

**Depth for a read is a non-question.** A per-layer probe on L0-vs-L1b sits at ceiling in **28 of
28 layers** — **AUC 0.9953 at layer 0**, because adversarial renaming changes the literal tokens
and the contrast is separable at the embedding layer. The pre-registration had named this
UNINFORMATIVE branch in advance. It retires a whole family of "a probe detects obfuscation"
experiments that would have looked successful.

### 15.5 The ladder, and three deflations

The one place the residual stream did beat its baseline, and what happened to it.

**Discovery.** Probing all five tiers, each against **its own** length baseline, selection-free
(mean ρ over 28 layers, no argmax):

| route | tiers | beats own length baseline |
|---|---|---|
| clean | L0 | −0.0593 |
| **atom** | L1, L1b | **−0.2006** |
| **relational** | L2, L3 | **+0.1167** |

L1 is the extreme case: ρ = **+0.0224** — nothing at any depth on nonsense-renamed code.

**It replicated.** Five independent draws, rule frozen on the *route contrast* rather than
per-tier bars: **Δ = +0.3195** against a bar of **+0.15**, within 0.002 of the discovery's +0.3178.
L2 +0.1253 (p = 0.005, split-half positive on 100% of splits); L3 weaker at +0.0418.

**Then it deflated, three times, each by a control named in advance.**

1. **Not state tracking.** Against a baseline that already knows reply length *and* the code's
   static shape, the residual stream adds **+0.0245** — bar +0.10. Static features alone predict
   correctness at ρ = +0.2989 on L2 and only +0.1198 on L1b, which has no dispatcher spans at all.
   The route contrast is fully explained: L2 carries an extra *static* predictor and the residual
   stream encodes it.
2. **Not deep structure.** Probing directly for `n_dispatcher_spans` gives ρ = **+0.8842**, +0.3413
   beyond code size — the model *does* represent how many dispatcher sites a program has. But five
   surface counts of the source text (max token frequency, distinct/total tokens, duplicated lines,
   max line frequency) reach **+0.8346 on their own**, and the residual stream adds **+0.0647** over
   size-plus-repetition. **Under the bar.**
3. **Not a computed state.** Position × depth on L2 returns `answer_line` − `last_prompt` =
   **−0.0307**: the signal is fully present at the prompt and gains nothing from reasoning. On L1b
   the pattern is the opposite — nothing at the prompt, signal only once the answer is committed.

**What survives, stated at the precision the data supports.** The residual stream carries a signal
about dispatcher count that is **largely predictable from how repetitive the text is**. It is a
surface-statistical representation. It is the first internal correlate in this programme of a
documented behavioural effect — the dispatcher-complexity result **r = −0.196 (q = 3.1×10⁻²³)** in
Papers 2–3 — and it is **not** semantic understanding of control flow. Describing it as the latter
would be exactly the interpretability illusion this project's charter warns about.

**Two failure signatures, fully characterised.** Atom route: nothing readable until the answer is
decided; the residual stream is redundant with a token count. Relational route: static structure
readable immediately; nothing added by reasoning. **Neither involves item-level state.**

**One correction of record.** L2 in this corpus is **dispatcher indirection**, not switch-based
control-flow flattening — 0 of 70 items contain a `switch`. Earlier ledger entries in this phase
said flattening.

---

# Part IV — What it all means

## 16. The ledger

| # | experiment | question | verdict | the number that decided it |
|---|---|---|---|---|
| G0 | replication gate | does the released pair reproduce its reference here? | **PASS** | 0/101 over tolerance; median \|Δcos\| 0.0037 |
| — | code probe | are reads about code at all on-topic? | **feasible** | 44 reads; L0 names the recurrence, L2 says "state machine" |
| — | overnight | does the behavioural pattern reproduce? | **yes** | tier accuracy non-monotone: A-L1b 0.80 ≥ L0 0.70, L3 floor |
| N4 | first-error oracle | can we localize the first wrong step? | **unsolved** | D2 vs D3 agree 39%; D3 self-agrees 46%; 0/140 traces have false arithmetic |
| N5 | dense capture | 4× read density without drift? | **PASS** | 91/91 pairs, 4,653 reads, 232 re-used reads match exactly |
| N6 / HT12 | does faithfulness predict correctness? | **✗ refuted** | β = −0.0002 [−0.0051, +0.0047], p = .93 |
| N7 / HT13 | does alignment drop after the first error? | **instrument null** | judge κ = 0.049; ρ = 0.032 vs a free baseline |
| N8 / HT14 | does the answer surface earlier when right? | **✗ refuted** | own 2.8% < foreign null 3.6%; 92% censored |
| N9 | how much do reads confabulate? | **measured** | 91% name a language, 37% wrong; 79.9% wrong on JS vs 1.0% on Python |
| N10 | which capability does this malware have? | **null** | own 0.874 < foreign null 0.903; framing 2.5× |
| N10b | is this malicious at all? | **✓ supported, trimmed** | AUC 0.764 [0.717, 0.808] → **0.648 without the generic `payload`**; misses 29.5% of malware |
| N11 / B3 | do misleading names induce a false belief? | **not established** | 30.9% on the name → 3.0% off it → 0.0% past 400 chars |
| N11-c | does an internal read track stated deception? | **not established** | 3/10 vs 2/24, Fisher p = 0.138 |
| N12 / B4 | can a written-back belief recover accuracy? | **✗ refuted** | V1 0.550 = V3 0.550, p = 1.00, at every α over a 16× range |
| N13 | can we detect that the model is torn? | **null** | AUC 0.491 / 0.442 / 0.408; instability ≡ describability at r = −0.958 |
| N13 | *(free)* does answer instability predict correctness? | **✓ positive** | **AUC 0.840 held out** (the reported 0.869 used a same-sample label; see §15.4) |
| B0 | does attention steering have a transform boundary? | **not established (2026-08-26)** | family Δ +7.02 at seed 1000 → sign flip at seed 2000 |
| B1 | does the NLA read a sibling model? | **✗ refuted** | `rt_cos` 0.694 vs host 0.864; loss is directional, not scale |
| B2 | does the stated belief beat a floor? | **unmeasurable** | only ~11 of 70 stimuli have a scorable true algorithm |
| B5 | do the two levers compose? | **null** | gate passed (\|Δ\| 0.0104 < 0.02); every contrast's CI spans zero; collision check +0.10 |
| B6 | does read↔CoT agreement predict correctness? | **✗ settled judge-free** | `rt_cos` adds nothing over reply length: LRT p = 0.27, ΔCV-AUC −0.0027, n = 512 |
| **P0.1** | is the single-layer constraint hard? | **HARD** | min cos(Δ₂₀, Δ_ℓ) = **0.277** vs a frozen 0.50; coherence peaks at L13, magnitude at L20 |
| **P0.2** | is the injection channel the bottleneck? | **NOT CHANNEL-LIMITED** | V4 **+0.0833 → −0.3333** at `all_reply`; parse rate collapses to **0.100** (V1) |
| **P0.3** | is layer 20 a live site for any intervention? | **SITE LIVE** | [20,20] moves P@1 **−1.434 / −1.658** vs a 0.794 seed spread |
| **P0.4** | is the instrument at the wrong depth? | **NOT DEPTH-LIMITED** | oracle at L13 nets **0.0000** (5 recovered, 5 damaged); L13 − L20 = −0.0667 [−0.1833, +0.0500] |
| **P1b-a** | does the residual stream beat reply length at the read site? | **✗ no** | **+0.010** at `last_prompt`; +0.04–0.06 per tier against each tier's own baseline |
| **P1b-b** | can N10b's score rank a single file? | **✗ population only** | split-half reliability **0.581 / 0.613 / 0.523** vs a pre-stated 0.70 |
| **P1b-c** | is depth better for a *read*? | **UNINFORMATIVE** | tier probe at ceiling in **28/28** layers — AUC **0.9953 at layer 0** |
| **P1b-d** | does the residual stream beat length on the relational route? | **✓ replicated, then deflated** | Δ = **+0.3195** vs a frozen +0.15 bar → but adds only **+0.0245** over static shape and **+0.0647** over repetition |
| **P1b-e** | is greedy decoding reproducible per item? | **✗ no** | **0.85–0.90** on a fixed GPU; unmoved by deterministic kernels or a doubled budget. Reads **are** bit-exact (120/120) |

## 17. The single pattern behind every result

**It is about resolution.**

| what a read can do | evidence |
|---|---|
| **theme-level — reliable** | malicious vs benign AUC 0.648–0.764 · L2 reads say "state machine" · reads track the reasoning structure · the security/neutral content dissociation |
| **item-level — not present** | which capability (N10) · which algorithm (N11/B3) · which run is torn (N13) · which answer is coming (HT14) · which read is well-aligned (HT13/B6) |

Every null in the programme is an **item-level** claim. The one positive is **population-level**.
And the mechanism that kills the item-level claims is nearly always the same: a base rate high enough
that a read borrowed from *another item* does as well as the item's own.

The causal result sits on top of that. If the belief is not there at item resolution, there is nothing
item-specific to write back — which is a coherent explanation of why V1 ties V3 and why reversing the
NLA direction is indistinguishable from a random one.

**And that explanation is measured, not inferred** — the belief direction is *orthogonal* to the
clean-minus-obfuscated direction at cosine −0.001, indistinguishable from a random vector. The
geometry and its controls are in **§12.7**.

**Phase 9 tested the pattern's one apparent exception and it held.** The relational route looked
like item-level content in the residual stream, survived a pre-registered replication, and then
resolved into *how repetitive the source text is* — a theme-level property that happens, on that
one tier, to predict item-level outcomes. Restated at full strength: **theme-level content is
represented (malice, complexity, language); item-level content is not (which capability, which
algorithm, whether this run was misled, how this dispatcher resolves).** And Phase 9 added the
reason the nulls were so uniform: they were all measured at a position where a linear decoder ties
a token count.

## 18. Nineteen ways a result died here

The most transferable output of this programme is not a finding. It is this list.

**Statistical confounds**
1. **Reply length.** Wrong traces run ~1.6× longer; `rt_cos` falls with length. Killed HT12 and 61% of the cross-corpus faithfulness gap.
2. **Answer-string commonness.** Wrong answers are rarer and longer (p = 1.3e-14). Manufactured HT14's significant log-rank.
3. **Base rate at ceiling.** ~15% per read × 14 reads ⇒ P(any fires) ≈ 0.90. Killed N10's 87.4%.
4. **Class imbalance.** 78% `REMOTE_EXEC` inflated N10's permuted null to 0.53–0.61.
5. **Prompt framing.** A security preamble doubles threat vocabulary on identical *code* — and because the two arms are separate captures (0/14 positions overlap), it moves the model's state and its trace too, not just the words in the read.
6. **Lexical echo.** 30.9% on the misleading name, 0.0% past 400 chars. Killed B3's 90%-vs-19%.
7. **Circularity.** N13's new signal was the old signal inverted: r = −0.958 by class.
8. **Any large perturbation helps.** A random norm-matched direction gains +0.083. Why the bar is V3.
9. **Small-sample optimism.** 3/3 → 3/10 · an n = 34 interim → an exact tie at n = 60 · B5's composition estimates (+1.81, +2.33) with every interval spanning zero · B0's family average flipping sign at a second seed · N13's instability edge p .029 → .234 under a hardened label · and N13's own manipulation check, +0.103 at 19 pairs → **−0.011** at 54. **Six results in this programme looked right before their own controls landed.**
10. **Seed variance.** A five-transform family average flips sign between draws.
11. **Decoding nondeterminism.** Two runs of an identical command on the *same GPU* disagree on 6 of 60 items. It flipped a per-layer curve's argmax from L27 to L13 at near-identical mean AUC — **the level replicates, the shape does not**. Curve-shape claims at n = 60 need a second label draw before they are reported.
12. **Same-sample dependency.** Computing a predictor and its outcome label from the same generations inflates the estimate — 0.18 AUC for an instability measure built on five shared greedy draws. Held-out draws are the fix and are usually already on disk.
13. **Argmax over a correlated family.** Taking the best of 28 layers and comparing it to a single-number baseline: **the max of 28 layers of pure noise averages +0.17–0.18**. A weak positive at L1b evaporated (p = 0.114) once the null was taken over the statistic actually claimed rather than over a fixed layer.
14. **A baseline borrowed from the wrong condition.** A +0.1383 tier gap turned out to be a +0.1567 *baseline* gap: reply length predicts correctness far better on clean code (+0.4770) than on renamed code (+0.3203). Each condition needs its own baseline, not the neighbouring one's.
15. **A ceiling nobody checked.** L0-vs-L1b is decodable at AUC 0.9953 *at the embedding layer*, so a probe "detecting obfuscation" measures token identity. An UNINFORMATIVE branch written into the pre-registration is what made this reportable rather than embarrassing.

**Instrument and measurement failures**
16. **Population separation ≠ item ranking.** AUC 0.757 with κ 0.049. Ship an AUC as evidence of a *sort key* only after an item-level reliability check.
17. **A κ gate needs a companion discrimination check.** G3 failed because the second rater was *degenerate* (91% one label), not because two competent raters disagreed — a distinction κ alone does not surface.
18. **The null must guard the deciding statistic**, not a descriptive trajectory beside it.
19. **A frozen rule needs its estimator and its affordability checked against the data's group structure *before* it is frozen.** `(1|case)` is unusable for a predictor constant within case (singular fit / β = 0). `k ≥ 3` was undefined for 57% of items.

**Engineering failures that would have produced confidently wrong papers**
- **142 snippet names collide across HumanEval and CruxEval**; three sites keyed on the bare name would have silently dropped or overwritten half the cross-dataset coupling set.
- **V2's direction builder zipped two unequal lists** — 8 of 60 directions built. It was the one interesting cell in the sweep.
- **`statistics.mean([])` crashed at 4 sites** in the unattended scorer; a truncated capture would have died in stage 4 after hours of GPU time.
- **The artifact's `cotHTML` clamped token spans**, shifting every highlight right (old: 4/5 misplaced; new: 0/5).
- **A `\bprime\b` grader could not match `isPrime`** — caught only by end-to-end integration, not by unit tests.
- **`reset()` rewound the cursor but did not uninstall the steering spec**, so a "clean" reference pass was itself steered.
- **The answer normalizer treated `**[]` and `[]` as different answers** (36.4% of items contain a `*`), inflating an outcome variable.

**Two standing rules that came out of these**
> *When a test contradicts a fix, suspect the test.* Twice, a harness bug produced a confidently
> wrong conclusion; measuring the property that actually matters (**where does the highlight land**)
> rather than a proxy (*is the index present*) is what settled it.
>
> *A palette validator and a structure check both pass on a chart that is unreadable.* Neither looks
> at layout. Rendering to an image and looking is the only step that catches colliding labels,
> missing tick marks, and silently overridden text colour.

## 19. What survives, and is worth keeping

**Results**
- **N10b**: theme-level malice is readable at **AUC ≈ 0.65** on unambiguous vocabulary (0.764 with the generic `payload` included), holding at 0.757 against hard negatives — while missing 29.5% of malware per item.
- **Answer entropy predicts correctness at AUC 0.840** (held out against a separate greedy run; the reported 0.869 used the plurality of the same eight samples). Behavioural, free, and still better than anything internal. Its "hardening strengthens the signal" corollary does **not** survive: held out, hardening moves it 0.8400 → 0.8428.
- **The confabulation profile**: 91% name a language, 37% wrongly, with a measured **Python prior**.
- **`rt_cos` is a describability meter**, dissociable from magnitude (`act_norm` falls monotonically across the trace while `rt_cos` recovers).
- **B1's clean negative**: cross-model transfer fails **directionally**, not by scale, on an architecturally identical sibling.
- **`arith_rewriting` is where attention steering hurts** — negative at both seeds (−8.22, −5.65).
- **The behavioural replication**: non-monotone tier accuracy, L3 floor, out of the box.
- **The belief direction is orthogonal to the obfuscation direction** — cosine −0.001, against V3↔V4's +0.702. A mechanical, reusable explanation for why the causal test tied.
- **A composed-lever rig with a passing validation gate** — infrastructure the next question can reuse even though its first run was null.
- **Phase 9's negative, which is the most reusable result here:** at the site every item-level read was taken, a linear probe on the raw residual stream **ties a token count** (+0.010 over reply length; +0.04–0.06 per tier). It explains B4, B5, N11, N13 and P0.4 at once.
- **The two failure signatures**, fully characterised: atom route — nothing readable until the answer is decided; relational route — static structure readable immediately, nothing added by reasoning. Neither involves item-level state.
- **Reads are bit-exact** (120/120, max |Δ| = 0.0), so every read-side measure in this report is free of the decoding floor.

**Measurement tools built in Phase 9, and reusable beyond this project**
- **Graded k/N labels.** Ten cheap repeated-greedy draws beat one careful grade: they turned a curve whose argmax flipped between runs into curves correlating 0.79–0.94 across independent halves. No sampling, no extra instrument.
- **The reproducibility floor itself**, measured rather than assumed: 0.85–0.90 per item on a fixed GPU, robust to deterministic kernels and to a doubled generation budget.
- **Per-tier baselines, selection-free statistics, and nulls over the statistic actually claimed** — the three controls that caught four of this phase's five attractive positives.
- **`--baseline-only`, `--layer`, `--max-new-gen`, `--deterministic`** on the steering runner, and a layer-indexing gate that verifies read site == write site to 1e-3.

**Instruments** — all judge-free; **86 Python unit tests** across five modules plus three JavaScript regression tests that run the *shipped* artifact code rather than a re-implementation
`steer.py` (write hook, α = 0 byte-identical, ports across transformers major versions) ·
`steer_vectors.py` (V1–V5 ladder, leakage guard in the type signature) ·
`belief_grade.py` (closed-vocabulary, camelCase-aware, with a surface-tracking diagnostic and an
affordability check) · `deception_stats.py` (C1 floor, distance, recurrence, coupling, foreign-read
null) · `unify_corpora.py` (one schema, no double-counting, redaction at the boundary) ·
`build_page.py` (the results browser).

**Corpora**
862 C0/C1/C2 Java triples across 10 injected algorithms (max class 11.4%) · 183 quarantined malware
samples (59% near-duplicates removed) · 850 Sleeper-Agents CWE pairs · four new execution-validated
obfuscation transforms (`opaque_predicates` 680, `loop_transformation` 247, `arith_rewriting` 147,
`branch_inversion` 136) · **all nine transforms pass 360 execution-equivalence checks with 0
failures** · the unified 14,619-reading table.

## 20. Where it stands, and what to do next

**The paper as originally outlined does not have its spine.** B3 says the belief is not there to read
at item resolution; B4 says the correction does not need the autoencoder; both ran without B2, the
gate that was supposed to license them. As of 2026-08-26, the fallback paper — B0's transform
boundary — has lost its headline to a second seed.

**As of 2026-08-31 the paper has a different spine, and it is a methodological one.** Phase 9
closed the three apparatus explanations, explained the null ledger in one measured sentence
(*the read site ties a token count*), and walked the one surviving positive down three levels of
control. The defensible paper is: **a pre-registered programme in which every internal measure
that looked useful reduced to a surface statistic, with the ladder of controls as the
contribution.** Five attractive positives, four deflated by cheap controls, one replicated and
then correctly bounded. That is a more useful result than the mechanistic claim the study set out
to make, and it is fully evidenced.

**Items 3(a) and 5 below were completed in Phase 9** — `rt_cos` adds nothing over reply length
(p = 0.27), and N10b's split-half reliability is 0.52–0.61, population-only. **Item 1's
`arith_rewriting` claim still stands.** The rest are unchanged.

Ranked by value per GPU-hour:

1. **B0 at a second seed, properly.** Run CodeSteer's original four transforms at seed 2000 (~20 GPU-h)
   so the full old-vs-new contrast exists at two seeds, and re-run the seed-1000 grid *seeded* so both
   arms are controlled draws. Then treat 2 seeds as the floor, not the target — this measurement needs
   3+ before any sign-dependent claim. **The single-transform `arith_rewriting` claim is defensible now.**
2. **Re-run B5 at a second seed, and use it for the V2 lead.** The composed runner exists and its
   gate passes, but the run is one seed and every CI spans zero — on a pipeline E5b has just shown
   swings by more than the entire effect between draws. The same rig is also the only way to test the
   V2 single-word-edit lead on a corpus larger than 70 programs (the Java corpus has 862), which is
   the more valuable use of it than repeating a null. Note the model tension: B5 runs on the Coder
   sibling that B1 refuted transfer to, so a clean belief arm needs the host model and therefore an
   attention-steering baseline that does not yet exist there.
3. **The two nearly-free unrun tests.** (a) Predict correctness from reply length alone vs length plus
   faithfulness — this settles B6 judge-free and answers whether `rt_cos` adds anything at all.
   (b) One generation run separates whether the Python prior lives in the **verbalizer** or in the
   **subject's residual stream**.
4. **The unified-table sweep.** `rt_cos` and `act_norm` by read class × corpus with length as a
   covariate, over all 14,619 rows at once, instead of three analyses that cannot be pooled.
5. **Split-half the N10b score per item.** AUC 0.757 separates *populations*; whether it can rank a
   single sample is untested — the same distinction that sank the LLM judge, pointed the other way.
   This matters more now that the score is known to lean on one generic term.
6. **Test the paper's own steering protocol.** B4 tested a cheaper variant; **editing the model's own
   verbalized read** — the protocol the source work actually proposes — has never been run.
7. **Cheap N13 follow-ups.** Sweep layers (no training needed for `act_norm` or instability) — the
   cheapest test of the "not linearly localized at 20" escape route. And induce torn-ness *properly*:
   Gate 1b shows adversarial renaming does not do it, so a real test needs genuinely ambiguous
   programs with two defensible outputs. That is a stimulus-design problem, not a measurement one.
8. **Make the malware labels trustworthy** (~100 hand-labelled samples, two labellers, ≥85%
   agreement) before any per-capability number is quoted again, and broaden the malicious sources
   beyond one detection tool.
9. **Parked, with the design already done:** N1–N3 (HT9–HT11) — NLA↔CoT agreement, frozen-judge
   reranking, intervention validation; gates G1–G3 specified.

> **The standing baseline.** Answer entropy predicts correctness at **AUC 0.840** held out,
> behaviourally and for free — and reply length alone reaches CV-AUC 0.693. Any future claim that
> an *internal* measurement is useful has to clear them. As of Phase 9 none does: the residual
> stream at the read site adds **+0.010** over a token count.

**Housekeeping — status as of 2026-08-31.** *Closed:* both log indexes are current (64 entries);
**B5's missing ledger entry is written** (`log/nla-harness/2026-08-26_b5-composition.md`, dated to
its run) — though the underlying defect stands, since the run persisted **no per-case rows**
(`results/runs/` gained nothing, and the stage-4 aggregate regenerated the pre-existing RQ1 grid),
so its six cells still cannot be re-analysed from disk; and `CLAUDE_SCRATCHPAD.md` is current.

*Still open:* the 2026-08-17 programme report's "currently running" and "where this leaves the
paper" sections remain superseded by the seed result and carry no pointer to it. Three wording
errors survive in the 2026-08-14 reports — **"identical activations"** (false: the two framing arms
are separate captures, 0/14 read positions overlap; say "identical *code*, two prompts"),
**"64 tests"** (it is 86 Python plus 3 JS), and **"twice in one month"** for interim reversals (it
is six). `nla/continuation/00_STATE.md` predates Phase 9 entirely and still quotes N13 at 0.869.

## 21. Caveats every headline number inherits

Collected in one place because they live scattered across a corpus doc, a plan file and five
reports, and because several of them bound results quoted above.

**The instrument**
- **FVE ≈ 0.75.** Each read loses roughly **25% of the vector's variance** by construction. That is
  the ceiling on everything, and no analysis here can see below it.
- **One model, one layer, one temperature pair.** Everything is `Qwen2.5-7B-Instruct` at layer 20 —
  the only NLA-instrumented coordinates that exist. A property that is real but *not linearly
  localized at layer 20* was never going to show up in a per-position scalar.
- **AR-space cosine is compressed** (0.971–0.994). Every N13 contrast was therefore also run on
  lexical overlap, which has ~15× the range and no learned component; the two agreed everywhere.
- **DRM gloss anchors are crude** — the absolute sign is untrustworthy, only relative contrasts
  (tier vs tier, item vs item) mean anything. Contrastive-activation anchors are the known upgrade.
- **Sub-word digit positions are noise** (`state_1395` → `1`,`3`,`9`,`5` reads as numerology) and are
  excluded from every quantitative set.
- **Reads cannot be batched** without breaking AV determinism (6.3× faster, 15% byte-identical), so
  read budget caps every design.

**The malware corpus (N10 / N10b)**
- **Selection bias, acknowledged upstream.** Most DataDog samples were caught by a *single* tool
  (GuardDog), so the collection reflects what that tool catches. It is **not a representative sample**
  of supply-chain malware and no claim should be phrased as if it were.
- **The capability labels are an unvalidated heuristic.** Nobody checked them against human judgment.
  **`UNASSIGNED` is 51 of 183 — 28% of the corpus is effectively unlabelled** — and it never means
  benign. 78% of the *labelled* files are `REMOTE_EXEC`. Hand-validation (~100 samples, two labellers,
  ≥85% agreement) **has not been done**, and every per-category number inherits the errors.
- **The obfuscation profile is a survivorship number.** 60% contain a URL, 8% a base64 blob ≥40
  chars, **0% hex escapes**, 9% any non-ASCII — mild, and weaker than the assumed confound. But the
  ≤20 KB / ≤6,000-char filters are precisely what select plain code *out of* an obfuscated
  population, so this describes what survived the filters, not the dataset.
- **"Benign" currently means popular library code.** A different benign population could move N10b.

**The measurement itself (added Phase 9)**
- **Graded generations do not reproduce per item.** 0.85–0.90 agreement between two runs of an
  identical command on a **fixed GPU**, unmoved by deterministic kernels or by doubling the
  generation budget, cause unidentified. Every paired per-item table in this report inherits it:
  B4's McNemar gate, B5's cells, P0.3's P@1 tables, P0.4's primary. **Reads do not** — they are
  bit-exact, so `rt_cos`, `act_norm`, the layer curves and the unified table are unaffected.
- **Absolute accuracies are biased low.** At the banked 1,100-token budget the parse rate is
  **0.8833**, so ~12% of items were scored wrong for not finishing rather than for being wrong.
  Raising the budget to 2,048 takes it to 1.0000 at unchanged accuracy. Paired contrasts are
  unaffected (parse failures count as wrong in both arms); absolute numbers are not.
- **Curve *shapes* at n = 60 are not identifiable.** Level replicates across label draws; argmax
  does not (0 of 20 split-half agreements at `last_prompt`). Report levels, not peaks.

**The stimulus sets**
- **70 programs is a hard cap** on the steering corpus (`dataset_a` + `dataset_b` with usable L0/L1b
  pairs and ground truth). No amount of GPU time raises it — only a corpus change does.
- **50 Dataset-A Python rows are read-only** — no calls or ground truth, by construction upstream.
- **N13's instability 2×2 rests on 14 pairs**, and its floor class is confounded with tier and token
  shape, which is why the interpretive weight sits on the shape-matched decoy-vs-meaningless pair.

## 22. Map — where everything lives

| what | where |
|---|---|
| **This report** | `transcoders/reports/2026-08-26_nla-master/REPORT.md` |
| Ledger entries (64) | `transcoders/log/nla-harness/YYYY-MM-DD_*.md` |
| Thread summary + hypothesis ledger | `transcoders/log/nla-harness/README.md` · `transcoders/docs/CHECKLIST.md` |
| Prior reports | `transcoders/reports/2026-08-{05,14,14,15,17}_*/REPORT.md` |
| Worked examples + decompositions | `transcoders/reports/2026-08-26_full-results/RESULTS.md` |
| B5 composed run | `transcoders/data/nla/b5/` · `nla/scripts/b5_autopilot.sh` · `nla/src/b5_gate.py` |
| Design of record (frozen) | `transcoders/docs/PLAN_believe_the_lie.md` |
| Instrument code | `transcoders/nla/src/{steer,steer_vectors,belief_grade,deception_stats,unify_corpora,steer_stats}.py` |
| Frozen checkpoints (~25 GB, not committed) | `transcoders/nla/data/checkpoints/{av,ar}/` |
| Configs | `transcoders/nla/configs/{b1_transfer_gate,b4_steering}.yaml` |
| Unified reading table | `transcoders/data/nla/unified/{readings,cases}.jsonl` |
| Malware corpus protocol | `transcoders/docs/N10_MALWARE_CORPUS.md` |
| B0 / attention-steering side | `allocation_replication/docs/B0_RESULTS_2026-08-16.md` · `allocation_replication/LOG.md` |
| Interactive results browser (9,511 readings) | `transcoders/reports/2026-08-15_n13-torn/results_browser.html` |
| Malware read browser (redaction-verified) | `transcoders/reports/2026-08-15_n13-torn/malware_browser.html` |
| Galleries | `data/nla/overnight/2026-08-04/GALLERY.md` · `data/nla/examples/EXAMPLES.md` · `data/nla/probe/probe_reads.json` |
| Tests | `nla/tests/` — 86 Python tests + 3 JS regression tests against the shipped page |
| **Phase 9 — triage** | `nla/src/{layer_rotation,p03_score,p0_summary,p0_verdict,p04_gate,p04_score}.py` · `data/nla/p0/` |
| **Phase 9 — reproducibility floor** | `nla/src/{p04_repro_score,p04_read_repro,p04_read_repro_score}.py` · `data/nla/p0/p04/{repro,readrepro}/` |
| **Phase 9 — read side** | `nla/src/{p1b_read_probe,p1b_position_depth,p1b_graded_labels,p1b_consensus_labels,p1b_readout_controls,p1b_answer_deviation,b6_length_vs_faithfulness,n10b_split_half}.py` |
| **Phase 9 — the ladder** | `nla/src/{p1b_ladder,p1b_ladder_score,p1b_ladder_null,p1b_ladder_replication,p1b_l2_mechanism,p1b_span_probe,p1b_span_positions,p1b_behavioural}.py` · `data/nla/p0/p1b/` |
| Cluster environment (juno) | `nla/scripts/juno_env.sh` · `nla/scripts/juno_build_envs.sh` |

---

*Compiled 2026-08-26 from the primary ledger; **Phase 9 (§15) and the Part IV revisions added
2026-08-31**. Every number above is quoted from a dated log entry or report; nothing here is
re-derived. Where two sources disagree, the later dated entry wins — which is why §14's seed result
overrides the B0 headline in the 2026-08-17 report, and why §15's held-out 0.840 overrides the
0.869 quoted for N13 in the 2026-08-15 normalization entry.*
