# Believe the Lie — the programme, and what each question returned

*Status report · 2026-08-17 · written to be read cold, no prior familiarity assumed.*
*Subject: `Qwen2.5-7B-Instruct`, layer-20 residual. Instrument: `kitft/nla-qwen2.5-7b-L20-{av,ar}`,
frozen released checkpoints. Seed 20260724 throughout.*

> **⚠ Superseded in part — appended 2026-08-31, not part of the 2026-08-17 status.**
> Two claims below no longer stand as written.
> - **B0's family contrast (+7.02 [+2.00, +10.09]) did not replicate.** At a second seed (E5b,
>   2026-08-26) the five-transform family average **flips sign** to +4.46 [+0.85, +7.09], and four
>   of five transforms reverse. What survives is the single transform `arith_rewriting`, negative
>   at both seeds (−8.22, −5.65). The "currently being extended from one model to three"
>   framing, and the FSE recommendation built on the family contrast, are withdrawn pending a
>   third seed. See `log/nla-harness/2026-08-26_e5b-seed-instability.md`.
> - **"Where this leaves the paper" is out of date.** A pre-registered triage (2026-08-28 to
>   08-31) established that the causal failure is not the injection channel, not a dead site and
>   not the wrong depth, and the read-side programme showed that at the site every item-level read
>   was taken a linear probe on the residual stream ties a token count. The current statement of
>   record is §15 and Part IV of `reports/2026-08-26_nla-master/REPORT.md`.

| | |
|---|---|
| research questions (B0–B6) | **7** |
| answered | **4** |
| clear positives | **1** |
| verbalized state reads | **20,250** |
| graded generations | **14,592** |
| injection failures | **0** |

---

## 1. The idea, from scratch

**Obfuscation** is any change to a program that keeps it doing exactly the same thing while making
it harder to read — renaming `fibfib` to `smoothArea`, flattening control flow into a dispatcher
loop, rewriting arithmetic into equivalent but opaque forms. Language models get worse at
predicting what such programs output. That much was already established by prior work.

An earlier paper, **CodeSteer**, proposed a fix: reweight the model's *attention* so it looks
harder at the lines that determine the answer. That is a **positional** lever — it redistributes
focus across words already present in the prompt. By its own authors' reasoning, it cannot supply
a fact that is not there.

This paper's idea was to add a **semantic** lever. A **Natural Language Autoencoder (NLA)** is a
pair of trained models:

- the **verbalizer** takes one activation — the model's internal state at a single word, 3,584
  numbers — and writes a sentence describing it;
- the **reconstructor** maps text back to a vector.

Together they let you read the model's internal state in English and, in principle, write a
corrected belief back in.

**The paper's spine:** read the belief, show it predicts failure, correct it causally, and report
honestly how much of that correction actually needs the autoencoder.

### The bar that decides everything: V3

Claiming the autoencoder does causal work requires more than beating "no steering" — any large
enough perturbation changes outputs. So the pre-registered comparison was against a **contrastive
difference vector (V3)**: average the internal-state difference between clean and obfuscated
versions of *other* programs, and push along it. It needs no verbalizer, no reconstructor, no
English. **If it matches the autoencoder, the autoencoder is buying interpretability but not
capability.**

---

## 2. How to read the verdicts

Each question was written down with its decision rule frozen *before* its experiment ran. That is
what lets a negative mean something rather than being a failure to find.

| verdict | meaning |
|---|---|
| **supported** | The pre-registered criterion was met, and the effect cleared its own null. |
| **not established** | A real effect appeared, but a matched control reproduced it. It is not what it looked like. |
| **refuted** | The criterion was missed, with a null informative enough to exclude the hoped-for effect size. |
| **not run** | Never executed, or found unmeasurable before it could be. |

---

## 3. The ledger — seven questions, matched to what ran

### B0 · Does attention steering really fail where the original paper said it did? — **SUPPORTED**

**What the outline asked.** CodeSteer reports its method failing on two transforms (arithmetic
rewriting −1.68%, branch inversion −8.96%). Those rows are 10 programs each, and branch
inversion's headline is a **−0.24-point** recovery. The outline called that negative result *not
established* and made it a gate: you cannot claim to fix what was never shown broken.

**What ran.** 90 units, 12,432 generations, nine transforms, one model
(`Qwen2.5-Coder-7B-Instruct` — CodeSteer's own primary model), 0 failures, ~9.8 h on 2 GPUs.

| family | steer − obf (P@1) | cells |
|---|---|---|
| CodeSteer's own four transforms | **+3.89** | 638 |
| The four they never tested | **−3.14** | 273 |
| **difference** | **+7.02** | 95% CI **[+2.00, +10.09]** |

Per transform, on HumanEval, with the denominator alongside:

| technique | orig | obf | steer | deficit | **steer − obf** | 95% CI | snippets |
|---|---|---|---|---|---|---|---|
| identifier_renaming | 77.56 | 66.48 | 69.22 | +11.09 | **+2.75** | [−0.90, +6.45] | 164 |
| dead_code_injection | 77.56 | 72.69 | 73.94 | +4.87 | +1.24 | [−2.38, +4.69] | 164 |
| control_flow_flattening | 77.15 | 72.97 | 73.20 | +4.19 | +0.23 | [−3.06, +3.39] | 146 |
| call_indirection | 77.56 | 71.24 | 72.90 | +6.32 | +1.66 | [−2.04, +5.31] | 164 |
| **arith_rewriting** | 78.77 | 71.58 | 63.36 | +7.19 | **−8.22** | [−16.77, +0.28] | 44 |
| **branch_inversion** | 69.69 | 71.24 | 69.03 | **−1.55** | −2.21 | [−9.58, +5.40] | 37 |
| **opaque_predicates** | 75.90 | 72.92 | 69.38 | +2.98 | **−3.54** | [−7.49, +0.64] | 133 |
| **loop_transformation** | 77.55 | 73.62 | 72.74 | +3.94 | −0.87 | [−5.87, +4.02] | 59 |
| `adversarial_rename` | 77.56 | 62.28 | 61.71 | **+15.28** | −0.57 | [−3.78, +2.56] | 164 |

**No individual technique's CI excludes zero.** Nine tests on 37–164 snippets are underpowered;
the finding is the *family contrast*, which is powered. The per-technique rows are reported for
shape, not significance.

Three things fall out:

1. **CodeSteer's negative is confirmed in direction and understated in size** — arithmetic
   rewriting is the worst cell at −8.22 points; steering does not merely fail to recover, it
   destroys accuracy the unsteered run had.
2. **`branch_inversion` has a *negative* deficit** (obf 71.24 > orig 69.69). Any Restoration Ratio
   there divides by a negative number, which is why the table script prints **+142.9%** for a cell
   where steering *hurt* by 2.21 points. Absolute Δ with its denominator is the only honest
   statistic in that regime; CruxEval R@1 is not reported at all for the same reason.
3. **On the belief-steering corpus the positional lever is inert.** `adversarial_rename` has the
   largest deficit in the table (+15.28 — the most room to recover of anything) and steering moves
   it by −0.57. That is exactly what CodeSteer predicts about itself, and it gives the paper's
   central comparison a shared-stimulus anchor.

**Status:** currently being extended from one model to three (~50 h, four GPUs).

---

### B1 · Can the released autoencoder read a different model's internal state? — **REFUTED**

**What the outline asked.** The NLA was trained on one specific model. A sibling
(`Qwen2.5-Coder-7B-Instruct`) shares its architecture, its `tokenizer.json` blob, its chat
template, and every dimension — so the checkpoint *loads*. Whether it *reads* was untested, and
the published work is silent on cross-model transfer. This decided the subject model for the whole
paper.

**What ran.** 164 Java programs, 820 reads per model, both read at **identical positions** (the
shared tokenizer makes the pairing exact — the only variable is the weights).

| check | verdict | detail |
|---|---|---|
| median round-trip cosine ≥ 0.70 | **FAIL** | **0.694** (host 0.864) |
| within 0.05 of the host | **FAIL** | gap **+0.170** |
| own vs foreign activation | PASS | Δ +0.355 [+0.344, +0.366] |
| activation-norm band | PASS | 154.0 (host 111.7) |
| mostly-CJK reads | PASS | 0 / 820 |

**It is not a scale artifact, and that is the interesting part.** The sibling's activations are
38% larger in magnitude, which looks like the obvious culprit. But injection rescales every vector
to a fixed L2 norm before the verbalizer sees it — **magnitude is discarded entirely**. So the
loss is *directional*: the sibling's layer-20 directions have diverged far enough that a frozen
verbalizer reads them measurably worse, despite identical tokenizer, template and architecture.
This also makes the plan's "re-derive the injection scale" remedy a provable no-op.

**Degraded, not broken:** own-vs-foreign is +0.355, so the reads still carry substantial
item-specific information — they are simply less faithful than on the host.

**Consequence:** the paper's NLA subject stays `Qwen2.5-7B-Instruct`. No downstream re-work — B3,
B4 and N13 are all already on that model.

---

### B2 · Does the model's stated belief beat a confabulation floor? — **UNMEASURABLE**

**What the outline asked.** This was the **gate** for B3 and B4. Reads are known to be reliable
about themes and unreliable about specifics — 37% of them name the wrong programming language.
"Which algorithm is this" is a *specific* claim, so B2 had to pass before B3 and B4 were worth
running.

**What ran.** Nothing. Scoring it needs a ground-truth "true algorithm" per program, and only
about **11 of 70** stimuli have one the closed vocabulary can score — these are programs like
"find the closest two elements", not textbook named algorithms. Widening the vocabulary is not a
fix: the diagnostic built for exactly this purpose caught a label that fired *more* as code got
harder to read, i.e. one that was describing the obfuscation rather than the program.

> **B3 and B4 ran anyway, without their gate, and both then failed.** That ordering is a genuine
> weakness in the programme and is recorded as one rather than explained away.

---

### B3 · Do deliberately misleading names induce a false internal belief? — **NOT ESTABLISHED**

**What the outline asked.** Rename a program's identifiers to those of a *different, wrong*
algorithm — call a duplicate-finder `bubbleSort`. Because we choose the wrong algorithm, ground
truth is exact and scoring needs no LLM judge. Then ask whether the internal state adopts that
false belief, against a neutral-rename floor.

**What ran.** 300 captures / 6,171 reads, then a targeted follow-up of 113 / 2,319.

The raw effect was enormous: 90% of programs versus a 19% floor. Then you ask *where* those reads
are:

| where the read was taken | original | neutral rename (floor) | adversarial rename |
|---|---|---|---|
| **on the misleading name itself** | 0.9% | 0.8% | **30.9%** |
| elsewhere in the code | 1.6% | 1.8% | 3.0% |
| in the model's reasoning | 1.4% | 2.2% | 3.6% |
| at the final answer | 0.3% | 0.4% | 3.0% |

By distance from the misleading name: **30.9%** on it → 3.7% at 1–50 characters → 2.5% at 50–150 →
**0.0%** past 400.

**A belief propagates through a representation; an echo is local. This is an echo** — the reads are
reading the word, not a belief about the program. Stripping the on-name reads and requiring the
claim to recur at two separated positions gives 9.0% against a 2.0% floor — nominally significant
at p = 0.039, but the matched null for *that same statistic* spans [0.02, 0.10] and 0.090 sits
inside it.

A follow-up on the one encouraging cell (internal reads tracking behavioural deception, 3-of-3)
returned **3-of-10**, Fisher p = 0.138.

---

### B4 · Does writing a corrected belief back in recover accuracy — and does it need the autoencoder? — **REFUTED**

**What the outline asked.** The causal test, and the paper's centre of gravity. Inject an
NLA-derived direction into the model's internal state and measure recovery on misleadingly-named
code — **against V3**, the difference vector that uses no autoencoder at all.

**What ran.** 60 programs. First at one injection strength (420 generations), then a sweep across a
16× range of strengths (2,160 generations, 0 errors).

The gate, at the pre-registered strength:

| | V1 (NLA) | V3 (no NLA) | Δ | discordant | McNemar p |
|---|---|---|---|---|---|
| α = 1.0 | **0.550** | **0.550** | **0.000** | 7 v 7 | **1.00** |

And at every other strength:

| α | V1 | V3 | Δ | p | q (BH) |
|---|---|---|---|---|---|
| 0.25 | 0.583 | 0.567 | +0.017 | 1.000 | 1.00 |
| 0.5 | 0.517 | 0.550 | −0.033 | 0.727 | 1.00 |
| 1.0 | 0.550 | 0.550 | 0.000 | 1.000 | *(primary)* |
| 2.0 | 0.633 | 0.617 | +0.017 | 1.000 | 1.00 |
| 4.0 | 0.550 | 0.550 | 0.000 | 1.000 | 1.00 |

Two details make this more than an underpowered shrug:

- **Reversing the NLA direction performs identically to a random one.** If the direction carried
  signal, reversing it should hurt. It does not — so it has no consistent *sign*, which is not a
  magnitude problem, and no injection strength fixes it.
- **Adding one sentence to the prompt beat every steering condition** (+0.100), including the
  oracle that gets to see the clean program.

**A confound the sweep exposed:** a random, norm-matched direction gains +0.083 by the strongest
setting. Any sufficiently large perturbation nudges accuracy on this task — which is precisely why
the bar is V3 and not zero.

---

### B5 · Do the positional and semantic levers compose? — **NOT RUN**

**What the outline asked.** Run attention steering and belief steering together and test whether
the channels add, interfere, or are simply independent. This is a *different* question from B4's:
it asks about interaction, not about which lever is better.

**What ran.** Nothing yet. It needs one process driving both levers, which currently live in
different software environments. One design hazard is already identified: attention steering
occupies layers 20–27 and the autoencoder lives at layer 20, so a naive "both on" cell would
confound **composition** with **collision**. A disjoint-layer control is specified.

---

### B6 · Does agreement between the reads and the model's reasoning predict correctness? — **NOT ADJUDICATED**

**What the outline asked.** If the internal reads and the model's written reasoning agree, is the
answer more likely to be right — better than simply measuring how long the reasoning is?

**What ran.** A judge model was built to score agreement. It separated real from shuffled pairs at
**AUC 0.757** — and could not rank a single item (**κ = 0.049** against a second rater, ρ = 0.032
against a free baseline).

> **Separating populations is not per-item reliability.** The hypothesis was declared
> *unadjudicated* rather than answered, and the judge was retired. A judge-free version — predict
> correctness from reply length alone versus length plus faithfulness — is nearly free and has not
> been run.

---

## 4. Three experiments outside the outline

Two came from the predecessor plan and one was added later. They matter because the programme's
only clear positive is among them.

| experiment | question | result | verdict |
|---|---|---|---|
| **N10** | Does a read say *which* capability a piece of malware has? | 87.4% of samples got a naming read — but reads borrowed from *other* samples scored 90.3%. | null |
| **N10b** | Does a read tell malicious from benign at all? | **AUC 0.764** [0.717, 0.808], holding at 0.757 on hard negatives, length neutralised at 0.477. | **supported** |
| **N13** | Can any internal signal tell "torn" from "confidently wrong"? | All three candidate signals at or below chance on the matched contrast (AUC 0.49 / 0.44 / 0.41). | null |

**The pattern across everything is consistent, and it is about resolution.** Reads carry
*theme-level* content — "this is doing something hostile", "this mentions sorting" — reliably
enough to separate populations. They do not carry *item-level* content: which capability this
sample has, which algorithm this program implements, whether this model has been misled. Every
null in the programme is an item-level claim; the one positive is population-level.

---

## 5. Where this leaves the paper

The paper as outlined **does not have its spine**. B3 says the belief is not there to read, B4 says
the correction does not need the autoencoder, and both ran without the gate that was supposed to
license them.

What *is* there is a different and more defensible paper: **a powered boundary condition on an
existing technique.** B0 shows attention steering helps on exactly the transforms its authors
tested and hurts on the ones they did not, and that result is currently being extended from one
model to three. Alongside it sits a clean negative about the autoencoder's cross-model transfer,
and a set of judge-free measurement tools plus the nulls that killed four plausible-looking
effects.

### Currently running

- **B0 on two further models** — four GPUs, ~50 h, testing whether the transform boundary
  generalizes. A two-model result lands at roughly the halfway point.

### The one lead still open

In the B4 sweep, a **minimal single-word edit** — changing one term in the description rather than
swapping the whole thing — was the only condition showing a dose-response (Δacc 0.000 → 0.033 →
0.117 → 0.117 → **0.183**), and at the strongest setting it separated from the no-NLA bar:
**0.683 vs 0.550**, discordant 12 v 4, **p = 0.077**.

It is exploratory rather than pre-registered, it does not clear its own multiplicity correction
(q ≈ 0.385), and it is one cell out of twenty-five. It also **cannot be confirmed on this corpus**,
which is hard-capped at 70 programs. That is the strongest argument for building B5, whose Java
corpus has 862.

> **Two results in this programme reversed at full sample size** — a coupling effect that was
> 3-of-3 and became 3-of-10, and a steering result that tracked its ceiling at n = 34 and tied
> exactly at n = 60. Both looked like the pre-registered success partway through. It is the reason
> nothing above is quoted from a partial run, and the reason the open lead is labelled a lead.

---

*Plan v0.3 "Do Models Believe the Lie?" · RQs B0–B6 · full ledger entries and pre-registrations in
`transcoders/log/nla-harness/`; B0 detail in
`allocation_replication/docs/B0_RESULTS_2026-08-16.md`.*
