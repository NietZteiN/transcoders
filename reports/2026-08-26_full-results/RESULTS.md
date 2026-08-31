# Believe the Lie — full results, with worked examples

*2026-08-26. Every number below is recomputed from banked run files; every example is a real
read or a real generation, quoted verbatim.*

**Subject** `Qwen2.5-7B-Instruct`, layer-20 residual (d=3584) · **Instrument**
`kitft/nla-qwen2.5-7b-L20-{av,ar}`, frozen · **Seed** 20260724

---

## 0. What was being tested

**Obfuscation** rewrites a program so it computes the same thing while being harder to read.
**CodeSteer** proposed fixing the resulting accuracy loss by reweighting *attention* — a
*positional* lever that redistributes focus over tokens already in the prompt.

This programme asked whether a *semantic* lever works instead. A **Natural Language Autoencoder**
is a verbalizer (activation → English) plus a reconstructor (English → activation). Together they
let you read the model's internal state and write a corrected belief back in.

**The spine:** read the belief → show it predicts failure → correct it causally → report how much
of the correction needs the autoencoder.

**The control that decides everything (V3).** Beating "no steering" proves nothing, since any
large perturbation changes outputs. The pre-registered bar is a *contrastive difference vector*:
`mean(h_clean − h_obfuscated)` over **other** programs. No verbalizer, no English. If it matches
the autoencoder, the autoencoder buys interpretability but not capability.

| RQ | question | verdict |
|---|---|---|
| B0 | does attention steering really fail where CodeSteer said? | **did not replicate** |
| B1 | does the autoencoder read a different model? | refuted *(and expected — see §7)* |
| B2 | does belief content beat a confabulation floor? | unmeasurable |
| B3 | do misleading names induce a false belief? | **not established (echo)** |
| B4 | does belief steering beat a no-NLA vector? | **refuted** |
| B5 | do the two levers compose? | **null** |
| B6 | does read↔reasoning agreement predict correctness? | not adjudicated |

---

## 1. B3 — the effect is a lexical echo, not a belief

### Design
Three versions of each Java program, differing **only** in identifiers:

- **C0** original
- **C1** neutral opaque rename (`m_O0lI1`) — the confabulation floor
- **C2** adversarial rename to a plausible **wrong** algorithm

C2's ground truth is exact *by construction*: we pick the injected algorithm from the grader's own
closed vocabulary, so scoring is a regex against a field the verbalizer never sees. 100 programs ×
3 conditions = **300 captures, 6,171 reads**, `rt_cos` median 0.870, 0/6,171 CJK.

### The headline, and where it goes

Read-level hit rate (a read names the injected wrong algorithm), by locus:

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

Item-level the raw contrast is 90% vs a 19% floor (McNemar p < 1e-15). Strip the on-identifier
reads and require the claim to recur at ≥2 separated positions: C2 **9.0%** vs 2.0% floor,
p = 0.039 — but the foreign-read null for *that same statistic* spans **[0.02, 0.10]** and 0.090
sits inside it.

### Worked example — `Java_104`

The program filters numbers whose digits are all odd, then sorts them. The adversarial rename
calls it **`binomialCoefficient`**.

**The model is not fooled.** Its stated answer under C2:

> "Filter numbers with all odd digits and sort the result"

versus under C0 (clean names): *"Filter and sort numbers with only odd digits"*. Same answer.

But the read taken **on the misleading token** (`Coefficient`, distance 0 chars, `rt_cos` 0.927):

> *"Code snippet format with Java/Python function definition pattern suggests a student or
> algorithmic context showing iterative **factorial or combinatorial** computation. The phrase
> `public static int binomialCoefficient` strongly implies a method name…"*

→ grader fires: `binomial_coefficient`, `factorial`. **The injected false algorithm.**

Two reads from the same program, 41 and 43 characters away:

> *(token `import`)* "Formal code snippet format with a question prompt ('Write a Java program to
> find maximum of three numbers')…" → fires `max_min` — a **confabulation**, unrelated to either
> the true or injected algorithm.

> *(token `.asList`)* "Java code demonstrating `ArrayList` sorting and filtering using Java 8
> Stream API…" → fires `generic_sort` — which is what the code **actually does**.

**A belief propagates through a representation; an echo is local.** The read on the name is
reading the *name*. Move 41 characters away and the injected algorithm is gone, while the model's
own answer was correct the whole time.

### The one promising cell, followed properly
Items where the model *stated* the wrong algorithm had a matching internal read 3/3 in the first
run. With a targeted follow-up (19 deceived + 19 matched controls, 2,319 reads) it became
**3/10 = 30.0%** vs 2/24 = 8.3%, Fisher **p = 0.138**. And in that deception-enriched set, reads
away from the identifier were **0.054 / 0.063 / 0.063** across conditions — no elevation at all —
while the identifier locus sat at 0.378.

---

## 2. B4 — belief steering ties a vector that uses no autoencoder

### Design
60 L0/L1b program pairs with execution ground truth. Inject `h ← h + α‖h‖Δ̂` at layer 20 and
measure balanced Δaccuracy over **all** items (not flip rate — a vector that makes the model shout
"sorting" at everything scores brilliantly on flips while destroying correct answers).

Conditions: **V1** NLA gloss edit · **V2** minimal single-word edit · **V3** contrastive task
vector (*no NLA*) · **V4** item oracle (uses the clean program — not deployable) · random ·
foreign-item · antipodal · prompt-only.

### The gate, and the full sweep
2,160 generations, 0 errors, five injection strengths.

| α | V1 | V3 | Δ | McNemar p | q (BH) |
|---|---|---|---|---|---|
| 0.25 | 0.583 | 0.567 | +0.017 | 1.000 | 1.00 |
| 0.5 | 0.517 | 0.550 | −0.033 | 0.727 | 1.00 |
| **1.0 (pre-registered)** | **0.550** | **0.550** | **0.000** | **1.000** | *primary* |
| 2.0 | 0.633 | 0.617 | +0.017 | 1.000 | 1.00 |
| 4.0 | 0.550 | 0.550 | 0.000 | 1.000 | 1.00 |

At the primary α the discordance is **7 vs 7** — a perfectly symmetric tie.

### Two things that make this more than "underpowered"

**1. Reversing the direction behaves like a random one.** Antipodal +0.033 vs random +0.033. If
V1 carried signal, reversing it should hurt. It has no consistent *sign*, which no injection
strength can fix.

**2. Everything drifts up with α, including noise.**

| α | mean Δacc over all 7 conditions | R_random | V3 |
|---|---|---|---|
| 0.25 | +0.045 | +0.017 | +0.067 |
| 1.0 | +0.050 | +0.033 | +0.050 |
| 4.0 | +0.083 | **+0.083** | +0.050 |

A **random, norm-matched** direction gains as much as most real ones by α=4. This is exactly why
the bar is V3 and not zero.

**3. Prompt-only beat every steering condition** (+0.100), including the oracle.

### Worked examples — the discordant items the gate counts

These are the 14 items where V1 and V3 disagree at α=1:

| item | truth | V1 | V3 | V4 oracle | **random** |
|---|---|---|---|---|---|
| `JavaScript/58` | `[2, 3, 4]` | `[2, 3]` ✗ | `[2, 3, 4]` ✓ | ✓ | **✓** |
| `leetcode/check-if-digits…` | `False` | `False` ✓ | `True` ✗ | ✗ | **✓** |
| `cruxeval-x-javascript/110` | `1` | `1` ✓ | `2` ✗ | `7` ✗ | **✓** |

Read the last column. **The random direction is right on all three.** On the first, V1 *broke* an
answer the baseline had correct while random preserved it. On the third, the *oracle* — which is
allowed to see the clean program — answered `7` against a truth of `1`.

### The one open lead
**V2** (change exactly one word of the description rather than swapping the whole gloss) is the
only condition with a monotone dose-response: Δacc **0.000 → 0.033 → 0.117 → 0.117 → 0.183**. At
α=4 it separates from V3: **0.683 vs 0.550**, discordant 12 v 4, **p = 0.077**.

Not banked: exploratory rather than pre-registered, q ≈ 0.385 under the same BH correction, and
one cell of 25. **n is hard-capped at 70 pairs** by the stimulus set, so it cannot be confirmed
here at all.

---

## 3. B5 — the two levers do not measurably compose

### Design
Six conditions on 164 Java programs each (identical stimuli, 1,930 cases per cell). Before any
cell ran, a **validation gate**: with the belief channel at α=0 (a unit-tested no-op) and attention
steering ON, the patched runner must reproduce the banked attention-only number.

**Gate: 62.746 vs banked 61.710 on identical 1,930 cases — |Δ| = 0.0104 against a 0.02
tolerance. PASSED.** So the integration does not change the measurement.

### Result

| contrast | estimate | 95% CI (cluster bootstrap over programs) |
|---|---|---|
| attention only | −1.30 | [−6.31, +3.61] |
| belief only | −2.12 | [−6.76, +2.39] |
| both levers | +0.52 | [−3.37, +4.38] |
| **both − attention** | +1.81 | [−2.61, +6.16] |
| **both − random-direction control** | +2.33 | [−2.17, +6.74] |
| **disjoint − both** *(collision check)* | +0.10 | [−3.93, +4.26] |

Raw cell accuracies: none 64.04 · attn 62.75 · nla 61.92 · both 64.56 · rand 62.23 ·
disjoint 64.66.

**Not one interval excludes zero.** The point estimates read like a composition result — the pair
beats either lever alone, and beats a random direction by 2.33 — and every one of them dissolves
against its interval.

**The one control that did its job:** attention steering occupies layers 20–27 and the autoencoder
lives at layer 20, so "both on" could be a *collision* rather than composition. Moving attention to
layers 21–27 changed the result by **+0.10**. Not a collision — but also not an effect.

---

## 4. B0 + E5b — the boundary result does not survive a second seed

### Design
CodeSteer reports its method failing on two transforms, from rows of 10 programs each (one
"failure" is a −0.24-point recovery). B0 re-measures on corpora ~14× larger: 90 units, 12,432
generations, nine transforms, 0 failures.

### Seed 1000 — what looked like the programme's one positive

| transform | orig | obf | steer | deficit | **steer−obf** | 95% CI | snippets |
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

Family contrast: CodeSteer's four **+3.89** vs the untested four **−3.14**, difference
**+7.02 [+2.00, +10.09]**. No individual CI excluded zero; the family contrast was the only
powered statistic.

### Seed 2000 — the sign flips

108 units, 15,018 generations, 0 failures, and the first grid in this repo where the seed actually
reaches the sampler (`[artifact-fix] seeded torch/numpy/random with 2000`).

| transform | seed 1000 | seed 2000 | swing |
|---|---|---|---|
| arith_rewriting | **−8.22** | **−5.65** | 2.6 |
| branch_inversion | −2.21 | +0.66 | 2.9 |
| opaque_predicates | −3.54 | **+4.22** | **7.8** |
| loop_transformation | −0.87 | +2.92 | 3.8 |
| adversarial_rename | −0.57 | +3.16 | 3.7 |
| **family of five** | **−1.24** [−4.41, +1.41] | **+4.46** [+0.85, +7.09] | **sign flip** |

**The +7.02 contrast was a one-draw result whose interval described sampling *within* the draw,
not variability across draws.** Four of five transforms cross zero.

**Two honest caveats.** (a) The seed-2000 grid does not contain CodeSteer's original four
transforms — it was forked from a config that deliberately omits them, so the *full* old-vs-new
contrast cannot be computed at the second seed; what is shown is the new-family absolute delta,
which is comparable. (b) Seed 1000 predates the seeding fix, so it is an uncontrolled draw and the
comparison absorbs sampling noise as well as seed variance. Neither rescues the finding.

### What survives
`arith_rewriting` is negative at **both** seeds (−8.22, −5.65) — and it is precisely the transform
CodeSteer itself flagged as resisting steering. A one-transform claim with two seeds is defensible
where a five-transform average is not.

---

## 5. N10b — malice detection, and how much of it is one generic word

### Design
183 real malicious PyPI packages vs **179 benign controls matched on length** (ratio 1.005,
medians 1044 vs 1020), **surface-obfuscation appearance**, and **77.7% hard negatives**:
legitimate code that imports `subprocess`, decodes base64, hits the network, or runs install
hooks. Scoring is a frozen regex vocabulary against reads the verbalizer produced from
activations alone — it never sees the code.

### The reported result, and its decomposition

| | neutral (primary) | security-framed |
|---|---|---|
| discrimination AUC | **0.764** [0.717, 0.808] | 0.778 [0.730, 0.822] |
| — hard negatives only | 0.757 [0.708, 0.805] | 0.757 |
| length as a confound | 0.477 | 0.477 |

**That 0.764 is inflated by a single generic term.** Decomposing the vocabulary:

| vocabulary | AUC | 95% CI |
|---|---|---|
| full (as originally reported) | **0.764** | [0.717, 0.808] |
| without `payload` | 0.684 | [0.638, 0.731] |
| **`payload` alone** | **0.708** | [0.665, 0.751] |
| unambiguous-malice terms only | **0.648** | [0.614, 0.685] |

`payload` alone nearly reproduces the full number, and removing it costs 8 points. It is
**46% of all malware vocabulary hits** (251 of 543). This is the same shape as the `encoding`
confound this project caught earlier — a label that looks like content and is partly detecting
surface. Malware does more HTTP POSTing than library code, and "payload" rides along.

### Per-term rates

| term | malware | benign | ratio |
|---|---|---|---|
| **payload** | **251** | 27 | 9.3 |
| webhook | 65 | **0** | ∞ |
| password | 62 | 27 | 2.3 |
| credential | 59 | 39 | 1.5 |
| wallet | 20 | 1 | 20 |
| telegram bot | 16 | 1 | 16 |
| reverse shell | 13 | **0** | ∞ |
| scheduled task | 11 | **0** | ∞ |
| remote code execut | 10 | **0** | ∞ |
| exfiltrat | 7 | **0** | ∞ |
| steal | 7 | **0** | ∞ |
| keylog | 1 | **0** | ∞ |
| **api_key** | **1** | **19** | **0.1 — backwards** |

### What real detection looks like — reads firing the unambiguous terms

These fire on malware and **never once** on 2,506 benign reads.

**`reverse shell`** — sample `mw-043e06d184c2`, REMOTE_EXEC, `rt_cos` 0.79:

> *"Technical chat format with code snippet showing a Python exploit tool, describing a 'netcat'
> connection behavior for **a malware agent**. The sentence structure 'The script connects to a
> remote server, **establishes a reverse shell, and transfers its stdin/stdout to the attacker's
> machine**' mirrors the original description of the backd[oor]…"*

**`exfiltrat`** — sample `mw-28fabe8b889b`, REMOTE_EXEC, `rt_cos` 0.87:

> *"…showing TCP packet payload extraction, detailing a specific protocol behavior pattern. The
> sentence ending 'The operation would demonstrate a remote connection to the target machine,
> **allowing the attacker to exfiltrate data to their own system**'…"*

**`keylog`** — sample `mw-c1e611c7060e`, REMOTE_EXEC, `rt_cos` 0.90:

> *"…suggesting a **hacker/developer context** with playful or mock project names. The quoted
> string is mid-sentence describing a specific project name (**'malicious keylogger'**)…"*

**`webhook`** — sample `mw-d8d18719011a`, REMOTE_EXEC, `rt_cos` 0.88:

> *"Code snippet showing a **Discord webhook** payload… 'And also check if it's a pipe to other
> services like **pwndb** or piped' suggests a list of obscure streaming destinations or **malware
> tools**…"*

When the instrument fires these terms, it is genuinely describing attack behaviour. That is the
real signal, and it is worth **AUC 0.648** on its own.

### What the inflated part looks like — `payload` on benign code

**`payload` + `credential`** — benign `bn-a48c33242ac7`, package **`sentry-sdk`**:

> *"Python code structure with a service definition pattern using **AWS Lambda** context…
> `def handler(lambda_context)` begins a function call with `process(event`…"*

> *"…code blocks showing **AWS SDK initialization and logging setup**… 'The code block
> demonstrates the initialization of a **FastAPI** application'…"*

Nothing malicious. Ordinary cloud-service code, firing two malice terms.

**`api_key` running backwards** — benign `bn-2edac6929aa8`:

> *"…showing Python CLI behavior using `requests` and `json` attributes, explaining a
> configuration file pattern. The sentence 'The `url` field is marked as `default='not-set'`,
> implying the actual token…'"*

Legitimate API-key configuration in a library. The term fires **19× on benign against 1× on
malware** — the only term in the vocabulary pointing the wrong way.

### Per-item error rates, which the AUC hides

| | |
|---|---|
| malware samples with **zero** malice reads (missed) | **54/183 = 29.5%** |
| benign samples with ≥1 malice read (false positive) | **41/179 = 22.9%** |
| worst benign false positive | `bn-73881a0e846c`, package **`openai`**, score 0.36 |

Roughly three in ten malware samples produce no malice read at all, and roughly one in four
benign packages produce at least one.

### What it cannot do
Asking *which* capability a sample has: 87.4% of items got a read naming a malicious capability
— but reads borrowed from **other** samples scored **90.3%**. The statistic is saturated: at a
~15% per-read rate over 14 reads, "at least one fires" is ~90% by arithmetic alone.

### Honest statement
Reads discriminate malicious from matched-benign code at **AUC ≈ 0.65** on vocabulary that
cannot be dismissed as generic, rising to 0.76 only when a generic web-programming term is
included. The signal is real and it is **population-level and theme-shaped**: it says "something
hostile is going on here", not which sample, not which capability, and it misses 30% of malware
outright.

## 6. N13 — no internal signal separates "torn" from "confidently wrong"

### Design
Correctness conflates two opposite states: *confidently wrong* (one reading, firmly held) and
*torn* (two readings, unresolved). Sampling K=8 answers per item at T=0.8 builds the missing label.

### Gates

| gate | result | verdict |
|---|---|---|
| entropy has spread | 18.8% perfectly consistent (stop if ≥80%) | **PASS** |
| the trap raises entropy | L1b 0.3815 vs matched L0 0.3820, **diff −0.0005** [−0.0815, +0.0816] | **FAIL** |
| instability floor | adversarial 0.0225 vs clean-name floor 0.0116, **+0.0109** [0.0065, 0.0153] | **PASS** |

Gate 1b failing is informative: **adversarial renaming does not make the model less decisive.** It
makes it wrong without visibly destabilising it — independent support for B3's echo account.

### Worked examples — what the three states look like

Of 330 graded items: 210 right, 43 torn-and-wrong, 16 committed-and-wrong.

| state | item | entropy | the 8 sampled answers |
|---|---|---|---|
| **torn** | `JavaScript/63:L0` | 0.917 | `480`, `927`, `1512`, `307`, 2×unparsed — every sample different |
| **torn** | `JavaScript/63:L2` | 1.000 | `36`, `987`, `199`, `144`, … all distinct |
| **committed wrong** | `cruxeval-js/106:L3` | 0.354 | `[4,16,-4,18,6,6]` **6 of 8 times** — held firmly, and wrong |
| **right** | `JavaScript/137:L0` | 0.000 | `2` **8 of 8** |

Note `JavaScript/63`: **torn at L0 (clean) but correct at L1b (adversarially renamed)** — a
concrete instance of the non-monotone tier effect.

### The 2×2, and the horse race
Wrong items matched on reply length, split torn vs committed (34 matched pairs):

| measure | AUC | torn | committed |
|---|---|---|---|
| `rt_cos` | 0.491 | 0.8619 | 0.8622 |
| `act_norm` | 0.442 | 108.12 | 109.07 |
| read instability | 0.408 (n=14) | 0.0241 | 0.0264 |

Length-confound check 0.5117, so the matching worked. **All three at or below chance.**

Predicting answer entropy, nested over reply length:

| model | R² | ΔR² | p(perm) |
|---|---|---|---|
| log reply length | 0.3511 | 0.3511 | 0.001 |
| + `rt_cos` | 0.3537 | 0.0026 | 0.402 |
| + `act_norm` | 0.3537 | **0.0000** | 0.983 |
| + **read instability** | 0.3703 | **0.0166** | **0.029** |

Instability is the one internal signal with incremental content — and it **fails the 2×2** at
n=14. Exploratory, unconfirmed.

**An incidental result stronger than any of the above:** answer entropy separates right from wrong
at **AUC 0.843** (mean 0.672 wrong vs 0.296 right), and only **6 of 120** wrong items are perfectly
consistent. On this corpus, being wrong and being unstable nearly coincide.

---

## 7. The three that produced no usable result

**B2 — unmeasurable.** Scoring needs a ground-truth algorithm per program, and only ~11 of 70
stimuli have one the closed vocabulary can score (these are "find the closest two elements", not
textbook algorithms). It was the **gate** for B3 and B4; both ran without it and both failed. That
ordering is a real weakness.

**B6 — not adjudicated.** An 8B judge separated real from shuffled read/text pairs at **AUC 0.757**
and could not rank a single item: **κ = 0.049** against a second rater (which answered "2" on 91%
of items), ρ = 0.032 against a free baseline. Population separation is not per-item reliability.
The judge was retired rather than swapped.

**B1 — refuted, and unsurprising.** The frozen autoencoder reads a sibling model at median
round-trip 0.694 vs 0.864 on its own host. As you noted, it was never trained for that model, so
the direction of the result is not news. The one part worth keeping: the loss is **directional, not
scalar** — the sibling's activations are 38% larger in norm, but injection rescales every vector to
a fixed norm first, so magnitude is discarded and cannot be the cause. That also makes plan v0.3's
"re-derive the injection scale" remedy a provable no-op.

---

## 8. What generalizes

**Reads carry theme-level content, not item-level content.** They separate populations
(malicious vs benign, AUC 0.648 on unambiguous vocabulary; 0.764 only with a generic term
included) and never identify *which* capability, *which* algorithm, or
whether *this* model was misled. Every null in the programme is an item-level claim; the only
result that held is population-level — and even it misses 30% of malware per item.

**The belief direction is orthogonal to the one that matters.** Measured on 164 Java programs, the
autoencoder's language-space direction sits at cosine **−0.001** [IQR −0.019, +0.021] to the
clean-minus-obfuscated activation direction — statistically indistinguishable from a random
vector's **−0.000**. Meanwhile V3 and V4 share direction at **+0.702** as they must. This is the
mechanical reason reversing V1 behaves like reversing noise, and it independently reproduces the
same finding from the Python/JS corpus (cos ≈ +0.02 there).

**Five promising point estimates did not survive their own intervals.**

| result | first look | at full evidence |
|---|---|---|
| B3 coupling | 3 of 3 | **3 of 10**, p = 0.138 |
| B4 gate | V1 +0.086 tracking the oracle at n=34 | **exact tie** at n=60 |
| B0 family | +7.02 [+2.00, +10.09] | **sign flips** at seed 2 |
| B5 composition | +3.94 interaction, +1.81 over attention | **every CI spans zero** |
| N13 instability | ΔR² 0.0166, p = 0.029 | **fails the matched 2×2** |

Each was caught by a control specified in advance — a foreign-read null, a no-NLA baseline, a
second seed, a random-direction control, a matched contrast. **The discipline is the part that
generalizes.**

---

## Provenance

| experiment | scale | file |
|---|---|---|
| B3 | 300 captures / 6,171 reads (+113 / 2,319 follow-up) | `data/nla/n11/` |
| B4 | 2,160 generations, 5 alphas, 0 errors | `data/nla/n12/` |
| B5 | 6 cells × 1,930 cases | `data/nla/b5/`, `artifact/artifacts/` |
| B0 | 90 units / 12,432 generations | `results/tables/grid.csv` |
| E5b | 108 units / 15,018 generations | `results/tables/grid_seed2.csv` |
| N10b | 183 malicious + 179 matched benign | `data/nla/n10b/` |
| N13 | 330 items × K=8, 228 instability cases | `data/nla/n13/` |

Pre-registrations and dated entries in `transcoders/log/nla-harness/`.
