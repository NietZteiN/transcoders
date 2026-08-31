# Can you read a language model's mind about code? Five experiments.

*Instrument-2 programme, complete report · 2026-08-08 → 2026-08-14*
*Written to be read cold: no prior familiarity with the project assumed.*

---

## Part 1 — What this is about

### The problem

Give a language model a program and ask what it outputs. Rename the variables to something
misleading — call a Fibonacci function `smoothArea` — and accuracy drops. We know this
behaviourally, from prior work. What we don't know is **what happens inside the model**: does it
form a *false belief* about what the program does, or does it just get distracted?

That distinction matters. If the model acquires a wrong belief, you might be able to correct it.
If it's noise, you can't.

### The instrument

A **Natural Language Autoencoder** (NLA) is a pair of trained models that turn a model's internal
state into English and back:

- The **verbalizer (AV)** takes one *activation vector* — the model's internal state at one word,
  a list of 3,584 numbers — and writes 2–3 English sentences describing it.
- The **reconstructor (AR)** goes the other way: text back to a vector.

Round-tripping (vector → text → vector) gives a quality score: if the reconstructed vector
matches the original, the English captured what was in there. We use Anthropic's released
checkpoints, frozen — we didn't train anything.

So: point the AV at a model while it reads code, and you get plain-English descriptions of what
it was "thinking" at each word. We call these **reads**.

### The question, in three parts

1. Do reads tell us what the model believes a program *does*?
2. Does adversarial renaming induce a *false belief*, visible in the reads?
3. Can we *edit* that belief and change the model's answer?

### Scale

| | |
|---|---|
| Subject model | Qwen2.5-7B-Instruct, layer 20 |
| Instrument | `kitft/nla-qwen2.5-7b-L20-{av,ar}`, frozen |
| **Reads collected** | **18,610** |
| Read-captures | 1,138 |
| Generation-only runs | 2,144 |
| **Total model calls** | **3,282** |
| Errors | 1 (a transient network drop) |
| Injection failures | **0 of 18,610** |

Read quality held throughout: median round-trip similarity 0.86–0.89, against a validated band
of 0.70–0.96. **The instrument was never the limiting factor** — every negative below is a real
negative, not a broken pipeline.

---

## Part 2 — How we measure, and why it's unusual

Three rules shaped everything, each learned from a prior failure in this project.

**No LLM judges.** An earlier experiment used a language model to grade whether a read matched a
piece of text. It told real pairs from shuffled ones at AUC 0.757 — and still couldn't rank a
single item (κ = 0.049 against a second grader). Population separation is not per-item
reliability. Everything here is scored by regular expressions against a ground-truth field the
verbalizer never sees.

**Every headline number carries a null attached to *that* number.** Not to a cousin of it. A
previous hypothesis died because its control guarded a descriptive statistic while the actual
decision rule ran unguarded, and a confounded p = 0.002 looked like support.

**The control condition is the tripwire.** Repeatedly, the thing that caught a broken experiment
was the *easy* condition failing. If a model scores 1/16 on unobfuscated code where it normally
scores 0.65, the task is broken, not hard.

---

## Part 3 — The five experiments

### N10 · Does a read say what a piece of malware does?

**Setup.** 183 real malicious Python packages from PyPI (600 downloaded; **59% were
near-duplicates** — typosquat families publishing near-identical code — and were removed). The
model reads each and explains it; we read its internal state.

Crucially, **two versions of the prompt**: one framing the task as security review, one
neutral — same task, same behaviours asked about, every threat word removed.

**Result 1 — the prompt does a lot of the work.**

| | security-framed | neutral |
|---|---|---|
| generic threat vocabulary | **17.5%** | **6.9%** |
| specific capability named | 15.5% | 11.4% |

Identical code, identical activations, **2.5× more threat talk** just from framing. Had we run
one arm, we'd have reported the prompt's own words as evidence about the model's mind.

**Result 2 — it can't tell you *which* malware.** 87.4% of samples had a read naming a malicious
capability. Impressive until you shuffle: reads taken from *other* samples scored **90.3%**. The
statistic is saturated — with a ~15% per-read rate over 14 reads, "at least one fires" is ~90%
by arithmetic alone.

> **Verdict: null.** Reads do not identify a specific sample's capability.

---

### N11 · Does a misleading name create a false belief?

**Setup.** Three versions of the same Java program, differing *only* in identifier names:

- **C0** original
- **C1** neutral gibberish names (`m_O0lI1`) — the floor
- **C2** names of a plausible **wrong** algorithm (`hasCloseElements` → `bubbleSort`)

C2's ground truth is exact because *we choose* the wrong algorithm. 100 programs × 3 versions,
6,171 reads.

**The headline looks spectacular.** 90% of C2 programs get a read naming the injected wrong
algorithm, against a 19% floor.

**Then you ask where those reads are.**

| where the read was taken | C0 | C1 (floor) | C2 |
|---|---|---|---|
| **on the misleading name itself** | 0.9% | 0.8% | **30.9%** |
| elsewhere in the code | 1.6% | 1.8% | 3.0% |
| in the model's reasoning | 1.4% | 2.2% | 3.6% |
| at the final answer | 0.3% | 0.4% | 3.0% |

And by distance from that name: **30.9%** on it → 3.7% at 1–50 characters → 2.5% at 50–150 →
**0.0%** past 400.

A belief propagates through a representation. An echo is local. **This is an echo** — the reads
are reading the word, not a belief about the program.

Strip out the reads sitting on the name and require the claim to recur at ≥2 separated positions:
C2 falls to 9.0% against a 2.0% floor. Nominally significant (p = 0.039) — but the matched null
for that same statistic spans [0.02, **0.10**], and 0.090 sits inside it.

> **Verdict: not established.** The effect is lexical echo.

---

### N11-c · The one promising thread, followed properly

N11 had one encouraging cell. In the handful of cases where the model *said* the wrong algorithm
out loud, its internal reads named it too — **3 out of 3** — versus 8.1% where it didn't. Right
shape, three items.

Getting more of those items looked expensive, until we noticed an asymmetry: a *read* costs ~4
seconds and we take 21 per program, but the model's *stated answer* is one generation, and
generations can be batched. So we screened all 862 programs for behavioural deception in 43
minutes (41/min), found the base rate is **2.6%**, then spent read budget only on the 19 deceived
programs plus 19 controls matched on injected algorithm and code length.

**Result:**

| | reads naming the wrong algorithm |
|---|---|
| model **said** the wrong algorithm | 3/10 = **30.0%** |
| model did **not** | 2/24 = 8.3% |

Fisher p = 0.138. **3/3 became 3/10.** The direction survives; the significance doesn't.

And a cleaner number from the same run: in this deception-*enriched* set, reads away from the
misleading name were **0.054 / 0.063 / 0.063** across the three conditions — no elevation
whatsoever — while reads on the name sat at 0.378.

> **Verdict: not established**, and the echo account is now supported twice on independent data.

---

### B4 · Can we edit the belief and change the answer?

Everything above is observational. This is the causal test: inject a direction into the model's
internal state and see whether accuracy on misleadingly-named code recovers.

**The design point that matters.** Beating "no steering" proves nothing — any big enough
perturbation changes outputs. So the bar was pre-registered as a **contrastive difference
vector** that uses no NLA at all: take the average internal-state difference between clean and
renamed versions of *other* programs, and push along it. If that matches the NLA's direction,
the verbalizer is buying interpretability but not capability.

60 programs, 7 conditions, 420 generations.

| condition | accuracy change | 95% CI |
|---|---|---|
| **just adding a sentence to the prompt** | **+0.100** | [−0.017, +0.233] |
| oracle ceiling (uses the clean program) | +0.083 | [−0.033, +0.200] |
| **NLA direction (V1)** | **+0.050** | [−0.083, +0.183] |
| **difference vector, no NLA (V3)** | **+0.050** | [−0.050, +0.167] |
| reversed NLA direction | +0.033 | [−0.083, +0.150] |
| random direction | +0.033 | [−0.067, +0.133] |
| another program's direction | −0.017 | [−0.117, +0.083] |

**The gate: V1 0.550 vs V3 0.550 — a dead tie.** Discordant 7 items each way, p = 1.00.

Two details make this more than an underpowered shrug. **Reversing the NLA direction performs
identically to a random one** (+0.033 vs +0.033) — if the direction carried signal, reversing it
should hurt; it doesn't, so it has no consistent sign. And **a sentence in the prompt beat every
steering condition**, including the oracle.

A geometry check ruled out the obvious bug: the difference vector and the oracle share direction
as they must (cos = +0.418), so V3 isn't sign-flipped. More interestingly, the NLA direction is
near-*orthogonal* to both (cos ≈ 0.02) — language-space and activation-difference-space are
genuinely different places, so the tie is a real coincidence of outcomes, not two names for one
vector.

> **Verdict: gate fails.** NLA-derived steering doesn't beat a difference vector, and neither
> beats prompting. *(Caveat: one α value, 60 items — this excludes a large effect, not a
> moderate one.)*

---

### N10b · The question N10 couldn't ask — and the one thing that works

N10's corpus was *entirely malicious*, so it could only ask "which capability", never "malicious
or not". That needs benign controls, and naive controls would be worthless: malicious PyPI
samples are short install hooks that shell out and read the environment, while popular packages
are long, clean libraries. Any signal would be a length-and-style detector.

So the 179 benign controls are matched on **length** (ratio 1.005; medians 1044 vs 1020),
**surface-obfuscation appearance**, and — the important one — **77.7% are "hard negatives"**:
legitimate code that imports `subprocess`, decodes base64, hits the network, or runs install
hooks. Both prompt framings, same items.

| | neutral (primary) | security |
|---|---|---|
| **discrimination AUC** | **0.764** [0.717, 0.808] | 0.778 [0.730, 0.822] |
| — **hard negatives only** | **0.757** [0.708, 0.805] | 0.757 |
| AR-space malice margin | 0.856 | 0.915 |
| length as a confound | **0.477** | 0.477 |

*(AUC 0.5 is a coin flip; 1.0 is perfect.)*

**It works, and it isn't an API detector.** Restricted to items where the benign side *also*
shells out and decodes base64, AUC is 0.757 — a drop of 0.007. Length is neutralised at 0.477.
Framing inflates the effect by only +0.014 here, against the 2.5× it inflated raw vocabulary.

> **Verdict: supported.** Reads discriminate malicious from matched benign code.

---

## Part 4 — What it all means

**The pattern across five experiments is consistent, and it is about resolution.**

Reads carry **theme-level** content — *"this is doing something hostile"*, *"this mentions
sorting"* — reliably enough to separate populations at AUC 0.76. They do **not** carry
**item-level** content: which capability *this* sample has, what algorithm *this* program
implements, whether *this* model has been misled.

Every null in this programme is an item-level claim. The one positive is population-level. That
isn't four failures and a fluke; it's one finding stated five ways.

**Practically:**

- ✅ Screening a corpus for "does this look hostile" — plausible, AUC 0.76 with the confounds controlled.
- ❌ Explaining *why* a specific sample is flagged — not supported.
- ❌ Detecting that a model has been misled on a specific program — not supported.
- ❌ Steering behaviour by editing a belief — no better than a difference vector, and worse than a sentence in the prompt.

**Three methodological results outlived the hypotheses:**

1. **A confound announces itself by direction, not magnitude.** One vocabulary entry (`encoding`)
   was the only label whose rate *rose* as code got harder to read — 1.68% → 3.69% — while every
   genuine content label fell. It was firing on "encoded variable names", i.e. describing the
   obfuscation, and made up 10.7% of all matches. The absolute rates looked completely normal.
2. **Check whether your statistic is affordable before you freeze it.** A rule requiring a claim
   to recur three times was undefined for most of the corpus, because the median program yields
   two scorable reads.
3. **When the base rate is low, screen wide and cheap.** Batched generation at 41/min found the
   rare deceived programs that 84-seconds-per-item reads would never have reached.

**And one about how this went wrong twice.** Both N11's coupling cell (3/3 → 3/10) and B4's
steering gate (V1 tracking the oracle at n=34 → an exact tie at n=60) looked like the
pre-registered success partway through, and reversed at full sample size. Partial-run tables
are not results.

---

## Part 5 — What we'd do next

| | |
|---|---|
| **Per-item reliability of the N10b score** | AUC 0.757 separates *populations*; a split-half check on each item's own reads would decide whether it can rank a single sample. This is the same distinction that sank the earlier LLM judge, pointed the other way. |
| **α sweep for steering** | B4 ran one injection strength. That's the biggest unforced limitation in the negative result. |
| **The paper's own steering protocol** | We tested a cheaper variant; editing the model's own verbalized read remains untested. |
| **A benign population that isn't top-PyPI** | "Benign" currently means popular library code. |

---

### Appendix — assets built

**Corpora:** 862 three-way Java programs (10 injected algorithms, max class 11.4%); 183
quarantined malware samples; 179 matched benign controls; four new obfuscation transforms
(680 / 247 / 147 / 136 programs) passing 360 execution-equivalence checks with 0 failures.

**Tools:** an activation write-hook verified inert at zero strength and exact at the target
position; a five-way steering-vector ladder with the leakage guard in the type signature; a
judge-free grader with a surface-confound diagnostic; and matched-control construction that
holds length, appearance and API surface fixed. 64 tests passing.

**Safety:** the malware corpus is quarantined — archives stay encrypted, extraction is in-memory
only, nothing is ever written to an import path, and a static lint proves no execution primitive
can reach a sample.
