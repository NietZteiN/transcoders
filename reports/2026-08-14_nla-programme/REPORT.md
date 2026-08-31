# Can a Natural Language Autoencoder read what a model believes about code?

*Instrument-2 programme report · 2026-08-08 → 2026-08-14*
*Subject: `Qwen/Qwen2.5-7B-Instruct`, layer-20 residual (d=3584). Instrument: `kitft/nla-qwen2.5-7b-L20-{av,ar}`, frozen released checkpoints. Seed 20260724 throughout. Env `nla-mi`.*

---

## Executive summary

A Natural Language Autoencoder turns one residual-stream vector into English and back. This
programme asked whether that lets us read a model's **belief about a program** — what it thinks
the code does — and whether a wrong belief can be detected when the model's stated output
disagrees with it.

Across three experiments and 13,598 verbalized reads, the answer is **no, not at the level of
individual items**. The instrument works: reads are on-topic, round-trip faithfulness sits in
the validated band, and injection never failed. What it produces is **theme-level** content —
"this is security-flavoured code", "this mentions sorting" — that does not identify *which*
capability a specific program has, and does not carry a belief away from the token that
suggested it.

Four findings, each with the mechanism that produced it:

1. **Prompt framing accounts for a large share of apparent "malice" reads.** On identical
   malware and identical activations, generic threat vocabulary appears **2.5× more often**
   under a security-framed preamble than a neutral one (17.5% vs 6.9%). A single-arm design
   would have reported that as evidence about the residual stream.
2. **Item-level capability identification fails its own null.** On 183 real malware samples,
   87.4% of items had a read naming a malicious capability — but reads drawn from *other*
   samples scored 90.3%. The statistic is at ceiling and carries no item-specific information.
3. **Adversarial renaming moves the reading of the *name*, not the belief about the program.**
   The raw effect is enormous (90% vs a 19% floor) and almost entirely lexical echo: 30.9% of
   reads *on* the misleading identifier name it, against 0.8% at the same locus in the neutral
   control, collapsing to ~3% anywhere else in the code.
4. **The one cell that pointed the other way did not survive a denominator.** Coupling between
   internal reads and behavioural deception was 3/3 in the first run; with a targeted follow-up
   it is **3/10** (30.0% vs 8.3%, Fisher p = 0.138).

The programme's most reusable output is not a positive result. It is a set of **judge-free
measurement tools plus the nulls that killed four plausible-looking effects** — and a corpus and
harness that make the next question cheap to ask.

---

## 1. What was measured, and how

**The instrument.** The AV (activation verbalizer) takes one layer-20 vector and emits 2–3
English snippets. The AR (activation reconstructor) maps text back to a vector; `cos(AR(text), v)`
is the round-trip faithfulness check. A read at position *t* reflects the model's whole
accumulated state at *t*, not a gloss of the token itself.

**Grading is mechanical everywhere.** This programme uses no LLM judge, because the previous one
failed: an 8B judge separated real from shuffled read/text pairs at AUC 0.757 yet could not rank
a single item (κ = 0.049 against a second rater), and the hypothesis it was built to test went
unadjudicated. What worked instead was a regex against a ground-truth field the verbalizer never
sees. Every measure below is of that kind.

**Every headline statistic carries a null attached to *that* statistic.** An earlier hypothesis
died because its null guarded a descriptive trajectory while the deciding test ran unguarded.

| experiment | corpus | captures | reads |
|---|---|---|---|
| **N10** malware triage, 2 framing arms | 183 PyPI malware samples | 366 | 5,108 |
| **N11** adversarial renaming, C0/C1/C2 | 100 Java snippets × 3 | 300 | 6,171 |
| **N11-c** coupling follow-up | 19 deceived + 19 matched × 3 | 113 | 2,319 |
| | | **779** | **13,598** |

Read quality never became the limiting factor: `rt_cos` medians 0.870–0.886 against a validated
band of 0.70–0.96, and **0 of 13,598 reads** showed the mostly-CJK signature of injection failure.

---

## 2. N10 — malware triage: framing confound, and a ceiling

**Corpus.** 600 DataDog `malicious_intent` PyPI samples fetched; **59% were near-duplicates**
(typosquat families publishing near-identical code) and were removed, leaving **183** samples,
median 1,044 chars. Handled under a quarantine protocol: archives stay encrypted, extraction is
in-memory only, and a static lint proves no execution primitive can reach a sample.

**Design.** Two arms differing *only* in the preamble — `security` ("You are a security engineer
reviewing a package…") vs `neutral` (same task, same behaviours asked about, every threat word
removed). Reads at code tokens and over the model's own reasoning.

### The framing confound is real and large

| | security-framed | neutral |
|---|---|---|
| generic malice vocabulary | **17.5%** | **6.9%** |
| specific capability named | 15.5% | 11.4% |
| item-level any-capability | 87.4% | 70.5% |

On identical code and identical activations, the security preamble more than doubles generic
threat vocabulary. The two-tier split earned its keep: the *specific* tier moved 1.36× where the
generic tier moved 2.5×, which is exactly the separation it was designed to have.

### The item-level statistic is at ceiling and fails its null

| arm | own | foreign-read null | 95% CI |
|---|---|---|---|
| security | 0.874 | **0.903** | [0.858, 0.945] |
| neutral | 0.705 | **0.813** | [0.754, 0.869] |

87% looks strong until reads drawn from *other* samples do it slightly better. With a ~15%
per-read rate over 14 reads, P(at least one fires) ≈ 0.90 by base rate alone.

A sharper test — does the read name the **right** capability — clears a permuted-label null by
only +0.055, and that null is inflated to 0.53–0.61 by class imbalance: **78% of labelled samples
are `REMOTE_EXEC`**, so "downloads and executes a payload" is right most of the time without
reading anything.

> **Limitation that bounds this experiment.** The corpus is entirely malicious. There is no
> benign arm, so this says nothing about malware-vs-benign discrimination — only about
> identifying which capability a sample has, among other malware.

---

## 3. N11 — adversarial renaming: the effect is lexical echo

**Design.** Three conditions per Java snippet, differing *only* in identifiers:
**C0** original · **C1** neutral opaque rename (the confabulation floor) · **C2** adversarial
rename to a plausible **wrong** algorithm (`hasCloseElements` → `bubbleSort`, class `Solution` →
`BubbleSorter`, `threshold` → `swapped`).

C2's ground truth is exact **by construction** — we choose the injected algorithm from the
grader's own closed vocabulary — which is what makes scoring judge-free. The injected label is
near-uniform across 10 algorithms (max class 11.4%), deliberately avoiding the class imbalance
that crippled N10.

C1 and C2 differ in exactly one respect: `AdversarialRenamer` subclasses the neutral renamer and
overrides only the name generator, so scope, collision and harness analysis are shared. If they
differed anywhere else, C1 would stop being a valid floor.

### Where the effect lives

| locus | C0 | C1 (floor) | C2 |
|---|---|---|---|
| **on the misleading identifier** | 0.9% | 0.8% | **30.9%** |
| code, away from it | 1.6% | 1.8% | 3.0% |
| chain of thought | 1.4% | 2.2% | 3.6% |
| answer line | 0.3% | 0.4% | 3.0% |

Distance from the nearest misleading identifier, within C2:

| on it | 1–50 chars | 50–150 | 150–400 | 400+ |
|---|---|---|---|---|
| **30.9%** | 3.7% | 2.5% | 5.4% | 0.0% |

The raw item-level contrast is 90% vs a 19% floor (McNemar p < 1e-15) and is **almost entirely
echo**: reads at the token where the misleading name literally sits. Reporting that number would
have been reporting that the NLA can read a token.

### The echo-free residue does not clear its null

Excluding the identifier locus and requiring the claim to recur at ≥2 non-adjacent positions:
C0 3.0% · C1 2.0% · **C2 9.0%**, Δ +7.0 pt, discordant 8-vs-1, **McNemar p = 0.039**. Nominally
significant — but the foreign-read null for that same statistic spans **[0.02, 0.10]** and the
observed rate is 0.090. Shuffling read *content* between items, holding each item's read
positions fixed, reproduces the effect.

*(k = 2, not the paper's k = 3, because validation on 5,090 banked reads found a median of 2
gradeable reads per item and only 43.4% of items reaching three. A k ≥ 3 rule would have been
undefined for most of the corpus.)*

**A confound found, not designed around:** the parse rate declines monotonically —
C0 98% → C1 82% → C2 77% of runs produce a final answer line at all. Renaming makes the model
less likely to commit, which is a real behavioural effect but also shrinks the answer-locus
denominator exactly where it matters.

---

## 4. N11-c — the coupling follow-up: 3/3 becomes 3/10

N11's one live cell was **behavioural coupling**: every C2 item where the model *stated* the
injected wrong algorithm also carried a recurrent internal read (3/3), against 8.1% where it did
not. Right direction, right structure, three items.

The fix was not a bigger read capture. A read costs ~4 s and there are 21 per item; a *stated
answer* is one generation, and generations can be batched (the read path cannot — batching breaks
AV determinism). So: **screen wide and cheap, then spend read budget only where behavioural
signal exists.**

- **Screen:** 862 snippets × {C1, C2} = 1,724 batched generations, ~43 min at 41/min.
  Marginal deception rate on parsed C2: **2.6%** — matching N11's 3/77, so the base rate was not
  a small-sample artifact.
- **Capture:** 19 deceived + 19 controls **matched on injected algorithm and code length**,
  × 3 conditions. Deceived items are not a random sample, so without matching "deceived items
  have more capability reads" could just mean "deceived items are longer".

### Result

| | read-hit | rate |
|---|---|---|
| model **stated** the wrong algorithm | 3/10 | **30.0%** |
| did **not** state it | 2/24 | 8.3% |

Fisher exact **p = 0.138**, odds ratio 4.7. The direction survives; the significance does not.

**The cleaner result is a flat line.** In this deliberately deception-enriched set, reads away
from the identifier show **C0 0.054 · C1 0.063 · C2 0.063** — *no elevation whatsoever* — while
the identifier locus sits at 0.378. The effect lives on the identifier token and nowhere else.

The answer-locus figure (C2 12.8% vs C1 0.0%) looks striking but is confounded by the design:
half this set was *selected* because the model stated the wrong algorithm, so an answer-line read
naming it is reading the model's own committed output. In N11's unselected sample the same cell
was 3.0%.

> **Bounding the claim.** `hashing` is 8 of 19 deceived items (42%). The matched controls hold
> that base rate fixed within the contrast, but a coupling effect carried largely by one
> algorithm should not be generalized. The alternative reading is not that the model is deceived
> by hash-like names, but that hash vocabulary is easy to emit about many programs.

---

## 5. What this programme leaves behind

**Tools, all judge-free and unit-tested (64 tests):**

| | |
|---|---|
| `steer.py` | activation write hook: `h ← h + α‖h‖Δ̂`, verified α=0 byte-identical, edit lands at target with neighbours bit-unchanged, cursor survives the prefill/decode cache boundary |
| `steer_vectors.py` | the V1–V5 direction ladder + control battery, with the leakage guard in the type signature |
| `belief_grade.py` | closed-vocabulary algorithm grader, camelCase-aware, with a **surface-tracking diagnostic** and an affordability check for the recurrence rule |
| `deception_stats.py` | frozen defences: C1 floor, distance, recurrence, coupling, foreign-read null |

**Corpora:** 862 C0/C1/C2 Java triples (10 injected algorithms, max class 11.4%); 183 quarantined
malware samples; and four new obfuscation transforms with execution-validated corpora —
`opaque_predicates` 680, `loop_transformation` 247, `arith_rewriting` 147, `branch_inversion` 136
snippets. All nine transforms pass 360 execution-equivalence checks with **0 failures**.

**Methodological results that outlived the hypotheses:**

- A label that fires **more** as code gets harder to read is describing the obfuscation, not the
  program. `encoding` was the only label whose rate rose with tier (1.68% → 3.69%) while every
  content label fell; it was 10.7% of all fires and would have loaded onto the very contrast it
  was meant to measure.
- **Direction, not magnitude, exposes a surface confound.** Absolute rates looked fine.
- A pre-registered recurrence threshold must be checked for **affordability** against the data
  before it is frozen.
- The cheap wide screen is worth more than the expensive deep one when the base rate is low.

---

## 6. Honest position

Three experiments, four nulls, one instrument that worked throughout. Adversarial renaming moves
the *reading of the name* and not, detectably, the *belief about the program*; malware reads
register "security-flavoured code" without identifying which capability a sample has.

None of this refutes NLAs as an interpretability tool — it bounds what this NLA, at this layer,
on this model, can say about **individual items**. The theme-level content is real and reproduced
in every run. What is missing is item-level resolution, and every attempt to extract it ran into
the same wall: a base rate high enough that a foreign-item read does as well as the item's own.

**The remaining question the programme has not touched is causal.** Every measurement here is
observational — read the representation, correlate with behaviour. The write hook is built and
tested but has never been pointed at a real activation. Whether an NLA-derived direction can
*change* what a model believes about a program, and whether it beats a contrastive difference
vector that needs no NLA at all, is the one thing that would distinguish this instrument from a
descriptive one.
