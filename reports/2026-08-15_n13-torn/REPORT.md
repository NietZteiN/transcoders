# Can we tell when a language model is *confused*?

**N13 — a null result, and what it taught us about the instrument.**
Run 2026-08-15 · Qwen2.5-7B-Instruct, layer 20 · seed 20260724 · one GPU · 4 stages, 0 errors

---

## The short version

We asked whether anything measurable inside a language model shows that it is **torn** — holding
two incompatible interpretations at once, the way a person reading *"a plus b squared"* is split
between `a + b²` and `(a+b)²`.

We tested three candidate signals. **All three are null.** The reason is a single fact we did not
expect: the most promising new measurement turned out to be a second way of measuring something we
were already measuring. Its correlation with the existing metric is **−0.96**.

Along the way we got one solid positive, for free, from a place we weren't looking: **how often the
model changes its answer predicts whether it is right, at AUC 0.87** — better than any internal
signal we have.

---

## 1. Background, from scratch

### What a "Natural Language Autoencoder" is

A language model processes text through ~28 stacked layers. At each layer, every token position
holds a vector of 3,584 numbers — the **residual stream**. That vector is the model's working state
at that point. It is not human-readable.

An **NLA** is a pair of small trained models that make it readable:

- **AV (activation verbalizer)** — takes one vector, returns an English sentence describing it.
- **AR (activation reconstructor)** — takes an English sentence, returns a vector.

Because you can go both directions, you can check the description: take a vector `v`, verbalize it,
reconstruct the verbalization back into a vector, and measure the cosine similarity between the
result and `v`. That number is **`rt_cos`** ("round-trip cosine"), and it is the instrument's
self-reported **faithfulness** — how much of the original state survived the trip through English.

We use the released checkpoints `kitft/nla-qwen2.5-7b-L20-{av,ar}`, frozen, at layer 20. Nothing is
trained here.

### The obfuscation ladder

The wider project studies how models read deliberately obscured code. Every program appears at five
tiers:

| tier | what it is |
|---|---|
| **L0** | the original program |
| **L1** | identifiers renamed to meaningless strings (`var_bd90`) |
| **L1b** | identifiers renamed to *misleading* ones — a decoy asserting the wrong algorithm |
| **L2** | control-flow flattened into a dispatcher loop |
| **L3** | everything at once |

**L1b is the interesting one.** It is a Stroop-like trap: the code does one thing, the names claim
another.

### What we already knew going in

Two prior results set up this experiment.

**`rt_cos` does not predict correctness.** We tested it (HT12) and refuted it: on a length-matched
corpus, β = −0.00022, CI [−0.0051, +0.0047], p = .93. The apparent effect was length — wrong
reasoning traces run ~1.6× longer, and `rt_cos` correlates +0.277 with how much text the verbalizer
produced.

**`rt_cos` looks like a "describability" meter.** Sorting reads by what kind of token they sit on:
meaningless renames read *worst* (0.847), function names read *best* (0.934), and — the striking one
— misleading names read exactly as well as real ones (0.879 vs 0.878, not significant). A decoy
injects semantics that are wrong but perfectly describable.

---

## 2. The question, and why correctness cannot answer it

`rt_cos` failing to predict correctness does not settle whether it detects confusion, because
**"wrong" hides two opposite internal states**:

|  | gives the same answer every time | gives different answers |
|---|---|---|
| **right** | confidently right | lucky |
| **wrong** | **confidently wrong** — one reading, firmly held, mistaken | **torn** — two readings, unresolved |

Both bottom cells score `correct = False`. A metric that tracks correctness cannot separate them.
A metric that tracks *torn-ness* would.

**Nothing in the project could see this distinction, for a mundane reason: every stored run was a
single greedy generation.** One sample cannot tell you whether the model would have answered the
same way ten times. The missing axis had to be built.

---

## 3. What we built

Four stages, run back-to-back unattended on one GPU.

### Stage 1 — a measure of torn-ness (1.5 h)

Sample **K = 8 answers per item at temperature 0.8** for all 330 graded items, and measure the
disagreement: how many distinct answers, and their normalized entropy (0 = always the same answer,
1 = eight different ones).

The prompt builder and grader were **imported verbatim** from the original capture script rather
than rewritten — a re-derived prompt cost four bugs in an earlier experiment. Verification: regenerate
greedy answers and check they reproduce the stored ones. **8/8 matched.**

### Stage 2 — recover the residual-stream norm (64 s)

`act_norm` (the magnitude ‖v‖ of the state vector) is the most obvious alternative signal, and the
project had **never tested it against anything** — for the accidental reason that it was stored only
on corpora that have no correctness labels. One forward pass per case recovers it at all 5,090
stored read positions.

Verification: each stored read recorded the token it was taken from, so we can check that
re-extraction still lands on that exact token. **5,090 / 5,090 matched**, 0 positions dropped, 0
outliers, median norm 109.7 (inside the documented 100–170 band).

### Stage 3 — read instability (2.1 h)

**The novel instrument, and the one most likely to work**, because its construction matches the
phenomenon. `rt_cos` asks whether *one* sentence carried the vector. Instability asks whether the
vector *has* one sentence:

> Verbalize the **same vector** K = 5 times at temperature 0.7 and measure how much the five
> descriptions disagree.

A vector holding one interpretation should describe consistently. A vector holding two incompatible
interpretations should not.

Agreement is measured in **AR space** — each of the five sentences is mapped back to a vector, and we
take the mean pairwise cosine. This asks "do these sentences *mean* the same thing", so paraphrases
count as agreement. A judge-free lexical overlap (Jaccard) runs alongside, because an earlier attempt
to use an LLM as a judge failed badly (κ = 0.049).

400 positions, 2,000 reads, stratified across six token classes. Two matter most:

- **`fn_orig`** — real function names in clean code, the most describable class. Unambiguous content,
  so it should be **stable**. This is the *floor*: if it is as unstable as everything else, we are
  measuring decoder randomness, not the model.
- **`adversarial`** vs **`l1_neutral`** — decoy names vs meaningless names. Both are renamed
  identifiers of the same token shape; they differ in exactly one thing: whether the new name
  asserts a confident wrong meaning. This pair separates the two ways instability can arise —
  **superposition** (two readings present) from **emptiness** (no reading present, so the decoder
  free-runs).

### Stage 4 — analysis (seconds)

A **horse race**: does any internal signal explain answer entropy *beyond reply length*? Length is
the baseline because length is what explained away the earlier result. Tested by permutation.

And the **2×2**: torn vs confidently-wrong items, matched on reply length, compared on every metric.

---

## 4. Results

### The manipulation does not induce torn-ness

The L1b trap does **not** raise answer entropy over length-matched L0 items:
**−0.011, CI [−0.098, +0.071]** (54 pairs).

*(An interim reading at 19 pairs showed +0.103 and reversed at full n — the same interim-reverses
pattern that has bitten this project before.)*

### All three internal signals are null

Horse race — predicting answer entropy, 1,000 permutations, n = 194:

| predictor | R² | p<sub>perm</sub> | beats length? |
|---|---|---|---|
| reply length alone | **0.344** | .001 | — |
| + `rt_cos` | 0.345 | .606 | no |
| + `act_norm` | 0.345 | .409 | no |
| + read instability | 0.354 | .234 | no |

The 2×2 — torn vs confidently-wrong, length-matched (length AUC 0.485, so the matching held):

| metric | AUC | CI 95% |
|---|---|---|
| `rt_cos` | 0.470 | [0.329, 0.617] |
| `act_norm` | 0.432 | [0.292, 0.583] |
| read instability | 0.424 | [0.209, 0.653] |

All at chance. **Reply length alone explains 34% of answer-entropy variance; every internal signal
fights over the remaining couple of percent and loses.**

### Why: instability *is* describability

Read instability by token class, against the previously measured `rt_cos` of the same classes:

| class | instability | `rt_cos` |
|---|---|---|
| chain-of-thought | 0.0279 | 0.8537 |
| answer line | 0.0256 | 0.8700 |
| meaningless rename | 0.0242 | 0.8519 |
| **decoy rename** | 0.0225 | 0.8749 |
| original name | 0.0193 | 0.8887 |
| **function name** | **0.0116** | **0.9342** |

The two columns are the same ordering, inverted. Correlation: **−0.757 per read, −0.958 across
classes.**

Read instability is not a new instrument. It is describability, measured a second way. A vector that
is hard to describe also gets described *inconsistently*.

That single fact explains everything else:

1. **The floor gate passes** (decoys are less stable than function names: +0.011, CI [0.007, 0.015]
   in AR space; +0.024, CI [0.009, 0.039] lexically) — so the measure genuinely discriminates. But
   it discriminates *describability*, not ambiguity.
2. **Superposition vs emptiness is null**: decoy − meaningless = **−0.002, CI [−0.006, +0.003]**.
   The point estimate is *negative* — meaningless names are, if anything, marginally less stable
   than decoys. Instability is nearer an emptiness meter than a superposition meter.
3. **A decoy is not a second reading.** It injects semantics that are wrong but confident and
   perfectly describable, exactly as the earlier 0.879-vs-0.878 result implied.

### What instability actually looks like

The most stable read in the corpus sits on the function-name token `unique`. Five independent
verbalizations:

> *"Code snippet format with TypeScript / JavaScript context … array operations or filtering"*
> *"JavaScript code snippet … `const` variable definition … a sample array"*
> *"JavaScript code snippet format with a comment prompt structure ("const unique") … array operations"*

Five different sentences, one consistent meaning.

The least stable read sits on a **whitespace token**:

> *"…Python `DataFrame` state transitions…"*
> *"…FSM state machine in a TDD context…"*
> *"…a finance model…"*
> *"…state machine for a finance app, Enum pattern…"*
> *"…a sequence diagram… using `GameState`…"*

All five agree on the *form* (structured Python with numbered steps) and invent a different domain
each time. That is not a model torn between two readings of the code. That is a decoder
confabulating because the vector gave it nothing to say.

### The one real positive

**How often the model changes its answer predicts whether it is right: AUC 0.869**
(mean entropy 0.668 when wrong vs 0.240 when right).

This is the known self-consistency effect, and it fell out as a by-product. It is a *better*
correctness signal than anything the NLA instrument provides, and it is the baseline any future
internal signal must beat.

Note also that the "confidently wrong" state is genuinely **rare**: only 6 of 107 wrong items gave
the same answer all eight times, against 83 of 223 right items. On this corpus, being wrong and being
unstable nearly coincide.

---

## 5. A correction we caught, and one we nearly missed

While pulling examples for this report, two of the three "confidently wrong" items turned out to be
**grading artifacts**:

| stored answers | problem |
|---|---|
| `**[]` (×4) and `[]` (×4) | markdown bold split one answer into two |
| `{1:none,3:none,4:none,2:none}` vs `{1:none,2:none,3:none,4:none}` | dictionary key order — identical mappings |

The answer comparator stripped whitespace and quotes but not markdown emphasis, and did not
canonicalize key order. **36% of items contain a `*` in at least one sampled answer.**

This is not cosmetic: entropy is the *outcome variable*. Inflating it adds noise, and noise in an
outcome **attenuates** associations — so a null could have been an artifact of the label rather than
a fact about the model. We re-ran everything under a hardened comparator.

**The result: the one marginal positive disappeared.** Read instability's edge over length went from
p = .029 to **p = .234**. Meanwhile the behavioural finding *strengthened*, AUC 0.843 → **0.869**.

Both movements are diagnostic. Removing label noise pushed the borderline internal result toward
null and the strong behavioural result away from it — which is what real and spurious effects
respectively do. All numbers in this report are the hardened ones.

We also caught a **statistical bug that would have produced a false positive as the headline.** The
horse race originally tested each signal by bootstrapping its ΔR² and asking whether the interval
excluded zero. But ΔR² is **non-negative by construction** — adding any column to a regression cannot
lower in-sample R² — so that interval essentially always excludes zero. On synthetic data with three
predictors we *knew* were pure noise, all three were reported "significant". Replaced with a
permutation test; the same data then gave p = .51–.58.

---

## 6. What this means

**For the original question:** the faithfulness score cannot signal confusion. Neither can the
residual-stream norm, nor read instability. And the reason is not that we measured badly — it is that
all three are measuring **describability**, which is dominated by **length**. Faithfulness is not
lower on code the model finds confusing; it is lower where there is *less to say*.

**For the instrument:** this is a sharper characterization of `rt_cos` than we had. It is a
length-and-describability meter, confirmed by a test designed to break that account. Read instability
is a *better-estimated* version of the same quantity (5 samples instead of 1) and still explains
nothing extra.

**For the trap:** adversarial renaming makes the model **wrong, not torn**. It does not install a
competing interpretation the model wavers between; it installs a confident wrong one.

---

## 7. Limitations

- One model, one layer (20), one temperature pair. **Torn-ness could be real but not linearly
  localized at layer 20**, in which case a per-position scalar was never going to see it.
- Read instability averages only ~1.9 positions per item, so the item-level term is underpowered.
  The read-level class contrasts (400 reads over 210 cases) are the well-powered tests; the 2×2
  instability cell rests on **14 pairs**.
- The floor class `fn_orig` is confounded with tier and with token shape (whole words vs
  bracket-glued fragments), which is why the interpretive weight sits on the shape-matched
  decoy-vs-meaningless pair instead.
- AR-space cosine is compressed (0.971–0.994). Every contrast was therefore also run on lexical
  overlap, which has ~15× the range and no learned component; the two agreed everywhere.
- Exploratory throughout — not pre-registered, and deliberately not added to the pre-registered
  hypothesis family. Nothing here claims significance; the nulls are informative, not proof of
  absence.

---

## 8. What to do next

1. **Sweep layers.** The cheapest remaining test of the "not linearly localized at 20" escape route.
   No training required for `act_norm` or instability.
2. **Induce torn-ness properly.** Gate 1b shows adversarial renaming does not do it. A real test
   needs genuinely ambiguous programs — two defensible outputs — not one answer behind a misleading
   name. That is a stimulus-design problem, not a measurement problem.
3. **Treat answer entropy as the baseline.** At AUC 0.869 it beats every internal signal in this
   programme. Any future claim that an internal measurement is useful has to clear it.

---

## Reproducing

```
nla/src/answer_entropy.py       # stage 1  — K=8 sampled answers, per-item entropy
nla/src/recover_act_norm.py     # stage 2  — forward-pass-only norm recovery
nla/src/read_instability.py     # stage 3  — K=5 verbalizations per vector
nla/src/reharden_entropy.py     #   ↳ robustness — hardened answer comparator
src/analysis/n13_torn.py        # stage 4  — horse race, 2x2, class contrasts
nla/scripts/n13_autopilot.sh    # runs stages 2-4 unattended on one GPU
nla/tests/test_n13.py           # 26 unit tests (91 in the suite)
```

Seed 20260724 throughout. Outputs in `data/nla/n13/`. Full detail in the ledger:
`log/nla-harness/2026-08-15_n13-torn-null.md` and its correction addendum
`log/nla-harness/2026-08-15_n13-normalization-robustness.md`.
