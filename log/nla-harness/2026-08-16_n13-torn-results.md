# 2026-08-16 — N13: no internal signal separates *torn* from *confidently wrong*, but read instability survives the horse race

**Exploratory throughout, labelled as such.** No pre-registration, no addition to the
{HT12, HT13, HT14} family. Any survivor gets a frozen confirmatory test on held-out data before
it is claimed.

## Goal / hypothesis

Correctness conflates two opposite internal states: **confidently wrong** (one reading, firmly
held, mistaken) and **torn** (two readings, unresolved). HT12 asked whether `rt_cos` predicts
correctness and refuted it — on the length-matched dense corpus β = −0.00022, CI
[−0.0051, +0.0047], because `rt_cos` correlates +0.277 with read length and wrong traces run
~1.6× longer. If any internal signal tracks torn-ness, correctness is exactly the label that
would hide it.

Nothing in this programme could previously see torn-ness, because every capture was a single
greedy run. N13 builds the missing label (answer instability under sampling) and races three
internal candidates against it.

## Setup

```
env      nla-mi (stages 1-3) / transcoders-mi (stage 4)
seed     20260724
GPU      1   (autopilot: nla/scripts/n13_autopilot.sh, tmux n13s1 -> stages 2,3)
corpus   banked overnight set, data/nla/overnight/2026-08-04/captures.jsonl
stage 1  answer_entropy.py   K=8 @ T=0.8, top_p 0.95, batched  — 330 graded items
stage 2  recover_act_norm.py forward-pass only                 — 380 cases
stage 3  read_instability.py K=5 @ T=0.7, sequential AV        — 228 cases / 194 with all covariates
stage 4  src/analysis/n13_torn.py
```

## Gates

| gate | result | verdict |
|---|---|---|
| **1a** entropy has spread | 18.8% of items perfectly consistent (rule: stop if ≥80%); mean entropy 0.433 | **PASS** |
| **1b** the trap raises entropy | L1b 0.3815 vs matched L0 0.3820, diff **−0.0005**, CI [−0.0815, +0.0816] | **FAIL** |
| **3** instability floor | adversarial 0.0225 vs clean-function-name floor 0.0116, diff **+0.0109**, CI [0.0065, 0.0153] | **PASS** |

**Gate 1b failing is the most informative line here.** Adversarial renaming does *not* make the
model's answer less stable — it makes it wrong. Whatever L1b does, it is not inducing visible
indecision, which is consistent with N11's finding that the effect is lexical echo rather than a
propagating belief.

**Gate 3 passing matters for validity:** read instability is not merely decoder temperature. The
same K-sampling applied to unambiguous clean function-name tokens is measurably *more* stable
than at adversarial identifiers, and the CI excludes zero. The measure has content.

## Results

### The 2×2 the programme has never been able to ask — null

Wrong items, matched on reply length (`|log ratio| ≤ log 1.5`), split into **torn**
(modal share ≤ 0.25) vs **committed wrong** (modal share ≥ 0.5): 34 matched pairs.

| measure | AUC | torn | committed |
|---|---|---|---|
| `rt_cos` | 0.491 | 0.8619 | 0.8622 |
| `act_norm` | 0.442 | 108.12 | 109.07 |
| read instability | 0.408 (n=14) | 0.0241 | 0.0264 |

Length-confound check **0.5117**, so the matching worked and these AUCs mean what they say.
**No internal measure separates the two states.** All three sit at or below chance.

### The horse race — instability is the one survivor

Predicting answer entropy, nested over the baseline that explained away HT12 (log reply tokens):

| model | R² | ΔR² | p(perm) | beats baseline |
|---|---|---|---|---|
| log_reply | 0.3511 | 0.3511 | 0.001 | — |
| + `mean_rt_cos` | 0.3537 | 0.0026 | 0.402 | no |
| + `mean_act_norm` | 0.3537 | 0.0000 | 0.983 | no |
| + **`mean_instability`** | 0.3703 | **0.0166** | **0.029** | **yes** |

*(n = 194; coefficient +3.61, CI [0.148, 7.69].)*

`rt_cos` adds nothing once length is in the model — the same verdict HT12 reached, now reached a
third way and on a different outcome variable. `act_norm`, analysed against anything for the
first time, adds literally nothing (ΔR² = 0.0000).

**Read instability adds incremental fit that survives a permutation test.** These are genuinely
different objects: entropy is instability of the *answer* across sampled generations,
instability is disagreement among K verbalizations of one *fixed* vector. So this is not
circular — but it is a 1.7% ΔR² at p = 0.029, exploratory, and it does **not** survive into the
2×2, where the same measure sits at AUC 0.408 on matched wrong items.

### An incidental result that is stronger than any of the above

**Answer entropy separates right from wrong at AUC 0.843** (mean entropy 0.672 wrong vs 0.296
right). And the "confidently wrong" state barely exists at this temperature: only **6 of 120**
wrong items are perfectly consistent, against 56 of the right ones. On this corpus, being wrong
and being unstable nearly coincide — which is itself the reason the 2×2 is a relative contrast
between the most-committed and most-split wrong items rather than an absolute one.

## Verdict

- **The describability account of `rt_cos` survives.** It is a length-and-describability meter,
  not a confusion meter, and it now fails to predict entropy just as it failed to predict
  correctness.
- **`act_norm` is inert.** Worth recording so nobody spends GPU time on it again.
- **Read instability is the only internal signal with any incremental content**, and its status
  is "promising, unconfirmed": it wins the horse race and loses the 2×2.
- **Torn-ness, as an internally readable state, is not established.**

## Observations

1. **The measure that worked is the one whose construction matched the phenomenon.** `rt_cos`
   asks whether *one* sentence carried the vector; instability asks whether the vector *has* one
   sentence. Only the second is about ambiguity, and only the second showed anything.
2. **Gate 1b's failure narrows N11.** The trap does not induce measurable indecision. Combined
   with the echo result, the picture is that adversarial renaming changes what the model *says*
   without visibly destabilising it.
3. The instability values are small in absolute terms (0.012–0.024 in AR space). The floor gate
   is what makes them interpretable at all; without it these numbers would look like noise.

## Next steps

- **Do not claim the horse-race result.** It is exploratory, ΔR² = 1.7%, p = 0.029, and
  contradicted by the matched 2×2. The honest move is a frozen confirmatory test on held-out
  cases with the rule declared first — or to report it as a negative-leaning null with the
  incremental fit stated.
- `n_matched_pairs` for instability is **14**. Any confirmatory design must budget reads to
  raise that number specifically; the current stratified quota was not built for this contrast.
- Fold gate 1b into the B3/N11 write-up: it is independent evidence on the same question.
