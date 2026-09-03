# 2026-09-03 — KV bypass: UNINFORMATIVE by the frozen rule, and the reason is itself the finding

**Thread:** nla-harness · **Resolves:** the pre-registration filed today · **Job:** 372940 (g-05-01, 4.3 h) · **Host:** `gemma12b` L32/48

---

## 1. Verdict

**UNINFORMATIVE**, computed by `nla/src/kv_bypass_stats.py`, not typed by hand.

> Neither arm moved off the unsteered baseline by more than the reproducibility floor, so their
> agreement is not evidence for H-C0. Same clause as P0.3's flat-curve rule.

The short-circuit fired exactly as designed. Single-layer V3 did not move the task, so there was
**no effect for multi-layer steering to rescue**, and the registered contrast has nothing to
measure. H-C1 is **not supported**; H-C0 is **not established**.

## 2. All arms (n = 60, unsteered L1b baseline 0.6333)

| arm | acc | parse | Δ vs unsteered |
|---|---|---|---|
| prompting baseline (`P_prompt`) | 0.6333 | 0.800 | +0.0000 |
| single V3 @ α = 1.0 | 0.6333 | 0.833 | +0.0000 |
| single V4 oracle @ α = 1.0 | 0.5833 | 0.800 | −0.0500 |
| **multi V3 @ α = 0.1 (matched — primary)** | **0.6333** | 0.800 | **+0.0000** |
| multi V3 @ α = 1.0 (loud) | 0.6333 | 0.800 | +0.0000 |
| multi V4 oracle @ α = 0.1 | 0.6333 | 0.850 | +0.0000 |
| multi V4 oracle @ α = 1.0 | 0.5833 | 0.817 | −0.0500 |
| **multi V5 exact state replacement** | 0.6167 | 0.783 | −0.0167 |
| multi R_random @ α = 0.1 | 0.6000 | 0.783 | −0.0333 |
| **multi R_random @ α = 1.0** | **0.4833** | 0.783 | **−0.1500** |

Paired comparisons, floor band ±0.075:

| comparison | Δ | 95% CI | inside floor |
|---|---|---|---|
| multi V3 matched − single V3 | +0.0000 | [−0.0833, +0.0833] | yes |
| multi V3 loud − single V3 | +0.0000 | [−0.0833, +0.0833] | yes |
| V5 replace − single V3 | −0.0167 | [−0.0833, +0.0333] | yes |
| V5 replace − unsteered | −0.0167 | [−0.0500, 0.0000] | yes |
| prompting − unsteered | +0.0000 | [−0.0500, +0.0500] | yes |

H-C2 **not flagged**: the loud arm did not move while the matched arm stayed still, and no parse
rate collapsed (0.783–0.850 throughout, against P0.2's 0.100 failure mode). **The energy match
held.**

## 3. The intervention is delivered. It is not inert.

This is the part the accuracy table hides, and it is why the null is worth anything.

- **Reply identity vs the single-layer reference arm is 0.000 for every multi-layer arm** — all 60
  items, completely different text. The hook fires during generation, the edit propagates, and the
  model says something else entirely.
- **Random directions through the same machinery damage the task**: `R_random` at α = 1.0 gives
  **−0.15** (10 items damaged, 1 recovered, net −9).
- **Contrastive directions change every word and change nothing about correctness.** Multi-layer V3
  at matched energy churns **4 of 60** items (2 recovered, 2 damaged) — a 0.067 disagreement rate
  against a measured reproducibility floor of **0.10–0.15**. The steered run agrees with the
  baseline *more closely than two baseline runs agree with each other.*

So the null is not "nothing happened." It is: **a belief-shaped edit rewrites the entire answer and
leaves its correctness within run-to-run noise, while an equal-energy random edit does real damage.**

## 4. Where the frozen rule was too conservative — flagged, not overridden

The UNINFORMATIVE clause was written to catch a flat, insensitive measurement. **This measurement
is demonstrably not insensitive**: `R_random` at α = 1.0 produces a −0.15 shift, far outside the
floor, so the setup can detect a 15-point effect and did not see one from V3, V4 or V5.

By that argument H-C1 was refuted rather than untestable. **The frozen rule contains no sensitivity
clause, so the verdict stands as UNINFORMATIVE.** Recording this rather than quietly re-reading the
result: the rule fired as written, and rewriting a decision rule after seeing the data is precisely
the forking path this project has spent six retracted positives learning to avoid.

**Owed:** any revised rule adding a positive-control sensitivity clause must be pre-registered
*before* it is applied to these rows, and the re-analysis reported as a second, dated test.

## 5. What V5 does establish, and how narrowly

V5's rule was frozen in advance, so its reading is licensed — but it must be scoped correctly now
that the primary is UNINFORMATIVE.

Assigning `h_l := h_clean,l` at the steered position at **every layer 0..32** — the position's state
*is* the clean run's state, everywhere the KV cache is read from — moves accuracy by **−0.0167**
[−0.0833, +0.0333]. The strongest write available to this design does nothing.

**Licensed:** at this read site, on this host, the whole *intervention class* is bounded — not a
null of the instrument, since the instrument demonstrably moves the model.
**Not licensed:** any claim that the KV bypass specifically is or is not the explanation for B4.
That question needs a host where single-layer steering moves the task at all, and **neither Gemma
here nor Qwen in the banked B4 provides one.**

## 6. The methodological finding

**The bypass test is a rescue test, and a rescue test needs something to rescue.** Designed against
B4's null, it can only discriminate where the baseline intervention has a measurable effect. Both
available hosts are flat, so the design cannot discriminate on either.

Anything of this shape must in future carry an entry criterion — *the baseline intervention must
move the metric by more than the floor before the rescue arm is worth running* — checked before GPU
time is committed, not after.

## 7. Also: no method beat any other

The prompting baseline (mandated by charter §4, since prompting has repeatedly beaten steering —
AxBench, ICML 2025) lands at **0.6333, Δ = +0.0000**. Steering does not beat prompting; prompting
does not beat nothing. On this task and host, **no intervention tested moves correctness**, and the
only thing that moves it is random damage.

## 8. Provenance

- Job **372940**, node g-05-01, 2026-09-03, 4.3 h wall. 600 rows, **0 errors**, 600 distinct resume
  keys, no collisions.
- Scripts sha256-logged in the job's stdout: `steer_run.py` `311b9012…`, `steer_multilayer.py`
  `4fa762ca…`, `multilayer_vectors.py` `a024f676…`.
- Bank `(60, 48, 3840)` deltas + clean, verified against the live model's `d_model 3840, 48 layers`.
- Matched α = **0.1** by the frozen displacement rule (ratio 0.95). Seed 20260724.
- `--deterministic` deliberately OFF (2026-08-29: it changes answers and would fork the corpus).
- Artifacts: `data/nla/p0/steerv2/gemma12b/{kv_bypass_stats.json, energy_match.json,
  kv_bypass_gate.json, multilayer_bank.npz, run/}`.

## 9. New questions

- **Is there any host where single-layer steering moves this task?** Without one, the bypass
  question is not decidable by this design. Cheapest probe: run the single-layer V3/V4 battery on
  `llama8b` — the one host with a surviving positive elsewhere in this programme.
- **Why does the oracle damage (−0.05) while the exact replacement barely does (−0.0167)?** The
  normalised V4 write is a *larger* perturbation than the exact one; consistent with §3's picture
  that magnitude damages and content does not.
- **Does the 0.067 churn under complete text change mean correctness is decided before this site?**
  That is the readout hypothesis Phase 1b already favours, and this is independent support for it.
