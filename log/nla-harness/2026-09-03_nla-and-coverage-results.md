# 2026-09-03 — V1 ties V3 exactly, coverage changes nothing, and there is no regime where these writes help

**Thread:** nla-harness · **Jobs:** 374556 (N, 720 rows), 374629 (C, 600 rows) · 0 errors · **Host:** `gemma12b` L32/48 · **Resolves:** [`2026-09-03_nla-and-coverage-prereg.md`](2026-09-03_nla-and-coverage-prereg.md)

---

## 1. Experiment N — the pre-registered gate

**V1 ties V3 at every α tested.** Not "no significant difference" — an exact tie, three times, with symmetric discordant pairs.

| α | V1 acc | V3 acc | Δ | 95% CI | discordant | McNemar p | verdict |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.6167 | 0.6167 | **0** | [−0.05, +0.05] | 1 / 1 | 1.00 | tie |
| **1.0 (primary)** | **0.6333** | **0.6333** | **0** | [−0.0667, +0.0667] | 2 / 2 | **1.00** | **tie** |
| 4.0 | 0.6167 | 0.6167 | **0** | [−0.0833, +0.0833] | 3 / 3 | 1.00 | tie |

**N-0 ✗ — the NLA does no causal work a plain contrastive difference does not already do.**
B4's refutation **replicates on a second architecture**, with a different AR checkpoint, a
different layer (32 vs 20) and a different width (3840 vs 3584). This is the **first
permitted-host evidence** on the NLA's causal claim, and it agrees with the Qwen result the
2026-09-02 model constraint put permanently out of reach.

The sharpest single number: **V1 at the primary α has CI [0, 0]** — 0 recovered, 0 damaged. Not
"no net effect"; **no effect on any individual item**, while rewriting **96.7%** of the reply text.

Full battery, baseline L1b 0.6333, L0 0.7333, parse 0.800–0.867 throughout (nothing destroyed):

| condition | Δacc | 95% CI | rec | dam |
|---|---|---|---|---|
| V1_gloss | +0.0000 | [0, 0] | 0 | 0 |
| V3_taskvec | +0.0000 | [−0.0667, +0.0667] | 2 | 2 |
| A_antipodal | +0.0000 | [−0.0667, +0.0667] | 2 | 2 |
| V2_wordedit | +0.0167 | [−0.0333, +0.0833] | 2 | 1 |
| F_foreign | +0.0167 | [−0.0333, +0.0833] | 2 | 1 |
| prompting | +0.0167 | [−0.0333, +0.0833] | 2 | 1 |
| R_random | −0.0167 | [−0.0833, +0.05] | 2 | 3 |
| V4_oracle | −0.0500 | [−0.1167, 0] | 0 | 3 |

**`A_antipodal` is the diagnostic row.** It is the *negation* of V1 — built to be actively wrong —
and it is exactly as harmless (+0.0000). When the intended intervention, its opposite, a foreign
item's vector, and a prompt change are mutually indistinguishable, the result is about **the site**,
not the directions.

**V2 ran on the full 60 for the first time** (0 skipped, where Qwen items lacked a single-occurrence
term), so the V1-vs-V2 contrast finally has matched denominators. Swapping the whole gloss and
swapping one word do the same thing: nothing.

## 2. Experiment C — coverage

| positions | cond | α | acc | parse | Δ vs base | rec | dam |
|---|---|---|---|---|---|---|---|
| last_prompt (1 tok) | V3 | 1.0 | 0.6333 | 0.817 | +0.0000 | 2 | 2 |
| last_prompt | V1 | 1.0 | 0.6167 | 0.800 | −0.0167 | 0 | 1 |
| **id_spans (44 tok)** | V1 | 0.149 | 0.6333 | 0.800 | **+0.0000** | 0 | 0 |
| **id_spans** | V3 | 0.149 | 0.6333 | 0.783 | **+0.0000** | 0 | 0 |
| id_spans | R_random | 0.149 | 0.5833 | 0.767 | −0.0500 | 1 | 4 |
| id_spans | **V3 @ max-delivery** | 8.0 | 0.5167 | 0.717 | −0.1167 | **0** | 7 |
| id_spans | **V1 @ max-delivery** | 8.0 | 0.2500 | 0.533 | −0.3833 | **0** | 23 |
| id_spans | **R_random @ max-delivery** | 8.0 | 0.2000 | **0.283** | −0.4333 | 1 | 27 |

**C-0 ✗ at the interpretable magnitude — the site was not the limitation.** At equal injected
budget, moving the edit from 1 token to 44 identifier tokens changes **88–98% of the replies** and
leaves correctness at **exactly** baseline, churn 0/60 for both V1 and V3.

**C-2 FLAGGED at α = 8.** Parse rate falls to 0.283, reproducing P0.2's failure mode precisely
(destroying generation rather than under-delivering), and the movement scales with magnitude rather
than placement. The α=8 arm is therefore **not** evidence about site.

**The primary matching convention was UNREACHABLE**, reported and not silently replaced — see
[`2026-09-03_coverage-delivery-ceiling.md`](2026-09-03_coverage-delivery-ceiling.md). Identifier
tokens asymptote at **48.8%** of the reference delivery at any α, because a layer-32 write reaches
the answer position only through attention in layers 33–47.

## 3. The finding that survives all of it

**There is no regime in which these writes help.**

| regime | outcome |
|---|---|
| any magnitude where generation survives | nothing moves beyond the floor |
| magnitude large enough to move the metric | **only damage** |
| **items recovered, across every arm run today** | **never more than 2 of 60** |

Across N and C plus the KV-bypass run — 1 layer and 33 layers, 1 token and 44 tokens, NLA and
contrastive and oracle and antipodal and foreign and random, exact clean-state replacement, and
prompting — **not one arm ever recovered more than two items.** Too weak and the writes do nothing;
strong enough and they destroy.

Every arm rewrote 88–100% of the text. **The site controls wording. It does not control
correctness.** That is independent support for the readout hypothesis Phase 1b already favoured:
correctness is decided before this site.

## 4. The sensitivity gap, closed

Three of today's results were formally UNINFORMATIVE because no positive control fired — a genuine
weakness in how I designed the family, not a property of the models. The α=8 arm closes it: the
pipeline **detects a −0.4333 shift**, so it demonstrably has power.

Stated precisely, because the pre-registered clause required *"R_random moves while V1/V3 do not"*
and that exact pattern never occurred at any single α: **the measurement has power, and at every
magnitude where generation survives nothing moves.** The small-α nulls are therefore not a dead
instrument. They are bounded negatives.

## 5. A mechanistic detail that ties back to the AR gate

At α = 8, **V1 damages three times harder than V3** (−0.3833 vs −0.1167; parse 0.533 vs 0.717)
despite being the *informed* direction. That is what a vector nearly orthogonal to the model's own
contrast (**cos(V1,V3) = 0.0446**, 0.20% shared variance —
[`2026-09-03_gemma-ar-alignment.md`](2026-09-03_gemma-ar-alignment.md)) should do when pushed hard:
leave the manifold faster. Two independent measurements agreeing.

## 6. Bounds

- **One host, one layer, n = 60, one tier (L1b).** Gemma's L1b penalty is 10 points (0.7333 →
  0.6333), far milder than the papers' 21.25% collapse, so this may be a weaker trap than the
  phenomenon the programme is about.
- **Says nothing about whether identifier positions matter to the model** — only that a layer-32
  write there cannot move the answer position's correctness.
- The AV was never exercised; **G0 stage B remains unrun on Gemma.** V1/V2 need only the AR, so
  this is not blocked — but the round-trip validation is still owed.
- Baseline parse rate is 0.80, so ~20% of items never emit a parseable answer even unsteered; the
  censoring work (2026-08-30) bounds how much that matters.

## 7. Provenance

Jobs **374556** (N: 720 rows, battery 3.15 h + sweep 1.37 h) and **374629** (C: 600 rows, 2.07 +
1.08 + 1.42 h), both g-05-01/h200 class, 0 errors, no resume-key collisions. Scripts sha256-logged
in each job's stdout. AR `models--kitft--nla-gemma3-12b-L32-ar/snapshots/3d6901d8…`, layer 32,
d_model 3840, mse_scale 61.968 — cross-checked against the sidecar at load. `--deterministic` OFF.
Seed 20260724. Artifacts under `data/nla/p0/{nla_steer,coverage}/gemma12b/`.

## 8. New questions

- **Is the ~49% delivery ceiling a depth artefact?** The KV-bypass machinery already writes layers
  0..L; identifier tokens edited at *every* layer would reach the answer through the full stack.
  Cheap, and it would say whether the ceiling is about position or about depth.
- **Does the ceiling equal the attention mass on identifier tokens?** If so it is measurable
  independently from attention weights and connects Instrument 3 directly to Instrument 1's `A_id`.
- **Is L1b on Gemma too weak a trap?** A 10-point penalty may not be the phenomenon. Running the
  battery on L2/L3 — the relational route, where the surviving Llama positive lives — would say.
