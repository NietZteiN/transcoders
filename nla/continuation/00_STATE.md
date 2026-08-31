# 00 — Where the NLA research stands (as of 2026-08-31)

*Every number below is quoted from a dated entry in `transcoders/log/nla-harness/` or a report in
`transcoders/reports/`. Nothing here is re-derived. Where two sources disagree, the later dated
entry wins.*

**The canonical long-form write-up is
[`transcoders/reports/2026-08-26_nla-master/REPORT.md`](../../reports/2026-08-26_nla-master/REPORT.md)**
— ~1,490 lines, written to be read cold, with a table of contents. Read it if you need depth on
anything below. **§15 (Phase 9) and Part IV were rewritten 2026-08-31** and supersede this file
wherever they disagree. The published artifact at
`https://claude.ai/code/artifact/9c19873d-2cb3-441b-8fff-5420731f7fa3` is the **pre-Phase-9**
version and has not been redeployed.

---

## What changed since this file was written (2026-08-28 → 08-31)

**Phase 0 is fully decided, and one apparatus explanation was added and closed.**

| stage | verdict | the number |
|---|---|---|
| P0.1 | **HARD** | min cos(Δ₂₀, Δ_ℓ) = 0.277 vs a frozen 0.50 |
| P0.2 | **NOT CHANNEL-LIMITED** | V4 +0.0833 → −0.3333; contrast −0.4166 vs +0.10; V1 parse rate → 0.100 |
| P0.3 | **SITE LIVE** | [20,20] moves −1.434 / −1.658 vs a 0.794 seed spread |
| **P0.4** | **NOT DEPTH-LIMITED** *(new)* | the oracle at layer 13 nets **exactly 0.0000** — 5 recovered, 5 damaged — parse rate 0.833 intact |

→ **licensing row 4**, *the site works but belief-shaped writes do not*. Three pre-registered ways
of blaming the apparatus are now closed.

**A measurement floor was found, and it bounds part of this file.** Greedy decoding does not
reproduce per item: **0.85–0.90 agreement between two runs on the same physical GPU**, unmoved by
deterministic kernels or a doubled generation budget, cause unidentified. **Reads are bit-exact**
(120/120, max |Δ| = 0.0), so every read-side number here is unaffected; the floor applies to graded
generations only. Separately, at the banked 1,100-token budget the parse rate is 0.8833, so ~12% of
items in every absolute accuracy were scored wrong for **not finishing** rather than for being
wrong.

**The read side explains the whole item-level null ledger in one sentence.** At `last_prompt` — the
site of every item-level read in this programme — a linear probe on the raw residual stream beats
reply length by **+0.010** (+0.04–0.06 per tier against each tier's own baseline). B4, B5, N11, N13
and P0.4 were all measured where a linear decoder ties a token count.

**Corrections to figures quoted below.**
- **N13's answer-entropy positive is AUC 0.840 held out**, not 0.869. Both reported figures scored
  entropy against the plurality of the *same* eight samples; `banked_correct` is an independent
  greedy run and gives 0.8400 / 0.8428. The positive stands; the "hardening strengthens it" claim
  does not (+0.003 held out).
- **N10b is population-only.** Split-half reliability of the 14-read instrument is 0.581 / 0.613 /
  0.523 against a pre-stated 0.70 — it separates crowds and **cannot rank a single file**.
- **B6 is settled judge-free:** `rt_cos` adds nothing over reply length (LRT p = 0.27, ΔCV-AUC
  −0.0027, n = 512).
- **B5 now has its ledger entry** (`log/nla-harness/2026-08-26_b5-composition.md`).

**One route-specific positive, replicated and then deflated three times.** The residual stream beats
reply length under dispatcher indirection (L2/L3) and not under renaming — replicated on independent
draws at Δ = +0.3195 against a frozen +0.15 bar. It then resolved into **a signal about dispatcher
count largely predictable from how repetitive the source text is**: +0.0245 over static code shape,
+0.0647 over size-plus-repetition, and fully present at the prompt (position adds −0.0307). It is
the first internal correlate of the documented r = −0.196 dispatcher-complexity effect, and it is
**not** semantic understanding of control flow.

**Terminology fix:** L2 in this corpus is **dispatcher indirection**, not switch-based control-flow
flattening — 0 of 70 items contain a `switch`.

---

## The instrument

| | |
|---|---|
| subject model | `Qwen/Qwen2.5-7B-Instruct`, **layer 20 of 28**, d = 3584 |
| instrument | `kitft/nla-qwen2.5-7b-L20-{av,ar}`, **frozen — nothing was trained here** |
| seed | 20260724 throughout (B0/B5 grids use seed_base 1000 / 2000) |
| scale | ~20,250 verbalized reads, ~14,592 graded generations, **0 injection failures** |
| `rt_cos` | validated band 0.70–0.96; observed medians 0.860–0.886 throughout |

**A read is not a gloss of the token it sits on.** It describes the model's whole accumulated
state at that position. Separating "the model believes this sorts" from "the model just read the
letters b-u-b-b-l-e" is the hardest measurement problem in the programme.

**`rt_cos` is a describability meter, not a comprehension meter.** It correlates +0.28–0.31 with
how much text the verbalizer produced and does not predict correctness.

---

## The ledger — every experiment and its verdict

| id | question | verdict | the number that decided it |
|---|---|---|---|
| G0 | does the released pair replicate here? | **pass** | 0/101 over tolerance; median \|Δcos\| 0.0037 |
| N4 | can we localize the first wrong step? | unsolved | detectors agree 39%; 0/140 traces have false arithmetic |
| N5 | 4× read density without drift? | pass | 91/91 pairs, 4,653 reads, 232 reused match exactly |
| **N6 / HT12** | does faithfulness predict correctness? | **✗ refuted** | β = −0.0002 [−0.0051, +0.0047], p = .93 |
| **N7 / HT13** | does alignment drop after the first error? | **instrument null** | judge κ = 0.049; ρ = 0.032 vs free baseline |
| **N8 / HT14** | does the answer surface earlier when right? | **✗ refuted** | own 2.8% < foreign null 3.6%; 92% censored |
| N9 | how much do reads confabulate? | measured | 91% name a language, **37% wrong**; 79.9% wrong on JS vs 1.0% on Python |
| N10 | which capability does this malware have? | **null** | own 0.874 < foreign-read null 0.903; framing 2.5× |
| **N10b** | is this malicious at all? | **✓ supported, trimmed** | AUC 0.764 → **0.648** without the generic `payload`; misses 29.5% of malware |
| N11 / B3 | do misleading names induce a false belief? | **not established** | 30.9% *on* the name → 3.0% off it → 0.0% past 400 chars |
| N11-c | does a read track stated deception? | not established | 3/10 vs 2/24, Fisher p = 0.138 |
| **N12 / B4** | can a written-back belief recover accuracy? | **✗ refuted** | V1 0.550 = V3 0.550, p = 1.00, across a 16× α range |
| N13 | can we detect the model is torn? | **null** | AUC 0.491 / 0.442 / 0.408; instability ≡ describability at r = −0.958 |
| N13 *(free)* | does answer instability predict correctness? | **✓ positive** | **AUC 0.840 held out** (0.869 was same-sample) — better than any internal signal |
| B0 | does attention steering have a transform boundary? | **not established** | +7.02 at seed 1000 → **sign flips** at seed 2000 |
| B1 | does the NLA read a sibling model? | **✗ refuted** | rt_cos 0.694 vs host 0.864; loss is **directional, not scale** |
| B2 | does the stated belief beat a floor? | **unmeasurable** | only ~11 of 70 stimuli have a scorable true algorithm |
| B5 / V5 | do the two levers compose? | **null** | gate passed; every contrast's CI spans zero; collision check +0.10 |
| B6 | does read↔CoT agreement predict correctness? | not adjudicated | same judge instrument null as HT13 |

### The single pattern

**Theme-level: reliable. Item-level: absent.** Every null is an item-level claim; the one positive
is population-level. The mechanism is nearly always the same — a base rate high enough that a read
borrowed from *another item* does as well as the item's own (the **foreign-read null**, this
project's signature control).

---

## The causal arm, and why it is the live question

The steering work is consolidated as §12 of the master report. Key facts:

- **The ladder.** V1 `nla_edit` = AR(edited explanation) − AR(original) · V2 `word_edit` = the same
  but exactly one word differs · V3 `task_vector` = mean `h_clean − h_obf` over **other** pairs,
  **needs no autoencoder** · V4 `item_oracle` = this item's own difference (**not deployable**) ·
  V5 = V1 stacked on attention steering. Plus random / foreign / antipodal / prompt-only controls.
- **The bar is V3, not zero.** Any large perturbation changes outputs — a random norm-matched
  direction gains +0.083 at high α. If V3 matches V1, the autoencoder buys interpretability but
  not capability.
- **The result.** V1 0.550 = V3 0.550 exactly, at every α over a 16× range. **Reversing the NLA
  direction performs identically to a random one** — it has no consistent sign. **Adding one
  sentence to the prompt beat every steering condition (+0.100), including the oracle (+0.083).**
- **The mechanism, measured.** cos(V1, `h_clean − h_obf`) = **−0.001** [IQR −0.019, +0.021],
  indistinguishable from a random vector's −0.000, while cos(V3, V4) = **+0.702**. The belief
  direction is **orthogonal to the direction that matters**.
- **The injection site is tiny:** one vector, at **one token position** (`last_prompt`, the final
  prompt token), at **one layer** (20). The edit propagates only to layers 21–27; KV entries at
  layers 0–20 for that position are unedited, so every later token can still read the decoy
  through the bottom 21 layers.
- **One lead survives.** V2, the minimal single-word edit, is the only condition with a monotone
  dose-response (0.000 → 0.033 → 0.117 → 0.117 → **0.183**) and separates from V3 at α = 4
  (0.683 vs 0.550, p = 0.077). Exploratory, one cell of 25, **unconfirmable on a corpus hard-capped
  at 70 programs**.

### Why Phase 0 exists

B4 and B5's negatives are currently read as *"there is no item-level belief to edit."*
**That reading is not licensed**, because a second explanation predicts identical tables: the
injection **channel** cannot deliver anything, whatever it carries. The evidence sits inside B4's
own battery — **V4, the oracle that has seen the clean program, gained only +0.083, less than a
prompt sentence.** If ground truth can't clear a sentence through this channel, no direction was
going to, and the gate measured *delivery* rather than *belief*.

---

## Phase 0 — the triage that was running when we moved

Pre-registration (frozen before any run, copied to `artifacts/`):
`log/nla-harness/2026-08-27_p0-triage-prereg.md`

### P0.1 — layer rotation — **COMPLETE, verdict HARD**

Does the task direction `h_clean − h_obf` point the same way at other layers? 60 pairs, all 28
layers, read at the final prompt token. **No NLA involved — forward passes only.**

**Verdict HARD** — min cos(Δ₂₀, Δ_ℓ) over layers 21–27 is **0.277**, under the frozen 0.50
threshold. **Re-using the layer-20 vector across 20–27 is not licensed.** Decay is smooth
(L21 0.818 · L22 0.672 · L23 0.583 · L24 0.500 · L27 0.277), so a *narrow* 20–22 band is arguably
defensible — exploratory, not pre-registered. Below L18 the direction is essentially unrelated to
L20's (cos 0.016–0.374 across L0–L17).

*The layer-index gate passed at exactly 0.0 relative error* — `hidden_states[K+1]` is bit-identical
to the hooked extractor, so the layer axis is correctly labelled.

**The unexpected result, and it may be the most useful thing here.** Cross-item **coherence peaks
at layer 13 (0.543)**, not at the instrument's layer 20 (0.459), while **relative magnitude peaks
at layer 20**. Magnitude and coherence dissociate — the same shape N13 found for `act_norm` vs
`rt_cos`, in a completely different measurement.

**Consequence:** V3 *is* a cross-item mean direction, so it was built at the depth where the task
direction is **least shared**. That may be part of why even the no-NLA baseline only reached
+0.050. **V3 and V4 are pure activation differences and are defined at every layer**, so this is
testable with no autoencoder at all. That is the proposed **P0.4** — it needs its own frozen
decision rule before it runs. Full data in `artifacts/layer_rotation.json`.

### P0.2 — is the channel the bottleneck? — **PARTIAL, unscored**

`steer_run.py --positions all_reply` (many positions instead of one), conditions V4/V1/V3/random.
Stopped mid-run: **baseline complete 60/60, 300 steering rows on disk, not yet scored.**

### P0.3 — is layer 20 a live site? — **8 of 10 cells complete**

Attention steering restricted to explicit layer bands, belief channel off, two seeds.
Full scale = 492 runs / 5,790 cases / 164 snippets per cell.

| band | seed 1000 | seed 2000 | status |
|---|---|---|---|
| **baseline** (no steering) | 64.97 | 64.18 | complete |
| **[20,20]** layer 20 only | 63.54 | 62.52 | complete |
| **[20,21]** | 61.88 | 62.97 | complete |
| **[20,23]** | 62.80 | 63.97 | complete |
| **[20,27]** | 61.82 | 61.78 | **INCOMPLETE (~34% of snippets)** |

> **Do not apply the decision rule to this table.** The pre-registration requires the [20,27] cell,
> and it is partial on a non-random snippet subset. The shape is worth seeing — baseline is the
> highest cell, every steering band sits below it, and the baseline's own seed-to-seed spread is
> 0.79 points, comparable to the gaps between bands — but a flat curve reads as **UNINFORMATIVE**
> under the frozen rule, *not* as SITE DEAD. Partial data in `artifacts/p03_partial_cells.json`.

---

## Corrections found in a full audit (2026-08-28) — do not re-import the errors

1. **"identical activations" is wrong.** The N10 framing result is stated in the 2026-08-13 entry
   and both 08-14 reports as holding on *"identical code and identical activations."* The second
   half is false: the two arms are **separate captures** — a different preamble changes the prefix,
   so state and trace both differ, and the malware-browser builder verifies **0 of 14 read
   positions overlap**. What was shown is that changing the prompt moves the whole pipeline. What
   was *not* shown is that one fixed vector reads differently.
2. **Test count** was quoted as 64; it is **86 Python tests** across five modules plus 3 JS
   regression tests.
3. **Six results reversed** between a partial run and full n, not "twice": N11 coupling 3/3→3/10 ·
   B4 n=34→exact tie at n=60 · B0 family sign flip at seed 2 · B5 CIs spanning zero · N13
   instability p .029→.234 · N13's manipulation check +0.103 at 19 pairs → −0.011 at 54.
4. **B4's pre-registered injection-position sweep never ran.** `configs/b4_steering.yaml` describes
   the result as holding "at three injection positions"; the run manifest shows
   `--positions last_prompt` only. Trim that claim to the alpha range.
