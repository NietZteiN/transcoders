# Results — master index

*Last updated: 2026-09-07 · generated from the 120 dated entries in [`log/`](log/); every number here
is quoted from the entry that produced it.*

This is a reading index over the experiment ledger, not a replacement for it. Each row names the
entry that settles the question; corrections are shown as corrections, because several of the
headline numbers in this project were retracted by their own controls and the retraction is usually
the more useful fact.

**Interactive summary of the flagship result:**
[The Edit Bottleneck](https://claude.ai/code/artifact/e4c53ff9-4e05-4119-bec7-c0f7b95658c7)

## 1. State of play in one page

The project is **Instrument 3** of a three-instrument study of how code LLMs fail on obfuscated
code. Its charter experiments (E1–E8, SAE/transcoder feature and circuit analysis) are **not
started** — Phase 0 is still open, blocked on one measurement. Essentially all completed science
sits in the **Instrument-2 sidebar**, the NLA (Natural Language Autoencoder) harness, which grew
from a feasibility check into the project's main line and now has 115 dated entries.

What that line established, in order of how much weight it can bear:

1. **The NLA channel is causally faithful to 98.3 %** [0.967, 0.999]. Turning an activation into
   English and back costs ~1.7 % of its causal effect on behaviour. This is not reported anywhere in
   the NLA literature, which scores reconstruction rather than causal power.
2. **Editing the English does nothing.** The autoencoder's own read → edit → reconstruct → write
   protocol lands on a random vector's benchmark (+19.47 vs +19.60) and below prompting (+27.53).
   The failure is located precisely: not the site, not the operator, not delivery, not depth — the
   **edited text**.
3. **What the restored state carries is identifier *meaning*, roughly half, and *decoy removal*, the
   other half; program structure is worth ~1 %.** Replicated in two languages on 471 spans.
4. **None of it moves accuracy**, and on this corpus it cannot: only 6 of 60 items can flip at all,
   so a perfect rescue equals the support threshold.
5. **Every belief-steering lever tried has failed** — 8 arms, 16× α ranges, 1 vs 33 layers, 1 vs 44
   tokens, two architectures. Across all of it, **no arm ever recovered more than two of 60 items**.
6. **Prompting has beaten every steering intervention, four times.**

The through-line worth carrying into the write-up is methodological: **eleven distinct defects, and
not one of them was in the science.** Every one was a measurement definition, a denominator, or an
arm that silently wrote nothing — and on 2026-09-03 four headline verdicts in a single day were
artifacts of the readout rather than results.

## 2. Charter status — the committed instrument (E1–E8)

`docs/CHECKLIST.md` records `Status: Phase 0 in progress (scaffold built, smoke passing; no science
runs yet)`. **All eight hypotheses HT1–HT8 are `open`; not one has moved.** None of E1–E8 has a
single completed method sub-box, and the thread folders they call for
(`log/sae-features/`, `log/steering/`, `log/attribution-graphs/`, `log/state-binding/`,
`log/triangulation/`) do not exist.

| Exp | Tests | What it is | Status |
|---|---|---|---|
| E1 | HT1 | Semantic-capture feature diff, L0 vs L1b — the flagship | not started; **blocked on Q-norm** (below) |
| E2 | HT2 | Causal feature steering, ablate decoy / amplify true | not started |
| E3 | HT3 | Transcoder attribution graphs on L0/L2/L3 | not started; **hard prerequisite unmet** (below) |
| E4 | HT4 | Dispatcher state-binding probe across hops | not started (panel-agnostic; could run early) |
| E5 | HT5 | Cross-model-type feature comparison | Phase 2 stretch |
| E6 | HT6 | Cross-tier feature stability (survives L1, breaks at L1b) | not started; shares E1's forward passes |
| E7 | HT7 | Cross-instrument triangulation | not started; needs E1 + attention/ISF tables |
| E8 | HT8 | Feature correlates of the ~2048-token plateau | Phase 3 opportunistic |

**The two blockers are both real and both measured:**

- **E1 — the Q-norm question.** The only transfer measurement taken: a layer-16 base-trained Llama
  Scope SAE applied to *Instruct* identifier-token activations gives **FVU 0.506, cosine 0.773,
  mean L0 = 28** against a k target of 50 → verdict **INCONCLUSIVE**. Until this resolves, an E1
  feature diff cannot be told apart from a dictionary-transfer artifact.
- **E3 — token-length matching.** Attribution/patching between tiers needs token-aligned stimuli.
  Measured on the banked L0/L1b corpus: **0 of 60 pairs admit a 1:1 map**; only 27.5 % of 1,289
  `replace` blocks are length-matched, 88.9 % of replace tokens sit in mismatched blocks, a median
  **19.4 %** of each obfuscated prompt is unpatchable, and the obfuscated variant runs a median
  **+27** tokens longer. Root cause is stimulus-level: only **60 of 328** renames are token-length
  matched, and **180 of 328** originals are single tokens. **This corpus cannot be retrofitted** —
  E3 needs stimuli length-matched by construction. L2/L3 has never been measured for this.

Infrastructure that *is* done: env `transcoders-mi` pinned; **14 dictionary entries pinned with
verified shas, 12 model ids verified, zero 404s**; stimuli and the Papers 2–3 behavioural tables
symlinked (Dataset A 100 rows, Dataset B 250 rows, `paper2_trials` 31,711 rows); converter and span
resolution at **1.0000 on 8,034 spans** across two tokenizers. A 34-agent adversarial review of the
scaffold produced **28 confirmed findings (~11 root causes)**, all fixed — the worst would have
silently corrupted E1's score through a BOS off-by-one.
## 3. The readout programme (2026-08-04 → 08-26) — can the NLA *read*?

Host for this whole period: Qwen2.5-7B-Instruct, layer 20, released pair
`kitft/nla-qwen2.5-7b-L20-{av,ar}`. The question is whether a natural-language read of the residual
stream carries information about the model's *comprehension* — and the answer, repeatedly, is that
it carries information about the **text**, not about the model's success on it.

### 3.1 Instrument validation — the pair works, out of the box

| id | question | result | verdict |
|---|---|---|---|
| G0 | Does the released pair replicate, and read code at all? | Stage A 101/101 tokens, median rel-err on ‖v‖ **0.00121**; Stage B round-trip cos **0.8501** vs reference 0.8494; 44 code probes cos **0.70–0.96**, 0 CJK | ✓ supported |
| — | Faithfulness by token class (n = 5,090) | overall **0.871**; function name 0.934 · misleading name 0.879 · original name 0.878 · reasoning token 0.864 · neutral rename 0.847. Misleading vs original **+0.0009** [−0.004, +0.006] — **a decoy name reads as well as a real one** | descriptive |
| — | Positional arc | 0–20 % **0.9411** → 60–80 % **0.8332** → 80–100 % 0.8710; first-vs-worst **+0.114** [+0.108, +0.120] | descriptive |
| N13 | Is the mid-trace dip length or state magnitude? | neither cleanly: rt_cos × read length **+0.173**, × act_norm **+0.542**; the judge-free lexical measure is **flat** (0.6030 / 0.6123 / 0.6006), so "less describable" cannot be separated from "the reconstructor handles mid-trace states worse" | descriptive |

Tier accuracy out of the box was **non-monotone**, which is the paper's own pattern: pooled
L0 .65 · L1 .55 · **L1b .63** · L2 .63 · **L3 .53**.

### 3.2 Does the read predict comprehension? — five falsifiable tries, five negatives

| id | question | result | verdict |
|---|---|---|---|
| HT12 | Does read faithfulness predict correctness? | GEE β **−0.00022** [−0.0051, +0.0047], p 0.928 on the length-matched corpus. The banked corpus had said +0.00774 (p 4.3e-05) — restricting the *same reads* to 182 length-matched cases collapses it to +0.00351 (p 0.208) | ✗ refuted |
| HT13 | Does NLA↔CoT alignment fall after the error? | **instrument null** — judge-vs-judge κ_quad **0.049** against a ≥0.40 bar (one judge answered "2" on 91 % of items); shuffled-null AUC 0.757 passed, so the *task* is learnable and the *rater* is not | not adjudicated |
| HT14 | Does the answer appear earlier in reads on correct runs? | **92 % of cases never reach onset**, so both medians are undefined; own-answer hit 2.8 % vs foreign-answer null **3.6 %** | ✗ refuted |
| N4 | Can a wrong trace's first error be located mechanically? | flagship gate passed (804/804 spans) but coverage **41/117 = 35 %** against a ≥60 bar; **0 of 140** traces contain an internally false arithmetic claim | ✗ instrument fails |
| N4b | Do two independent localizers agree? | ±1-sentence agreement **7/18 = 39 %** (bar ≥60), r **+0.10**; the judge's own 3-seed self-agreement is 46 % | ✗ refuted |
| N9 | Do reads confabulate, and do they plan ahead? | **91.1 %** of 9,511 readings name a language, **37.2 % name the wrong one** (Python tasks 1.0 % wrong; **JavaScript tasks 79.9 %**). Look-ahead: own 0.5481 vs foreign 0.5319, excess **+0.0161** [−0.0217, +0.0533] | descriptive; planning ✗ |

### 3.3 What the reads *do* carry

| id | question | result | verdict |
|---|---|---|---|
| N10b | Do reads discriminate malicious from matched benign code? | item-level AUC **0.764** [0.717, 0.808]; **hard negatives only 0.757** [0.708, 0.805] (a 0.007 drop); length-confound AUC **0.477** | ✓ **supported** — the first positive in the programme |
| N10 | Do reads carry item-specific detail about *what* a malware sample does? | **null.** Foreign-read null: own 0.8743 vs null 0.9034 → excess **−0.029**; correct-family excess **+0.055** over a permuted null. Security framing inflates the generic malice rate 2.5× (0.175 vs 0.069) — framing, not content | ✗ refuted |
| N13 | Does answer entropy predict correctness? | **AUC 0.843**, and **0.869** under the hardened normalizer — this is the declared baseline any internal signal must beat | ✓ supported (incidental) |
| — | Are malware reads more faithful than obfuscated-code reads? | raw gap +0.0389, **length-adjusted +0.0152** [+0.0073, +0.0232] — 61 % of the raw gap was length | descriptive |
| N13 | Is there a "torn" internal state distinct from confidently wrong? | **null**: rt_cos AUC **0.491**, act_norm 0.442, instability 0.408, on a length-matched 2×2 (confound AUC 0.512). Decisive: instability ↔ rt_cos **r = −0.958 at class level** — it is describability measured twice | ✗ refuted |
| N13 | Does any internal signal beat reply length on answer entropy? | **withdrawn** — `mean_instability` ΔR² 0.0166 (p_perm .029) under the original normalizer became **p_perm .234** under the hardened one | void |

### 3.4 Belief steering — the whole lever fails, at every scale tried

| id | question | result | verdict |
|---|---|---|---|
| N11 | Do misleading identifiers install a false *belief*, or just a lexical echo? | echo. Read-level hit on the identifier itself **30.9 %**, decaying to **0.0 %** at 400+ chars. Item-level echo-free C2 9.0 % vs the foreign-read null CI **[0.02, 0.10]** — does not clear | ✗ not established |
| N11 | With a real denominator, does read-hit couple to *stating* the wrong algorithm? | 30.0 % (3/10) vs 8.3 % (2/24), Fisher **p = 0.138**; own 0.132 inside the null CI [0.026, 0.211] — second failure on independent data | ✗ refuted |
| B4/N12 | Does an NLA-derived direction beat a no-NLA task vector? | V1 **0.550** vs V3 **0.550**, delta **exactly 0**, McNemar **p = 1.00**. Prompting **+0.100**; random **+0.033** | ✗ refuted |
| B4 v2 | Anywhere across a 16× α range or a second site? | every \|Δ\| ≤ 0.033, every **q = 1.00**. General drift confound: mean Δacc across all 7 conditions rises with α, with **random +0.083 at α = 4** | ✗ refuted |
| B1 | Does the Qwen NLA read a *sibling* model (Coder-7B)? | **gate fail** — median rt_cos **0.694** (bar 0.70), control − subject **+0.170** (bar 0.05). Not a scale artifact: injection is direction-only | ✗ refuted |
| B5 | Do belief steering and attention reallocation compose? | **null** — not one of six contrasts excludes zero (both levers **+0.52** [−3.37, +4.38]); 164 programs × 1,930 cases × 6 cells. One seed; **no per-case rows persisted**, so it cannot be re-analysed | ✗ refuted |
| e5b | Does the attention-steering result hold at a second seed? | **sign flips** — family of 5 goes **−1.24** [−4.41, +1.41] → **+4.46** [+0.85, +7.09]; opaque_predicates swings 7.8 points. The B0 headline is not established | ✗ refuted |
## 4. Phase 0, the floors, and the diagnostic families (2026-08-27 → 09-03)

### 4.1 Phase-0 licensing — where can we intervene at all?

| id | question | result | verdict |
|---|---|---|---|
| P0.1 | Does the layer-20 task direction survive to layers 21–27? | min cos **0.277** against a 0.50 bar (decay 0.818 → 0.672 → 0.583 → 0.500 → 0.277) | **HARD** — reusing the vector above its layer is not licensed |
| P0.2 | Was the *injection channel* the bottleneck? | widening the write to every reply position gives **−0.4166** [−0.4833, −0.2000] against a required +0.10; parse rate **0.917 → 0.383** | **NOT CHANNEL-LIMITED** |
| P0.3 | Is layer 20 a live site for *any* intervention? | attention steering at [20,20] moves P@1 **−1.434 / −1.658** points against a seed spread of 0.794, same direction both seeds | **SITE LIVE** — a reproducible *degradation* |
| P0.4 | Is it a *depth* failure — does the oracle work at layer 13? | V4@13 − V4@20 = **−0.0667** [−0.1833, +0.0500], p 0.424; against its own baseline **0.0000** (5 recovered, 5 damaged) | **NOT DEPTH-LIMITED** |

Together these pick row 4 of the licensing table: **the site works but belief-shaped writes do not.**
That is what routed the whole programme to a readout paper with a bounded causal negative.

### 4.2 The reproducibility floor — three interventions, cause still unidentified

| id | question | result |
|---|---|---|
| R1 | Is the per-item greedy floor card-to-card? | **No — run-to-run.** Same physical GPU: L1b **0.9000**, steered V4 **0.8500**. Cross-node **0.9500** — *not worse*. Corrects P0.4's stated cause |
| H-det | Can deterministic kernels remove it? | **No.** Same card, deterministic: L1b **0.8333**; every pairing in the same **0.80–0.95** band. And deterministic mode *changes answers* — D1 vs A1 agreement **0.8000**, the lowest in the table → `--deterministic` stays off |
| H-trunc | Is it truncation at 1100 tokens? | **No.** At budget 2048 the band is unchanged (0.8833 / 0.8500 / 0.8833) — though the budget fixed censoring completely, parse **0.8833 → 1.0000** |
| H-read | Do *reads* carry the floor? | **No — reads are bit-identical**, max abs Δ **0.0** on 120/120 within-process, across processes, and across cards. The floor is **generation-only** |

Consequence: **±0.075** binds every graded-generation result (B4's gate, B5's cells, P0.3's tables);
`rt_cos`, `act_norm`, the P0.1 curves and the 14,619-row read table carry no such caveat.

### 4.3 What the residual stream says about correctness — five retractions and one survivor

This is the part of the record most worth reading carefully: **five positives were retracted by
their own controls.**

| id | claim as first reported | what killed it | status |
|---|---|---|---|
| depth gradient | correctness decodability rises monotonically to **L27** | relabelling at a longer budget moves the argmax to **L13**; the two label sets differ on **8 of 60 items**; four label treatments give L27 / L13 / L27 / L13 | **retracted** |
| tier effect | correctness more decodable on clean than renamed code, Δρ **+0.1383** | against each tier's *own* length baseline: L0 **+0.0431**, L1b **+0.0615** → Δ **−0.0184**. The whole gap was a baseline gap | **retracted** |
| L1b argmax advantage | position matters — `answer_line` beats `last_prompt` by **+0.1123** | bootstrap CI **[−0.0303, +0.2564]** includes zero; underpowered by construction at n = 60 | **refuted** |
| Qwen relational mechanism | the L2 residual signal is dispatcher *state tracking* | beats a length+static-complexity baseline by only **+0.0245** against a +0.10 bar. It is **static complexity**, and L2 is dispatcher indirection — **0 of 70 items contain a `switch`** | **refuted** |
| Gemma L2 | beyond the baseline at **+0.105** | under the strict `max(length, static, combined)` baseline: **−0.013**. The combined ridge scored worse than its own best component in **6 of 9 cells** | **artifact** |
| span probe | residual encodes dispatcher span count beyond code size, **+0.3413** | a repetition control the same day: residual − (size + repetition) = **+0.0647** against a +0.10 bar | **artifact** |

**The one survivor:** Llama-3.1-8B's L2 advantage. Discovery **+0.1507**, pre-registered replication
**+0.1062** against a frozen +0.075 bar, **p = 0.0149**, and it survives the repetition control that
deflated the Qwen analogue (repetition ρ at L2 is **+0.0005** / +0.0363). Scope is small and stated:
1 host of 3, 1 tier of 5, n = 58, on the *weakest* host (L0 accuracy 0.567 vs Gemma's 0.777). In the
full 3×3 matrix **8 of 9 cells reduce to a surface statistic.**

Two robust negatives came out of the same work. **Mid-reasoning positions are empty** — five
independent measurements (`rt_cos` p = 0.59; binary probe 0.4040; graded ρ **+0.0075**; L2 grid
0.5370) with a deflationary account: reply-position activations are about the reply, and the prompt's
structure lives in the KV cache. And **the incumbent read site is worth about as much as counting
tokens** — `last_prompt` beats a length-only baseline by **+0.043** (L0) and **+0.062** (L1b), and
its best layer is **not identifiable** across split halves (argmax agreement 0.00, against 0.50 at
`answer_line`).

### 4.4 The steering post-mortem (2026-09-03) — and the day four verdicts were artifacts

| id | question | result | verdict |
|---|---|---|---|
| KV bypass | Does a single-layer write leave the K/V entries below it unedited? | **yes, measured**: max \|Δ\| **0.000** at layers 0–20 vs **682.698** for multi-layer prefill | supported |
| — | *claim precision* | the prereg's "**never** changes what the model reads" is **an overstatement**. Accurate: a write at layer 20 reaches later tokens through layers 21–27 only — **21 of 28 layers, 75 % of the depth**, are bypassed (Gemma: 33 of 48). Filed before any result existed | correction |
| H-C1 | Does closing the bypass rescue the null? | **+0.0000** [−0.0833, +0.0833]. Not inert — reply identity vs the single-layer arm is **0.000 on all 60 items** (completely different text) | uninformative; *a rescue test needs something to rescue* |
| V5 | If the state simply *is* the clean run's state at every layer 0..L? | **−0.0167** [−0.0833, +0.0333] — the whole intervention class is bounded at this site | bounded negative |
| α sweep | What does a 16× α range buy? | **1.63×** in delivered perturbation. So "V1 does not beat V3 across a 16× α range" should read "…a range spanning 16× in α but ≈1.6× in delivered perturbation" | descriptive |
| coverage | Can identifier-span writes deliver as much as a direct edit at the answer? | `id_spans` asymptotes at **0.618 vs 1.268** = **48.8 %** of the reference; gain per doubling falls to +1.5 % — a ceiling, not a grid limit | descriptive |
| H-D1 | Is the denominator big enough to see a rescue? | **6 of 60** items flippable → perfect-rescue ceiling **+0.100** = the support threshold. **Every accuracy-level null that day was underpowered by construction** | supported |
| N-0 | Does the NLA direction beat a plain contrastive difference on a permitted host? | **ties exactly at every α** — at the primary α, CI **[0, 0]**, 0 recovered and 0 damaged, **while rewriting 96.7 % of the reply text** | refuted |

**Four same-day artifacts, named in the ledger:** R's ratio denominator (the metric is monotone in
damage, so a "belief" verdict fired on an arm whose gain came from the corrupt trace collapsing —
Δ log P(clean) **−0.485** vs Δ log P(corrupt) **−1.758**); S's computed-but-ungated veto (`S-LIVE`
with L0 specificity down to **−48.81** and greedy accuracy **0.767 → 0.000**, rescuing **1** item
across five splits); T's unreachable threshold (0.784 against a 0.80 gate **no item can reach**);
and G's per-token normalisation.

**The G_sum fix is the one to carry forward.** ρ(G, 1/reply-length) = **+0.889** — the per-token
readout was very largely a length meter, and normalising *injected* item-level variance
(CV 0.590 vs 0.353). It inverted one result and manufactured another: the renaming-dose hypothesis
went from **−0.166** ("renaming is not a dosage effect", withdrawn) to **+0.378** [+0.104, +0.606],
vindicating the pre-registered prediction; and `decoy_verbosity` went from q = 0.046 to **q = 0.508**.
What survives on the corrected readout is **`decoy_overlap` ρ_partial −0.382, q = 0.006** — decoys
that share terms with the truth cost the model less. Arm-level contrasts are unaffected, because the
denominator cancels within an item.
## 5. Experiment W — can the NLA channel *write*? (2026-09-03 → 09-07)

The largest single family. It asks whether the Natural Language Autoencoder's own protocol —
read an activation into English, edit the English, reconstruct — can steer the model, and if not,
which step fails. Host `google/gemma-3-12b-it` at layer 32 of 48, autoencoder pair
`kitft/nla-gemma3-12b-L32-{av,ar}`, write by norm-matched replacement at L1b identifier-span token
positions, readout `G_sum` (total teacher-forced log-probability of the clean reply, in nats).

Reference scale, measured on the same corpus: whole-prompt swap (clean prompt instead of obfuscated)
**+121.13** = the unit; prompting **+27.53**; a random unit vector at the same positions **+19.60**;
frozen support threshold **+12.11**.

### 5.1 Stage 0 — is the round trip even span-specific? (`NLA-LIVE`)

| measure | matched span | another span, same item | another item |
|---|---|---|---|
| mean-centred cosine | **+0.6807** [+0.6696, +0.6916] | +0.0525 [+0.0420, +0.0639] | −0.0011 [−0.0100, +0.0077] |
| raw cosine | +0.9877 | +0.9669 | +0.9646 |

60 items / 476 spans / 0 skipped / CJK 0.000 / 25 min. The centred metric was added as a
pre-registered secondary out of a worry that anisotropy was *manufacturing* a separation; it went the
other way — raw cosine was compressing a 0.68 effect into 0.02 of range. The cross-item null at
−0.0011 with a CI containing zero is the evidence that centring is correct rather than flattering.
This bears on every banked `rt_cos` in the project (`NLACritic.score` normalises but never centres).

### 5.2 Stage 1 — the protocol does not steer (`W-GENERIC-PERTURBATION`)

| arm | G_sum | 95 % CI | % of prompt swap |
|---|---|---|---|
| **W1_edit** — the NLA protocol | **+19.47** | [+16.51, +22.51] | 16.1 % |
| C1_roundtrip — unedited round trip | +8.84 | [+7.21, +10.51] | 7.3 % |
| C2_foreign — wrong content, same shape | +10.68 | [+9.04, +12.35] | 8.8 % |
| **C3_ceiling** — round trip of the **clean** state | **+37.91** | [+32.49, +43.47] | 31.3 % |
| *banked* random vector at these positions | *+19.60* | — | 16.2 % |
| *banked* prompting | *+27.53* | — | 22.7 % |

60/60 items, 1,480 positions over 476 spans, 198 editable (41.6 %), 1:43:09, 0 errors.
W1 lands on the random benchmark to within **0.13 nats** and below prompting. A real
content-specific component exists and is sub-threshold: W1 − C2 = **+8.79** [+6.35, +11.41] on 47/60,
and it scales with the number of editable spans (ρ = **+0.402**). No arm trips the behavioural veto.
**The headline was the arm nobody was watching:** C3 at +37.91 is nearly double the edited protocol
and above prompting — the channel can deliver; the *edited text* is what fails.

### 5.3 Fidelity — the channel is 98.3 % causally faithful

| arm | G_sum | 95 % CI | % of prompt swap |
|---|---|---|---|
| **C3pure** — clean state → English → back | **+44.94** | [+40.25, +49.90] | 40.3 % |
| **P_patch** — clean state written directly | **+45.71** | [+40.82, +50.88] | 41.0 % |
| C1r — round-trip null (obfuscated read) | +9.30 | [+7.45, +11.25] | 8.3 % |
| *banked* prompting, same 49 items | *+27.66* | — | 24.8 % |

Ratio **0.983** [0.967, 0.999]: turning an activation into English and back costs ~1.7 % of its
causal effect. C3pure's CI lower bound (+40.25) clears prompting outright — the first arm in this
programme to do so. Reported with its own caveat: the released pair is scored on reconstruction
(`2(1−cos)`), and 98.3 % causal fidelity and the raw cosine 0.9877 are **the same fact viewed twice**
(the operator writes ‖h‖·unit(v), so the arms differ by ~9°); they are never to be presented as
independent confirmations.

### 5.4 What the effect is made of

**Nulls (49 items / 375 spans).** The effect is item-specific but *not* span-specific:

| arm | G_sum | 95 % CI | % of `P_patch` | veto |
|---|---|---|---|---|
| `P_patch` — this span's clean state | **+45.71** | [+40.89, +50.78] | 100 % | pass |
| another span, **same item** | +39.94 | [+35.56, +44.72] | **87.4 %** | pass |
| another **item's** clean span | +15.18 | [+12.41, +18.05] | 33.2 % | pass |
| random unit vector | **−365.59** | [−510.93, −237.13] | −800 % | **TRIPPED** |

H-W9 (item specificity) **+30.53** [+25.75, +35.79] on 49/49 ✓; H-W12 (span specificity) **+5.77**
[+3.37, +8.43] on 36/49 ✗ against the +12.11 bar. Two instruments on the same positions give
opposite granularity — the round trip separates spans *representationally* (+0.68 vs +0.05) while
87.4 % of the *causal* work survives swapping in a sibling span. The random arm at −365.59
(accuracy 0.714 → 0.306) is the family's only veto trip and the proof that the operator alone
destroys, so every positive here comes from content.

**Dose (`W13-ACCUMULATIVE-TIER-GENERIC`).** One span recovers 0.225 [0.194, 0.257] of the full
effect against a 0.50 redundancy bar — accumulative, with diminishing returns (+10.29 → +17.95 →
+28.19 → +45.71 for 1/2/4/all spans; per-span yield 10.29 → 8.98 → 7.05 → 5.98).

| component of the +45.71 | nats | share |
|---|---|---|
| generic — any activation from another program | +10.10 | 22.1 % |
| + that vector being **clean** rather than obfuscated | +4.17 | 9.1 % |
| + it coming from **this item** | **+25.67** | **56.2 %** |
| + **correct span** assignment | +5.77 | 12.6 % |

**Source (`W16-MEANING-CARRIES-IT`).** Patching from other obfuscation tiers separates *meaning*
from *decoy removal* from *structure*. Final numbers on the repaired anchoring — 471 of 476 spans,
60 items, both languages:

| arm | G_sum | 95 % CI |
|---|---|---|
| `T_L0_all` — true names, original flow (anchor) | +41.52 | [+36.84, +46.21] |
| `T_L2_all` — true names, **flattened** | +41.06 | [+36.51, +45.82] |
| `T_L1_sub` — **nonsense** names, original flow | **+20.47** | [+17.31, +23.69] |

| split | n | gap (true − nonsense) | 95 % CI | L1/L0 |
|---|---|---|---|---|
| all | 60 | **+21.05** | [+17.80, +24.28] | 0.493 |
| python | 30 | +17.83 | [+13.71, +22.33] | 0.554 |
| javascript | 30 | **+24.28** | [+19.57, +28.84] | 0.437 |

Structure is worth **+0.46** [+0.07, +0.88] — flattened code with true names recovers ~99 % of the
clean state's effect at these positions. **Decomposition: meaning ≈ 51 %, decoy removal ≈ 49 %,
structure ≈ 1 %.** This replicates an earlier Python-only result (+21.82 [+15.52, +28.26] on 20
items / 131 spans) at 3.6× the span set and lifts its language bound.

### 5.5 The result that does not move: accuracy

Restoring the clean identifier representations at every locatable span, 98 % faithfully, shifts the
likelihood 41 % of the way to the clean reply and moves accuracy **0.714 → 0.694**. Partly a
ceiling — the corpus has only 3–6 items whose answer can flip at all — but the direction says the
adversarial trap's damage is **not confined to identifier positions**.

The flippable census (2026-09-07) tested whether another host has a real denominator:

| host | n | L0 acc | L1b acc | penalty | →wrong | →right | net |
|---|---|---|---|---|---|---|---|
| llama8b | 60 | 0.583 | 0.533 | 5.0 pts | **9** | **6** | 3 |
| gemma12b | 60 | 0.733 | 0.633 | **10.0 pts** | 6 | **0** | 6 |

Llama clears the ≥ 9 gate on paper and fails it in substance: its 6 *reverse* flips (obfuscation
making an item correct) sit inside the 0.85–0.90 greedy-reproducibility floor's predicted 6–9
disagreeing items. Gemma has zero reverse flips and twice the net penalty. **The accuracy question
is unavailable on either host at n = 60** — which is what retrospectively justifies `G_sum` as the
readout.

### 5.6 Running now — H-W31, which heads carry it

The write lands on *prompt* positions; `G_sum` is scored on *reply* tokens. Nothing carries a nat
between them except an attention head in layers 33–47, which makes "which heads" a pre-registrable
question. Every head and MLP of those layers (255 components) is patched between an unsteered run
and a steered one, both directions (sufficiency and necessity), plus a read-side knockout that
blinds one head to the span keys. Two identities gate the run — patching a run with its own
activations must change nothing, and patching *all* components must reproduce the full effect —
and a failure of either voids it. Top-k is selected on one half of the programs and scored on the
other, against uniform-random-k and depth-matched-random-k nulls.

Status: job 382366, h200. Smoke passed both identities exactly. Full run in progress; verdicts
(`W31-CONCENTRATED` / `DISTRIBUTED` / `INTERMEDIATE`, and same-circuit vs different-circuit between
the reconstructed and raw states) **are not yet in**. Pre-registration:
[`log/nla-harness/2026-09-07_head-mediation-prereg.md`](log/nla-harness/2026-09-07_head-mediation-prereg.md).
## 6. Coverage — what the corpus has and has not been asked

### 6.1 A stimulus fact discovered while writing this document (2026-09-07)

**L3 is not what the charter says it is.** `CLAUDE.md` describes the ladder as L0 original · L1
renaming (nonsense) · L1b adversarial renaming (decoy) · L2 flattening · **L3 stacked ("all at once
— the decisive test, where the two routes compete for one budget")**. Measured directly from
`data/stimuli/dataset_{a,b}_source.json`, on all 70 items carrying every tier:

| | fraction surviving into L3 | items with ≥ 1 |
|---|---|---|
| identifiers **L1b** introduces (the decoys) | **0.0000** | **0 / 70** |
| identifiers **L1** introduces (nonsense) | **1.0000** | **70 / 70** |

The flagship shows it plainly: L0 `fibfib(n)` → L1b `smoothArea(_lastNSecs)` → **L3 `fibfib(a)`**
with L1's parameter name and a dispatcher table. **L3 = L1 ∘ L2 (nonsense renaming + flattening),
not L1b ∘ L2.** The adversarial decoy never appears in the stacked tier.

Two consequences, neither previously recorded:

1. **The "two routes compete for one budget" test does not exist in this corpus.** If the two routes
   are the *adversarial* atom route (L1b) and the relational route (L2), no stimulus combines them.
   Whatever L3 measures, it is not the decisive competition the charter names — and any write-up
   inheriting that framing would be wrong about its own stimuli.
2. **The span converter mislabels L3.** `src/convert_stimuli.py` classifies a renamed span as
   `l1_neutral` if it matches `^(?:var|func|fn|v|f)_[0-9a-fA-F]{2,8}$` and `adversarial` otherwise.
   Python L3's `var_XXXX` names match and are labelled correctly; JavaScript L3's mangled names
   (`a`, `b`, `uepoi`) do not, so **1,327 L3 spans carry an `adversarial` label on code containing
   no decoy.**

Filed as [`log/nla-harness/2026-09-07_l3-is-l1-not-l1b.md`](log/nla-harness/2026-09-07_l3-is-l1-not-l1b.md).

### 6.2 What has actually been measured on L3

Only five kinds of number exist for the stacked tier, all behavioural or read-probe:

| measure | L3 value | note |
|---|---|---|
| accuracy, banked corpus | **.53** pooled (A .50 / B .54) | the **floor** of all five tiers |
| read-probe beats-length (discovery → replication) | **+0.0938 → +0.0418** | "the stacked tier does not carry the claim on its own" — the effect is carried by L2 |
| same, after the repetition control (Llama) | **−0.0845** | repetition is a *strong* predictor at L3 (+0.2531) and becomes the binding baseline |
| dispatcher-span probe, beyond code size | **+0.0830** — not beyond | a baseline artefact: stacking inflates code size, so the size baseline rises to +0.7893 |
| NLA round-trip faithfulness | **0.860** | the lowest-faithfulness tier |

**Not one** attribution graph, error-node mass, SAE feature measure, HCI, ISF, dispatcher-complexity
correlation, or item-level L2-vs-L3 comparison has ever been computed for L3 in this repo.

### 6.3 Tier coverage of the causal programme

Every B-, N-, P0- and W-family arm draws from `steer_run.load_pairs`, whose docstring reads
*"L0/L1b pairs sharing a snippet_id"*. **The entire causal and steering programme is L0/L1b-only** —
the clean anchor and the atom route. As the ledger puts it, the nulls "were not wrong; they were
**route-specific and nobody knew it**, because the relational tiers had never been read." The
cross-tier patching family did read L1 and L2 — but omitted L3 with no stated reason, despite all
five tiers existing for all 49 items.

### 6.4 The identifier-pairing hole

`rename_map` pairs an original identifier with its obfuscated form by **equal-length positional
matching**, and emits `?unpairedN` sentinels when that fails. Share of keys lost:

| tier | Python | JavaScript |
|---|---|---|
| L1 | 48.7 % | **97.3 %** |
| L1b | 41.7 % | **0.0 %** |
| L3 | 45.4 % | **91.8 %** |

The failure is large everywhere — the best cell still loses 42 % — and it **inverts between tiers**,
because JS L1 rewrites member access (`lst.length` → `c['length']`) and L3 stacks renaming on
flattening, while L1b is a pure rename. Repaired offline for the W corpus by `repair_pairs.recover()`
(precision/recall 1.000; coverage 375 → **471** of 476 spans). **Unrepaired everywhere else it is
used — including Instrument 1's identifier-level attention measures.** One banked arm substituted
the literal string `?unpaired0` on 11 of 60 items before a guard was added.

## 7. Constants and standing caveats

| constant | value | consequence |
|---|---|---|
| greedy reproducibility floor | **0.85–0.90** per-item agreement (**±0.075**) | binds every generated-text result; reads are bit-exact and carry no caveat |
| flippable denominator | **6 / 60** items (Gemma); Llama 9 →wrong but **6 →right** | perfect rescue = **+0.100** = the support threshold; accuracy claims are unavailable at this n |
| support threshold | **+12.11** nats = 10 % of mean `G_sum` **121.127** | but a *random vector* scores +19.60, so a positive must clear that and prompting (+27.53) to mean anything |
| readout | `G_sum`, not per-token G | per-token G is ~1/reply-length (ρ **+0.889**); use G_sum for anything item-level |
| answer entropy | AUC **0.843**, held out **≈0.84** | the baseline any internal correctness signal must beat |
| seeds | **2 is the floor, not the target** | one sign flip (+7.8 points on a single family) was found by adding a second seed |
| host | Gemma-3-12B-it / Llama-3.1-8B | Qwen hosts are excluded by standing constraint; **no number from the Qwen programme survives a host change** |
| dictionaries | 14 pinned, shas verified | E1 blocked on Q-norm; E3 blocked on token-length matching |

**Eleven defects, none of them in the science.** In order: sglang absent after the cluster move ·
`MAX_NEW_READ` 96 under a comment claiming 180 (22 of 22 reads truncated) · vectors discarded on a
false claim that cosines answered everything · H-W2 vacuous by construction · H-W2′ vacuous in the
opposite direction · `tier_anchor` fallback dilution · `--strict-anchor` stripping L2's only
anchoring path (an arm wrote **zero** positions and still produced a verdict) · 29 structural zeros
averaged into a live contrast · the ladder's position-count mismatch (caught by an automated
invariant, in a banked result) · the `?unpaired` sentinel · and the L3 composition above. The
harness answers are `arm_guard` (an arm that wrote nothing cannot score), identity assertions that
exit non-zero, and pre-registration with frozen thresholds.

## 8. Open questions, ranked by value per GPU-hour

1. **H-W31 — which heads carry the transported state.** Running (job 382366); decides where a second
   NLA layer would go.
2. **H-W28 — re-derive `rename_map` offline** as a sibling field, and tell Instrument 1. CPU-only;
   the current hole is 42–97 % and silently inherited.
3. **The L3 question, now that L3 is known to be L1 ∘ L2.** Either build a true L1b ∘ L2 tier (the
   charter's decisive test does not currently exist), or re-scope every L3 claim as
   *nonsense-renaming + flattening*. CPU to decide, corpus work to fix.
4. **H-W25 — fidelity and the null battery on the repaired 471/60 set.** The fidelity half falls out
   of H-W31's persisted vectors for free.
5. **H-W18 / E3 — is the relational route simply invisible at identifier spans?** Structure is worth
   +0.46 nats there, so L2's causal positions are probably elsewhere. Reached from the other side,
   this is E3's question.
6. **Q-norm**, which is the single blocker on the charter's flagship experiment.
7. **A corpus with a real denominator.** Every accuracy claim in this project is bounded by 6
   flippable items, and both available hosts are flat. Dataset B's 250 snippets are the stated
   prerequisite.

## Changelog
- **2026-09-07** — created, covering the 120 entries from 2026-07-24 to 2026-09-07. Records the L3
  composition finding (§6.1) discovered while assembling it.
