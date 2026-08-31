# Do Models Believe the Lie? — belief readout and belief steering on obfuscated code

*Last updated: 2026-08-15*

*Plan v0.3. Supersedes the malware/NLA plan previously in this file (that work shipped; see
`transcoders/log/nla-harness/`). Revises the user's draft v0.2 against what the workspace
actually contains.*

> **Provenance and status (added 2026-08-15).** This is the paper's design document — the
> CodeSteer replication with belief steering replacing attention steering. It lived only in
> `~/.claude/plans/1-find-malware-dataset-imperative-kurzweil.md`, which was **overwritten in
> place** the same day by plan v0.4 (the N13 torn-ness plan), so the text below was recovered
> from the session transcript and is placed under version control here. It is reproduced
> **verbatim**; nothing has been edited to match later results.
>
> Read it alongside the current status of B0–B6, which has moved since it was written:
> **B3 run — not established** (the effect is lexical echo; `2026-08-13_n11-deception-results.md`,
> `2026-08-14_n11-coupling-followup.md`) · **B4 run at one α — gate failed**, V1 0.550 vs V3 0.550,
> McNemar p = 1.00 (`2026-08-14_n12-steering-gate.md`) · **B2 was skipped**, despite being the
> pre-registered gate for B3/B4 · **B0, B1, B5 not run** · **B6's judge route declared
> instrument-null** at κ = 0.049 (`2026-08-07_n7-instrument-null.md`). The malware line
> (N10/N10b) is not in this plan — it came from the predecessor plan and shipped separately.
>
> The execution schedule for the remaining work is a separate document; this file stays frozen
> as the design of record.
>
> **One clause below is now known to be unexecutable, recorded here rather than edited out.**
> Gate E1's decision rule offers: *"Norm band off but content sane → re-derive `injection_scale`
> from the Coder norm distribution, re-run."* That branch is a **no-op**.
> `nla_inference.normalize_activation` rescales every activation to a fixed L2 norm before
> injection, so magnitude is discarded and re-deriving the scale cannot change what the
> verbalizer sees. B1 measured the Coder's median `act_norm` at 154.0 against the host's 111.7 —
> a 38% difference that is normalized away. The transfer loss it was meant to rescue is
> directional, not scalar (see `log/nla-harness/2026-08-17_b1-transfer-refuted.md`).

---

## Context

The draft v0.2 proposes a CodeSteer sequel: mirror its RQ1–RQ4 on identical data, models and
metrics, replace the positional lever (attention steering) with a semantic one (NLA belief
steering), and add an internal-deception result. The instinct is right and the mechanism is
real. Three facts from this workspace change how it must be built.

**1. The instrument cannot be trained here, but it may not need to be.** The draft's E1 —
train an NLA for Qwen2.5-Coder-7B — is not a long pole, it is a wall. The GRPO RL stage needs
disjoint actor/critic/rollout GPU pools; a 2-way-sharded 7.6B actor needs ~58 GB/GPU and the
5.4B critic ~86 GB on one, against 48 GB cards, and gradient checkpointing is banned during RL
(NCCL deadlock). Skipping RL caps FVE at **~0.19 against the released 0.75** — a broken
instrument. Add ~$500–1,700 of API datagen, a 973-line Miles patch on a pinned commit, and
regex-anchored SGLang patches with no version pin.

The draft's route (b) — fall back to a released NLA on a different model — understates the
option. Qwen2.5-Coder-7B-Instruct shares architecture, `d_model=3584`, 28 layers and tokenizer
with Qwen2.5-7B-Instruct, so the released pair (already local at
`transcoders/nla/data/checkpoints/{av,ar}`) loads and asserts cleanly against Coder
activations. Whether it *reads* them is empirical, forward-pass-only, and answerable in hours.
**The NLA repo is silent on cross-model transfer**, so the measurement is itself a
contribution. This becomes gate E1 and sits on the critical path ahead of everything.

**2. The baseline being mirrored does not reproduce, and this project found the bugs.** From
`allocation_replication/LOG.md`: the artifact's recommended `slice_hybrid` prior is numerically
uniform (entropy/max = 1.000), so Eq. 9 reduces to the identity and the intervention does
nothing; with plain `slice` the cell gives **+24.8% [7, 41] against the paper's +104.99%**. The
76.49→40.20 identifier-renaming headline is a case-pack extraction bug —
`artifact/evaluation/java_counterfactual.py:117` hardcodes a regex for `correct`, their renamer
rewrites it to `v2`, and packs collapse to 2 trivial cases (2.00/snippet at 97% `if_throw`
origin vs 11.77 for originals). Their control-flow flattening has a brace-imbalance bug that
silently voided ~60% of that technique's evaluation. CruxEval R@1 must not be reported at all.

The consequence for scope is liberating rather than damaging: **since their stored numbers
cannot be reused as anchors, the argument that the subject model must match theirs collapses.**
Regeneration is required either way, which is why the E1 fallback below is cheap.

**3. The draft's RQ4 headline targets a result that was never established.** CodeSteer Table 7
is 80 cases per row; branch inversion's −8.96% is a −0.24-point recovery. This project's own log
already says it: *"CodeSteer's RQ4 negative result is not established… it needs a powered rerun
first."* Claiming to fix what was not shown broken is the one criticism that would sink the
paper, and it would come from a senior author who is in the group.

**Decisions taken (user, this session):** subject-model fallback is Qwen2.5-7B-Instruct if E1
fails; a powered rerun (**B0**) gates the RQ4 claim; all four additional obfuscation types get
ported to Java; and the steering arm is broadened beyond NLA explanation edits to include
**combined attention+belief steering** and **task-vector / contrastive difference vectors**.

---

## The spine

Obfuscation degrades comprehension partly by inducing a **wrong internal belief** about what a
program computes. Attention steering redistributes mass over tokens that already exist — as
CodeSteer itself states — so it cannot supply a fact absent from the prompt. NLA belief
steering injects exactly such a fact. The paper reads the belief, shows it predicts failure,
corrects it causally, and reports honestly how much of that correction needs the NLA at all.

**RQ labels are prefixed `B` throughout** to avoid the collision with `PROPOSAL.md`, whose
RQ0–RQ5 are different questions on the same corpus.

| | Question |
|---|---|
| **B0** | *(gate)* Powered rerun: does attention steering actually fail on arithmetic rewriting and branch inversion, at adequate *n* and ≥2 seeds? |
| **B1** | Does the released NLA read a *different* base model's residual stream? (transfer gate, and a standalone contribution) |
| **B2** | What does the model internally believe about a program under obfuscation, and does belief content separate correct from incorrect runs above the confabulation floor? |
| **B3** | Do adversarially misleading identifiers induce a *false* internal belief, and does it couple to errors? (internal deception) |
| **B4** | Does belief steering recover accuracy, and how much of the recovery survives against a no-NLA contrastive baseline? |
| **B5** | Does the recoverability profile invert against attention steering, and do the two compose? |
| **B6** | Does NLA↔CoT agreement predict correctness better than trace-level metrics and CoT length? |

---

## Gate E1 — cross-model transfer (do this first, it costs hours)

Nothing else is worth planning until this resolves. Forward passes only, one GPU.

Extract Qwen2.5-Coder-7B-Instruct L20 residuals on ~200 Java snippets from the existing case
packs, reusing `transcoders/nla/src/extract.py` (`ActivationExtractor`, hook on
`model.model.layers[20]` — do not re-derive the off-by-one). Run the released AV, score with
the AR. **Run the identical pipeline on Qwen2.5-7B-Instruct over the same inputs as the paired
control** — the comparison, not the absolute, is what decides.

| check | pass |
|---|---|
| `fve_nrm` on Coder activations | ≥ 0.55, and ≥ 0.75× the Instruct control on identical inputs |
| round-trip `cos` median | ≥ 0.70 (the established code-text band is 0.70–0.96) |
| activation norm band | Coder L20 median within ~2× of the Instruct band (~100–170); `injection_scale=150` was tuned for that band, and mismatch is the classic injection failure |
| CJK smoke | mostly-CJK reads < 1% |
| content sanity | reads on `fibfib`-style clean Java name the actual computation |

**Decision rule.** All pass → subject model is **Qwen2.5-Coder-7B-Instruct**, matching
CodeSteer's own primary model and this project's 244 completed runs. Norm band off but content
sane → re-derive `injection_scale` from the Coder norm distribution, re-run, report transfer
quality as a first-class limitation. Content degraded → **fall back to Qwen2.5-7B-Instruct**
(user's choice) and regenerate the CodeSteer conditions on it; this is affordable precisely
because their stored numbers were never reusable.

**Struck from the draft:** E1b's layer sweep. The released NLA is trained for layer 20 and
cannot be swept. What remains sweepable is the *steering injection position*, which is E4's
job. State the fixed-layer constraint as a limitation rather than discovering it later.

---

## The steering ladder (B4) — the user's task-vector addition, made into a control structure

Five vector sources at one layer (L20), α-swept, all sharing the write hook:

| source | Δ | what it isolates |
|---|---|---|
| **V1 NLA-edit** | `AR(AV_edited) − AR(AV_original)` | the NLA paper's method — belief correction via explanation edit |
| **V2 word-edit** | same, but a single word changed (`bubbleSort` → `binomial coefficient`) | whether one lexical item carries the belief |
| **V3 task vector** | `mean h_clean − mean h_obf` over *other* matched pairs | a deobfuscation direction needing **no NLA at all** |
| **V4 item oracle** | `h_clean(this program) − h_obf(this program)` | upper bound; uses the clean program, so **not deployable** — label it so |
| **V5 combined** | V1 applied on top of CodeSteer attention steering | do the channels compose (user's merge request) |

**V3 is the most important row in the paper.** If a contrastive task vector matches V1, the NLA
is buying interpretability but not capability — an honest, publishable finding, and exactly the
lesson `ar_baseline.py` encodes for the alignment score ("if a free dense baseline does as well,
the judge is buying nothing"). Pre-register that V1 must beat V3 to claim the NLA is doing
causal work.

Controls at matched norm, every one of them: random direction; foreign-item Δ; shuffled-text Δ;
sham layer (10 and 30); antipodal (−Δ must push the other way); prompt-only oracle. Success
statistic is **balanced Δaccuracy** including already-correct items, not flip rate — a vector
that makes the model shout "sorting" at everything scores brilliantly on flip rate.

Write hook: new `transcoders/nla/src/steer.py`, hooking the same `layers[20]` site
`extract.py` reads from, resolved through the same `_layers()` helper. Unit tests first:
α=0 reproduces the unsteered generation **byte-identically** under greedy decoding; the
activation at position *p* equals `h + α‖h‖Δ̂`; steering at *p* leaves *p−1* untouched. The
KV-cache position arithmetic (hook fires once on prefill, once per decode step) is the
bug-prone part.

---

## B3 — internal deception, graded mechanically

Three conditions per program: **C0** clean · **C1** neutral rename (existing
`identifier_renaming` variants — also the confabulation floor) · **C2** adversarial rename
(new: plausible *wrong-algorithm* names). C2 is built in
`allocation_replication/pipeline/obfuscation/rename.py`, which already exists and is
execution-validated; the new part is a wrong-algorithm label set. `obtune`'s `L1b` is the same
idea in Python/JS and is the design reference.

**The grading must not be an LLM judge.** This project already spent two GPUs and 5,653 judged
comparisons on a read-alignment score that separated populations at AUC 0.757 but could not
rank a single item (κ = 0.049, declared instrument-null). The fix that worked was a regex
against a ground-truth field the verbalizer never sees. Here that field exists by construction:
**C2 injects a known wrong algorithm**, so
`DeceptionHit(read) = 1[read names the injected algorithm or its canonical description]`
over a closed synonym set — mechanical, deterministic, judge-free.

Confound defenses, in ascending strength, all mechanical:
1. **C1 floor** — the same measure with no false content available to echo.
2. **Distance** — restrict to tokens far from the misleading identifier; echo is local, belief propagates.
3. **Recurrence** — count a belief only if it recurs at k ≥ 3 non-adjacent positions (the NLA paper's own heuristic).
4. **Behavioral coupling** — the rate must differ between correct and incorrect runs; constant echo predicts no difference.
5. **AR-ablation** — deleting the wrong-algorithm claim should hurt reconstruction if encoded, not if invented.
6. **Causal backstop** — if steering the belief flips outputs, the belief was load-bearing.

Nulls: foreign-read (a read from a different item at matched locus) guards the reader-prior
problem; permutation of the injected label within (technique × dataset) strata gives the chance
rate, which is *not* uniform because algorithm names are not equiprobable.

**Standing risk, stated up front.** NLA reads are theme-reliable and specifics-confabulated —
37% of readings in this project's banked corpus named the wrong *language*. "Which algorithm is
this" is a **specific** claim, not a theme. B2 must clear the confabulation floor before B3 and
B4 are worth running; that is the gate, not a caveat.

---

## Reusable vs build-from-scratch — the draft's reuse map corrected

| asset | status |
|---|---|
| Zenodo artifact, vendored | ✅ `allocation_replication/artifact/` |
| Java corpora | ✅ HumanEval 1,930 cases / 164 snippets; CruxEval 1,396 / 698 — exceeds the draft's 1,922 + 1,378 |
| 4 core obfuscation types, Java, execution-validated | ✅ `pipeline/obfuscation/` (240 checks, 0 failures) |
| Metrics + bootstrap CIs | ✅ `pipeline/analysis/metrics.py`, 8 unit tests |
| Authors' own obfuscated corpus | ✅ recovered, 3,395 files hash-verified |
| Completed runs on Qwen2.5-Coder-7B | ✅ 244 |
| Released NLA pair + validated harness | ✅ `transcoders/nla/` (G0 passed, 9,511 banked reads) |
| **3,242 execution-trace cases** | ❌ **do not exist** — code present, Stage 3 never run |
| **RQ3 trace metrics** (RP, DA, Stmt-Recall, Avg-Stmt-Mentioned, Cohen's d) | ❌ **zero implementation anywhere, including the artifact** — build + rubric calibration, not reuse |
| **4 additional obfuscation types in Java** | ❌ none; loop transformation and branch inversion absent from every pipeline here |
| **Second seed base** | ❌ everything is `seed_base 1000`; no cross-seed variance exists |
| NLA steering write hook | ❌ unbuilt |
| `obf_prompt` baseline arm | ❌ `pipeline/matrix/runner.py:62` raises `NotImplementedError` |

---

## Experiments, dependency-ordered

| # | Work | Gate | Rough cost |
|---|---|---|---|
| **E1** | Cross-model transfer gate (above) | thresholds table | ~3 GPU-h |
| **E0** | Consolidate replication: wire `obf_prompt`, add a **second seed base**, settle the identifier-renaming reporting stance | 2-seed variance reported | ~20 GPU-h |
| **E2** | Belief readout, clean vs 4 obfuscated types; confabulation floor; grader freeze | **belief signal > C1 floor** — else stop | ~15 GPU-h |
| **E3** | B3 deception: build C2 adversarial renames, measure | rate > floor **and** couples to errors | ~10 GPU-h |
| **S**  | Build `steer.py` + unit tests (CPU/1 GPU, parallel with E0–E2) | α=0 byte-identical | ~1 GPU-h |
| **E4** | B4 steering ladder V1–V5 + full control battery, core types | **V1 > V3** to claim NLA causality | ~30 GPU-h |
| **E5a** | Port 4 additional types to Java (`opaque predicates`, `loop transformation`, `arith/Boolean rewriting`, `branch inversion`) | execution-equivalence checks, 0 failures | CPU |
| **E5b** | **B0** powered rerun of attention steering on those 4 types, ≥2 seeds | denominators reported alongside every ratio | ~40 GPU-h |
| **E6** | B4/B5 on the 4 additional types — the headline cells | — | ~30 GPU-h |
| **E7** | B6 NLA↔CoT agreement vs trace metrics vs length | reportable either way | ~10 GPU-h |

**Cut unless the spine holds:** the 3,242 execution-trace corpus (Stage 3), RQ6's
buffer-overflow study, the Qwen2.5-14B scale check, and CruxEval R@1 in any form.

**Two project rules already earned here, carried forward:** never report a Restoration Ratio
without its denominator alongside; and no headline statistic without a null attached to *the
statistic that carries the verdict*, not to a descriptive companion.

---

## Verification

1. `pytest` on the new steer tests — α=0 byte-identical is the one that matters.
2. E1 transfer table with both models on identical inputs, plus the CJK and norm-band checks, written into `allocation_replication/LOG.md` and `transcoders/log/nla-harness/` before anything downstream starts.
3. Every run records exact command, config, commit hash, GPU ids, timestamp and **both seed bases** (CLAUDE.md §4; the single-seed gap is the largest current weakness in the replication).
4. Obfuscation transforms: execution-equivalence checks on all four new Java types, matching the existing 240-check standard.
5. Pre-registration entry, dated before E4 runs, freezing: the V1 > V3 rule, the balanced-Δaccuracy success statistic, the deception synonym set, and the refute clauses.

## What would make this null, stated now

B2 fails the confabulation floor because algorithm identity is a *specific* claim and specifics
confabulate — the most likely single failure, and it kills B3 and B4 with it. V3 matches V1,
making the NLA an interpretability device rather than a capability one (publishable, smaller).
Steering works only at α values that wreck fluency. B0 finds attention steering *does* recover
arithmetic rewriting once powered, removing the gap the paper aimed at — still publishable, and
better found by us than by a reviewer. Transfer fails outright and the subject model moves,
costing direct comparability with the 244 existing runs.
