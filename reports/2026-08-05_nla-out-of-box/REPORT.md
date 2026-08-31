# Can NLAs Read Code Comprehension Out of the Box? — Full Report

*2026-08-05 · transcoders project, Instrument-2 line (`log/nla-harness/`)*
*Model: `Qwen/Qwen2.5-7B-Instruct`, layer-20 residual (d=3584) · NLA pair: `kitft/nla-qwen2.5-7b-L20-{av,ar}` (frozen, released checkpoints) · seed 20260724 throughout · env `nla-mi`*

---

## Executive summary

We asked whether a released Natural Language Autoencoder pair — trained generically, never on code — can read a model's internal comprehension of **obfuscated code** out of the box. Over two days: exact replication of the reference gates, ~5,400 verbalized reads across three probe rounds and one 380-task unsupervised capture run, on tasks with ground truth (output prediction across 5 obfuscation tiers, program slicing).

**Main findings:**

1. **The NLA reads code semantics, reliably at the theme level.** On unobfuscated code, reads at identifier tokens name the actual computation (at `fibfib` tokens: *"recursive Fibonacci sequence… fibfib(n) = fibfib(n-1) + fibfib(n-2)"*). On control-flow-flattened code, reads at the dispatcher variable say *"state machine", "iterative state transitions"*.
2. **Adversarial renaming is visible in the reads.** On the `fibfib`→`smoothArea` trap, reads at the decoy name adopt decoy semantics ("moving average", "sensor readings") while reads at recursive call sites recover the true structure ("three recursive calls (n-1), (n-2), (n-3)"). Quantitatively, the decoy shifts an AR-space Decoy-Read Margin by **+0.047 vs the L0 baseline** in the matched-pair probe.
3. **Reads over the model's own chain of thought track the reasoning**, including the moment of answering: answer-line reads always describe "completing the final answer in the required format" — with approximately-right but digit-unreliable content. Crucially, one wrong-answer case showed the read **echoing the model's own wrong answer, not the truth** — the instrument is faithful to the computation, not the world.
4. **At scale (300 graded runs), the tier-accuracy pattern reproduces the papers' non-monotonicity** on a model outside the original panel: adversarial renaming does *not* reliably hurt (Dataset A: L1b .80 ≥ L0 .70), while **stacked obfuscation is the floor** (L3 .50/.54). Slice prediction is genuinely weak (.33).
5. **Failure-mode dissection works.** The flagship wrong answer (653 vs 927) reads as *comprehension right, arithmetic wrong*: every CoT read describes correct Fibonacci-recurrence evaluation. The accuracy bit alone cannot make this distinction; the reads can.

**Main limitations:** themes are reliable, specifics confabulate (languages, libraries, digits); sub-word digit tokens read as noise; DRM gloss anchors support only relative comparisons; everything below is single greedy runs on one model at one layer.

---

## 1. Instrument and validation (G0)

**What an NLA read is.** The verbalizer (AV) is a fine-tuned copy of Qwen2.5-7B-Instruct that takes one residual-stream vector — layer 20 of the target model, at one token position — injected into its prompt, and emits 2–3 English snippets describing it. The reconstructor (AR) maps text back to a vector; `cos(AR(text), v)` is the round-trip faithfulness check. A read at position *t* reflects the model's **entire accumulated state while processing position *t*** (20 layers of causal attention over the whole prefix), skewed toward next-token anticipation — not a gloss of the input token itself.

**Replication gates passed exactly** after three environment fixes (transformers-5.12 `apply_chat_template` return type, patched at 2 call sites; a missing `<|im_end|>` in the reference reply layout; `ninja` for flashinfer JIT):

| Gate | Criterion | Result |
|---|---|---|
| Stage A (extraction) | per-token ‖v‖ vs reference transcript, 101 tokens | **0/101 over 2% tol**, median rel err 0.0012 |
| Stage B (full round trip) | AV read + AR score vs reference cos | mean cos **0.8501 vs 0.8494**, median \|Δcos\| **0.0037** |

Round-trip cos on *code* text (0.70–0.96 across all later probes) is comparable to the chat-text reference — code is not badly out-of-distribution for this NLA.

## 2. Feasibility probe (44 reads, flagship stimuli)

Stimuli from the obfuscation benchmark (Papers 1–3 scaffold; tiers L0 original / L1 neutral rename / L1b adversarial rename / L2 control-flow-flattened / L3 stacked).

- **L0 `fibfib`:** every identifier read names Fibonacci/recursion; one read reconstructs the recurrence nearly verbatim.
- **L1b `smoothArea` (same function, decoy name):** position-resolved mixture — decoy semantics at the definition site (*"smoothing algorithm", "Moving Average", "audio volume meter"*), true structure at recursive call sites (*"three recursive calls (n-1), (n-2), (n-3)"*, told through decoy-tinted names like "treeHeight").
- **DRM_AR** (cos to decoy gloss − cos to true gloss, AR space): L0 **−0.102**, L1b **−0.055** — both true-leaning in absolute terms (gloss anchors are crude), but the decoy pulls the representation **+0.047 toward decoy semantics**, matching the deck's E1 prediction, from n=12 reads per condition.
- **L2 dispatcher (`state_1395`):** word-token reads say *"state machine", "while count < threshold"*. Sub-word digit tokens (`1`,`3`,`9`,`5`) read as invented numerology — excluded from all later quantitative sets.
- **Consecutive-position scan** (44 adjacent tokens of the L1b prompt): content accumulates token by token — generic "code block follows" at `const`; decoy adoption as `smoothArea = (_lastNSecs` builds; and at the exact position where `== 0 || == 1` enters the prefix, **"Fibonacci" appears in a read for the first time** — the base-case shape is a Fibonacci signature strong enough to override the decoy name, ~25 tokens before the recursion itself is visible.

## 3. Interactive captures (galleries)

5 tasks (output prediction + hand-built slicing) with graded answers + 54 reads → `data/nla/examples/EXAMPLES.md`.

- **Answer-line reads (12):** at the tokens where the model commits to `Output:`/`Lines: […]`, the NLA **always reads the act of answering** ("concluding summary… completing the answer in the required format"). The values inside are lossy variants (true `[2,3,4]` read as "`[1,2,3]`, the sorted result").
- **Faithful-to-computation:** on a wrong slice answer (`[2,5,7]` vs truth `[1,2,4,5]`), the read reports *"the filtered indices are 3, 5, 7"* — echoing the model's own wrong answer rather than the truth. This is the property a faithfulness instrument must have.
- Also surfaced here: model replies truncate at small budgets (`fibfib(14)` exhausts even 900 greedy tokens — the papers' bounded-System-2 budget, reproduced incidentally); slicing needs an explicit print-line convention; grading must tolerate backtick-wrapped answers.
- **Bilingual incidental finding:** one read reported the model's internal conclusion phrased in Chinese (*"因此最终结果是：f(10) = 5…"*) — the NLA-paper "language switching" observation reproduced in the wild on a code task.

## 4. Overnight capture at scale

**Rig:** resumable unsupervised runner (`nla/src/overnight_capture.py` + task bank + tmux driver owning the SGLang server): per-task error rows, one server auto-restart, CJK injection-failure monitor, 11-h wall guard, heartbeats, per-task JSONL flush, provenance manifest. One operational incident: the first launch aborted after 19 min on a **CJK-monitor false positive** (any-CJK-anywhere matched benign reads quoting Chinese); detector redefined on the failure signature (mostly-CJK free association, fraction > 0.3) and relaunched with resume. The rerun then went **361 tasks / ~6.5 h unattended with zero errors and zero aborts**.

**Yield:** 380 rows (300 graded output-prediction: A 10/tier, B 50/tier × 5 tiers; 50 prompt-read-only; 30 synthetic slice tasks), **5,043 reads**, full CoTs, ground truth. `data/nla/overnight/2026-08-04/{captures.jsonl, summary.md, run_manifest.json, GALLERY.md}`.

**Accuracy by tier** (single greedy runs):

| tier | Dataset A (n=10) | Dataset B (n=50) |
|---|---|---|
| L0 original | .70 | .64 |
| L1 neutral rename | .50 | .56 |
| L1b adversarial rename | **.80** | .60 |
| L2 flattened | .60 | .64 |
| L3 stacked | .50 | **.54** |

- **Non-monotone, echoing Papers 1–3:** adversarial renaming does not reliably hurt (A-L1b beats A-L0), while stacked obfuscation is the floor in both datasets — on a model that was never in the original panel.
- **L1 ≤ L1b in both datasets** — unexpected; single-greedy-run noise vs real effect is undecidable here (K-sampled runs, i.e. N1's design, would decide).
- **Slice prediction .33** (10/30, convention stated in-prompt) — replicating the earlier 3-task result: the model over-includes the print line and drops initializer lines.
- **DRM_AR at trap identifiers:** L1b **−0.0146** (333 reads), L3 **−0.0349** (172) — true-leaning pooled means under crude template glosses; per-item analysis against matched L0 baselines is the meaningful next computation, not these pooled values.

**Case study (from `GALLERY.md`):** flagship L1b, model answers 653 (truth 927). Trap-identifier reads: mixed decoy/generic. All six CoT reads: correct Fibonacci-recurrence evaluation with worked examples. Diagnosis readable from internals: **the decoy did not fool comprehension; arithmetic execution failed.**

## 5. Limitations

1. **Theme-reliable, specifics-confabulated** — languages misattributed (JS read as "Python"), libraries invented, digits unreliable. Single reads must never be trusted for exact values; aggregate or use AR-space companions.
2. **DRM gloss anchors are crude** — absolute sign untrustworthy; only relative contrasts (tier vs tier, item vs item) are meaningful. Contrastive-activation anchors are the known upgrade path.
3. **Sub-word digit positions are noise** (excluded from quantitative sets).
4. **One model, one layer, greedy only:** everything here is Qwen2.5-7B-Instruct @ L20 (the only NLA-instrumented coordinates available), single greedy runs — data points, not conclusions. FVE ≈ 0.75 ceiling means each read loses ~25% of the vector's variance.
5. **50 Dataset-A Python rows are read-only** (no calls/ground truth by construction upstream).

## 6. What this enables

The banked dataset is exactly the raw material for the parked **N1–N3 faithfulness program** (design frozen in the plan file; HT9–HT11): N1 needs only K-sampling on top of this harness; N2 reuses N1's artifacts; N3's write-hook steering is the one unbuilt mechanism. Immediate cheap analyses available now, GPU-free: read-vs-correctness contrasts (do CoT/answer reads differ on wrong runs?), per-item DRM vs wrong L1b answers, and the L1-vs-L1b oddity at item level.

*(Instrument-3/SAE line status, for completeness: the E1 extraction path is live end-to-end on Llama-3.1-8B with all dictionaries pinned; the single open question before its full sweep is Q-norm — whether the Llama Scope FVU 0.506 on Instruct activations is a normalization mismatch or real transfer loss. See `log/infra/`.)*

## Artifact index

| Artifact | Path |
|---|---|
| Overnight raw data (380 rows / 5,043 reads) | `data/nla/overnight/2026-08-04/captures.jsonl` |
| Overnight summary + provenance | `data/nla/overnight/2026-08-04/{summary.md, run_manifest.json}` |
| Curated overnight gallery (12 cases, full reads) | `data/nla/overnight/2026-08-04/GALLERY.md` |
| Interactive gallery (5 tasks + answer-line reads) | `data/nla/examples/EXAMPLES.md` (+ `captures.json`) |
| Feasibility probe reads | `data/nla/probe/probe_reads.json` |
| Consecutive scan | `data/nla/examples/consecutive_scan.json` |
| Replication gate outputs | `nla/results/g0a_stage_{a,b}.json` |
| Runner / task bank / driver | `nla/src/{overnight_capture,task_bank}.py`, `nla/scripts/overnight.sh` |
| Extraction & probe scripts | `nla/src/{extract,probe_code_reads,capture_examples,answer_reads,scan_consecutive}.py` |
| Environment patch | `nla/patches/transformers512_chat_template.patch` |
| Session-by-session log | `log/nla-harness/` (6 entries) |
