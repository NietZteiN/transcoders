# Transcoders — Hypotheses, Experiments & Task Checklist

*Last updated: 2026-09-14*
**Status:** Phase 0 in progress (scaffold built, smoke passing; no science runs yet) · **Instrument 3** (SAE + transcoder feature/circuit analysis)

The living plan for the `transcoders/` sub-project. Master hypotheses ledger + experiment
tracker + phased to-do list. Source of truth for *what* and *why*:
[`docs/experiment_menu.md`](docs/experiment_menu.md) (full E1–E8 menu), the proposal deck in
[`docs/`](docs/), [`papers/REFERENCES.md`](papers/REFERENCES.md) (Papers 1–3), and
[`CLAUDE.md`](CLAUDE.md) §3–§4. Per-thread progress lives in
[`log/`](log/); this file is the master list those threads resolve against.

**Legend** — task boxes: `[ ]` todo · `[~]` in progress · `[x]` done.
Hypothesis status: `open` · `testing` · `✓ supported` · `✗ refuted` · `~ inconclusive`.
Tiers (from the menu): **T1** committed flagship · **T2** stretch · **T3** opportunistic/triangulation.

---

## 1. Hypotheses ledger

Eight transcoder-instrument hypotheses (**HT1–HT8**), each mapped to one experiment, one
Block-Model cell, and a specific prior finding from Papers 2–3 it must mechanistically explain.

| ID | Hypothesis (falsifiable) | Block-Model cell | Prior finding | Exp | Status |
|----|--------------------------|------------------|---------------|-----|--------|
| **HT1** | Under L1b, SAE features for the **decoy** semantics activate at renamed-identifier positions while **true-semantics** features are suppressed (vs L0); the Semantic-Capture Score (SCS) is positive on trap items and predicts per-item HCI/accuracy. | Atoms × Text-surface/Function | HCI collapse acc **21.25%** / HCI **10.0%** | E1 | open |
| **HT2** | **Ablating** the decoy feature (or **amplifying** the true feature) causally **recovers** L1b accuracy — monotone dose-response, beating both a random-feature-ablation control **and** a dense-steering/prompting baseline. | Atoms → Function | HCI failures; Stroop ×3.22 | E2 | open |
| **HT3** | A reproducible **dispatcher/state-tracking circuit** exists under L2 and **breaks** at L3 (more error-node mass, longer/broken paths), tracking the dispatcher-complexity gradient. | Relations & Macro × Program-exec | **r = −0.196** (q=3.1e-23); L2 OR 0.57 | E3 | open |
| **HT4** | The model holds a **linearly-decodable feature for the current dispatcher state** across hops; its fidelity **decays with hop count** and predicts output correctness. | Relations × Program-exec | r = −0.196; variable-binding / state-tracking lit | E4 | open |
| **HT5** | Human-aligned (**reasoning-tuned**, ρ=0.30–0.47) models share features (e.g. a task-difficulty / trace-length feature) that **non-aligned** (coder/instruct, ρ≈0) models lack. | all levels | **ρ = 0.30–0.47** vs ≈0 alignment split | E5 | open |
| **HT6** | A feature class is **robust to uninformative renaming (L1)** but **flips under misleading renaming (L1b)**; the decoy-captured fraction correlates with ISF/HCI. | Atoms × (Text-surface vs Function) | L1 vs L1b dissociation; L1b cancels by item | E6 | open |
| **HT7** | SAE feature-space measures **corroborate** attention (A_id/A_struct) and generation-uncertainty (ISF) measures — the three instruments **converge** on the H1–H4 verdict (or disagree interpretably). | cross-cutting | ISF; attention H1–H4 | E7 | open |
| **HT8** | Feature-space trajectories (dispatcher-state fidelity, decoy/true mass) **saturate at the same ~2048-token** budget where System-2 accuracy plateaus. | Macro × Function | **~2,048-tok** plateau | E8 | open |

### External hypotheses HT7 triangulates against (Instrument 1 — attention; from the deck)
Mutually exclusive predictions about how attention mass is allocated at **L3**. E7 tests
whether SAE features agree with whichever wins.

- **H1 Identifier-focused** — ↑ identifiers, ↓ structure (names are the last usable beacon).
- **H2 Structure-focused** — ↓ identifiers, ↑ structure (expert-like discounting of corrupted names).
- **H3 Reduced attention** — ↓ both; mass migrates to sinks/delimiters; entropy over code drops.
- **H4 Increased attention** — ↑ both (vigilance); entropy rises, no class-selective shift.
- **Bridge:** if H2 holds for reasoning models but H1 for coder/instruct, that mechanistically
  explains the ρ=0.30–0.47-vs-≈0 split (**HT5** is the feature-level version of this).

---

## 2. Experiments (E1–E8)

Committed instrument (menu recommendation): **E1 + E2 + E3 (PoC on a supported small model) + E7**,
with **E4** as a cheap panel-agnostic add. E5/E6 stretch; E8 opportunistic.

### E1 — Semantic-capture feature diff (L0 vs L1b) · T1 flagship · tests HT1
Feasibility: ✅ forward-pass only, **Llama-3.1-8B** (Llama Scope) / Qwen3 (Qwen-Scope). ~1–2 wk.
- [ ] Assemble matched L0/L1b snippet pairs with aligned identifier token positions (reuse Paper 2's SFR-embedding alignment).
- [ ] Extract residual-stream SAE feature activations at identifier positions (mid-layer; sweep layers).
- [ ] Auto-interp label features → define **decoy-domain** and **true-domain** feature sets (validate labels on the real checkpoint).
- [ ] Compute **SCS = (Δ decoy-feature mass) − (Δ true-feature mass)** from L0→L1b at identifier positions.
- [ ] Correlate SCS with per-item HCI / accuracy from Paper 2.
- **Confirm HT1:** SCS significantly positive on trap items and predicts HCI (top-SCS quartile shows the accuracy collapse). **Refute:** decoy features don't rise above L1 baseline, or SCS uncorrelated with HCI.
- **Risks:** auto-interp illusion (→ validate causally in E2); base→Instruct transfer; decoy/true sets may not separate (absorption/splitting → try **Matryoshka** / multiple widths).
- **First concrete item:** `fibfib` disguised as `smoothArea(_lastNSecs)`; `myFunct(14) → 927`.

### E2 — Causal feature steering (ablate decoy / amplify true) · T1 causal payoff · tests HT2
Feasibility: ✅ Llama-3.1-8B, Qwen3; intervention tooling exists (SAELens, circuit-tracer). ~2 wk.
- [ ] Using E1 feature sets, intervene during the forward pass: set decoy feature → 0, and separately scale true feature up.
- [ ] Measure Δaccuracy / ΔHCI on L1b trap items; report a **dose-response curve** vs steering coefficient.
- [ ] **Mandatory baselines:** random-feature-ablation control **+ dense steering-vector / prompting** (AxBench).
- [ ] Specificity check (RAVEL-style) for off-target effects.
- **Confirm HT2:** decoy-ablation / true-amplification produces monotone accuracy recovery exceeding random-feature ablation. **Refute / alternative result:** no recovery, or the dense/prompt baseline dominates — publishable **if pre-registered**.
- **Pre-register** the baseline comparison so a null is still informative.

### E3 — Transcoder attribution graphs on L0/L2/L3 · T1 transcoder core · tests HT3
Feasibility: ⚠️ mixed. Small supported models (Gemma-2-2B, Llama-3.2-1B, Qwen3-4B) ✅ Colab-scale; Llama-3.1-8B with a **trained CLT** ≈130–150 H100-hr (8×H100). Full panel: not feasible.
- [ ] Stand up `circuit-tracer` with CLTs; **PoC first** on a supported small model.
- [ ] Generate attribution graphs at the output-prediction token for matched L0 / L2 / L3 variants.
- [ ] Quantify graph divergence: node/edge overlap, path length to the answer logit, presence of a dispatcher-token → state-feature → branch path.
- [ ] Report **error-node mass** per graph (treat high-error graphs as low-confidence).
- [ ] Correlate a **dispatcher-circuit-integrity** score with dispatcher state count and accuracy.
- **Confirm HT3:** a dispatcher-state circuit appears under L2 and degrades at L3, tracking the r=−0.196 gradient. **Refute:** graphs dominated by error nodes, or no systematic L2→L3 change.
- **Risks:** error nodes carry causal weight (OOD regime); per-prompt & labor-intensive; matched-snippet graph comparison is **methodologically novel** (contribution, but risky).
- **First concrete item:** `isBalanced` control-flow-flattened into a while-if state machine (`s = 0/1/2`).
- **⚠️ PREREQUISITE, measured 2026-09-03 — stimuli must be token-length matched by construction.**
  Token-level activation patching between obfuscation tiers requires a 1:1 token map between the
  matched variants. On the banked L0/L1b corpus **0 of 60 pairs admit one**: only 27.5 % of 1,289
  `replace` blocks are length-matched, 88.9 % of replace tokens sit in mismatched blocks, a median
  **19.4 %** of each obfuscated prompt is unpatchable, and the obfuscated variant runs a median
  **+27** tokens longer (range −1 … +149). Root cause is stimulus-level: only 60 of 328 rename pairs
  are token-length matched, and 180 of 328 originals are single tokens replaced by multi-token
  descriptive decoys. **A patching-based E3 on L1b needs a tier generated under a length-matching
  constraint; an existing corpus cannot be retrofitted.** L2/L3 (flattening) has not been measured
  for this and must be checked the same way before any patching run.
  See [`log/nla-harness/2026-09-03_patch-alignment-impossible.md`](log/nla-harness/2026-09-03_patch-alignment-impossible.md).
  *(The related worry that the +27-token inflation confounds the L1b effect itself was tested and
  is **not** supported — ρ_partial = −0.159 [−0.420, +0.123]; see
  [`log/nla-harness/2026-09-03_length-confound-results.md`](log/nla-harness/2026-09-03_length-confound-results.md).
  The prerequisite above is about the patching **method**, not about the stimuli being invalid.)*

### E4 — Dispatcher state-binding probe across hops · T2 · tests HT4
Feasibility: ✅ cheap, **needs no pretrained dictionary** → runs on *any* panel model (incl. coder / reasoning-Qwen). ~2 wk.
- [ ] Train a linear (or SAE-feature) probe for the current state variable `s` at each dispatch site (Li–Andreas prefix-patching + probing; Wu–Geiger dereferencing).
- [ ] Track probe accuracy across hop depth; correlate decay with output accuracy and dispatcher state count.
- **Confirm HT4:** state-feature fidelity decays with hop count and predicts errors. **Refute:** state not linearly decodable, or decoding unrelated to accuracy (would echo "aggregate-at-query" entity-tracking result — itself interesting).
- **Caveat:** probing ≠ causal use (amnesic-probing); linear decodability may reflect info *present* not *used*.

### E5 — Cross-model-type feature comparison (reasoning vs instruct vs coder) · T2 · tests HT5
Feasibility: ⚠️ confounded by different dictionaries; clean version needs SAEs trained per model (coder: ~10–30 GPU-hr each). ~3–4 wk.
- [ ] Compare feature dictionaries / activation profiles on identical obfuscated snippets across model types.
- [ ] Restrict the *clean* comparison to models with SAEs (Llama-3.1-8B Instruct vs Qwen3 reasoning vs a coder model you train an SAE for), **or** use model-agnostic measures (probing accuracy for "difficulty", **RSA/CKA** between families) as primary.
- **Confirm HT5:** reasoning models exhibit a task-difficulty / trace-length feature absent in coder models. **Refute:** no consistent feature-level difference tracks the alignment split.
- **Risk:** "SAEs don't find canonical units" → prefer RSA/probing as primary, SAE features as illustration.

### E6 — Cross-tier feature stability (survives L1, breaks at L1b) · T2 · tests HT6
Feasibility: ✅ forward-pass only; Llama-3.1-8B / Qwen3. Natural companion to E1. ~1–2 wk.
- [ ] For each feature active at identifier positions on L0, measure activation persistence across L1 and L1b (Jaccard/cosine of activation patterns).
- [ ] Classify features: **lexical** (break at L1) / **semantic-robust** (survive both) / **decoy-captured** (flip at L1b).
- [ ] Relate the decoy-captured fraction to ISF and HCI.
- **Confirm HT6:** a well-defined feature class flips specifically at L1b and correlates with ISF/HCI. **Refute:** L1 and L1b indistinguishable at the feature level.

### E7 — Cross-instrument triangulation (SAE vs ISF vs A_id/A_struct) · T3 triangulation · tests HT7
Feasibility: ✅ pure analysis on already-collected activations; cheap. ~1 wk. **Run regardless — this is what makes the SAE work a genuine third instrument.**
- [ ] Per item, compute correlations among (a) identifier-feature mass (SAE), (b) A_id/A_struct (attention), (c) ISF (generation), (d) NLA verbalizer flags.
- [ ] Test whether SAE feature-space deltas Δ(L3−L2), Δ(L3−L1b) discriminate **H1–H4** independently of attention.
- [ ] Pre-register which agreement pattern supports which hypothesis.
- **Confirm HT7:** the three instruments agree on the H1–H4 verdict (strong result) or disagree interpretably. **Refute:** measures are mutually uncorrelated noise.

### E8 — Feature-level correlates of the CoT reasoning plateau (~2048 tok) · T3 opportunistic · tests HT8
Feasibility: ⚠️ reasoning-Qwen lack pretrained SAEs (use R1-Distill-Qwen-1.5B k-SAE or train); long-generation passes costlier. ~3 wk.
- [ ] On reasoning models, track dispatcher-state-feature fidelity (E4) and decoy/true mass (E1) as a function of generated reasoning-token count; test for a plateau.
- [ ] Cross-check whether internally active features match the displayed CoT (feature-level CoT faithfulness).
- **Confirm HT8:** feature-fidelity plateaus co-locate with the accuracy plateau. **Refute:** no plateau / no relationship.
- **Risk:** token-budget vs difficulty confound; SAEs on reasoning traces under-explored.

---

## 3. Task checklist ("things to do")

### Phase 0 — Infrastructure, data & dictionaries *(blocks everything)*
- [x] Create `src/`, `configs/`, `data/` (large artifacts → `data/`, never `$HOME` — [`CLAUDE.md`](CLAUDE.md) §2). *(2026-08-03: full tree + `scripts/smoke.sh`; see `src/README.md`.)*
- [x] Pin conda env `/data/jvl210002/conda_envs/transcoders-mi`; write `environment.yml`. *(2026-08-03: created + frozen (`environment.yml` exact pins + `environment.lock.txt`); circuit-tracer 0.5.2 installed (downgrades transformers→4.57 — documented); `pip check` clean; smoke green in-env; `scripts/env.sh` handles the CXXABI `LD_LIBRARY_PATH` fix.)*
- [x] Pull the reused **stimuli**: Dataset A + Dataset B. *(2026-08-04: located, symlinked, AND **converted** — `src/convert_stimuli.py` → 350 Snippet-JSONL rows with per-span classes and the cross-tier-derived decoy↔true `rename_map` (100% pairing on A-L1b; alignment parquets proved to be a different generation round — see `DATA_SOURCES.md`). Span→token resolution **1.0000** on Llama-3.1 + Qwen3 tokenizers.)*
- [~] Ingest the **behavioral tables** from Papers 2–3. *(2026-08-04: **located + symlinked** — `paper2_trials.parquet` (31,711 rows; `is_core==1` → 29,546), `paper2_adv_features.parquet` (ISF etc.), `paper2_dispatcher_cf.csv`, `paper3_human_graded.csv`, `paper3_model_results.xlsx`. Gotcha: HCI is **derived**, not stored — recipe in `DATA_SOURCES.md`.)*
- [x] Obtain **pretrained dictionaries** — identity + revision recorded. *(2026-08-04: all 14 registry entries pinned with HfApi-verified shas — Llama Scope LXR/LXTC (+widths), Llama Scope R1, EleutherAI ×2, Goodfire l19 (Instruct-trained!), AIRI, Qwen-Scope (official `Qwen/SAE-Res-*`), BluelightAI CLT, mwhanna PLTs (post-trained Qwen3 match), Chanin, Gemma Scope res/tc + circuit-tracer shim. Weights not yet downloaded — that happens per-experiment.)*
- [~] Build the **activation-extraction harness** (batched forward passes; cache under `data/`; GQA-aware; sink tokens tracked separately). *(2026-08-03: `src/extract_activations.py` built + smoke-tested end-to-end on tiny-gpt2 — config merge/seed/capture/safetensors/manifests all pass. Still TODO: real batching, GQA-aware attention handling, sink-token tracking, `apply_dictionary` once repo ids are pinned.)*
- [ ] Build the **auto-interp** feature-labeling harness (recall-biased — pair with causal/specificity validation).
- [ ] Build the **steering/intervention** harness (SAELens / circuit-tracer) + the **dense-steering + prompting baselines**.
- [ ] Build the **probing** harness (linear + SAE-feature probes) for E4.
- [ ] Wire results into the **existing GLMM stack** (binomial GLMMs, crossed random effects snippet×model, Wilson CIs, BH-FDR).
- [x] Seed + provenance logging scaffold (script sha256, GPU id, timestamp, dictionary id) — [`CLAUDE.md`](CLAUDE.md) §4. *(2026-08-03: `src/seedutil.py` + `src/provenance.py` → `run_manifest.json` per run; `src/gpu.py` enforces idle-GPU pinning before torch import.)*

### Phase 1 — Committed Tier-1 instrument (E1 · E2 · E3-PoC · E7) + E4
- [x] **Smoke-test** the E1 extraction path before the full sweep. *(2026-08-04: ran directly on the cached Llama-3.1-8B-Instruct itself — 10 real Dataset-A L0/L1b items, GPU 1, span resolution 1.000, `resid_post_L{12,16,20}` captured; L16 Llama Scope SAE encodes them at L0=28 / cos 0.773 / **FVU 0.506 → Q-norm open** (scale-convention vs real transfer gap) — resolve before trusting E1 feature masses.)*
- [ ] **E1** on Llama-3.1-8B → SCS vs HCI (open `log/sae-features/`). *(Blocked only on Q-norm; extraction + pairing + loaders all live.)*
- [ ] **E6** alongside E1 (shares the forward passes) → feature-stability classes.
- [ ] **E2** steering + baselines → causal recovery curve (open `log/steering/`).
- [ ] **E4** state-binding probe (panel-agnostic; can run early) (open `log/state-binding/`).
- [ ] **E3 PoC** attribution graphs on a supported small model (open `log/attribution-graphs/`).
- [ ] **E7** triangulation on collected activations (open `log/triangulation/`) — run as soon as E1 + attention/ISF tables exist.

### Instrument-2 sidebar (NLA — thread `log/nla-harness/`)
- [x] **G0 gate:** replicate the reference worked example. *(2026-08-04: Stage A 0/101 over tol; Stage B median |Δcos| 0.0037. Three fixes: transformers-5.12 `return_dict` patch, `<|im_end|>` reply layout, `ninja` for flashinfer JIT.)*
- [x] **Feasibility probe — NLAs out-of-the-box on code:** ✓ confirmed (44 reads). L0 reads name the true recurrence; L1b decoy/true mixture with **DRM_AR +0.047 toward decoy vs L0** (E1-direction signal); L2 reads say "state machine"; CoT reads track reasoning. Caveats: specifics confabulate; digit-piece tokens are noise.
- [ ] **N1–N3 faithfulness program (HT9–HT11) — PARKED downstream** by user decision. Full staged design (agreement metric, leakage controls, reranking baselines, 4 intervention levers, gates G1–G3) lives in the plan file `~/.claude/plans/let-s-make-a-claude-md-recursive-pond.md`; promote into this checklist when un-parked.
- [x] **Banked corpus** — 380 graded cases / 5,090 reads across all 5 tiers + synthetic slicing *(2026-08-05; tier accuracy non-monotone, reproducing the papers' L1b/L3 pattern out-of-the-box)*.
- [x] **N4 first-error oracle** (instrument) — flagship known-answer gate PASSED 4/4; 41/117 localized. **0/140 wrong traces contain internally false arithmetic** — the model computes the wrong thing correctly. Precise per-case localization **unsolved** (D2 vs D3 agree 39%; D3 self-agrees 46%).
- [x] **N5 dense capture** (instrument) — 91 length-matched wrong/control pairs, **4,653 reads**, ~4× read density, 0 errors, no instrument drift *(2026-08-06)*.
- [x] **Pre-registration of HT12–HT14** — decision rules frozen (BH-FDR as one family) before any of N6–N8 ran *(2026-08-06)*.

**Instrument-2 hypotheses (NLA faithfulness — HT12–HT14, pre-registered 2026-08-06):**

| id | claim | experiment | status |
|---|---|---|---|
| **HT12** | Reconstruction faithfulness (`rt_cos`) predicts per-case correctness after controls. | N6 | **✗ refuted** *(2026-08-07)* — length-matched β = −0.0002, CI [−0.0051, +0.0047], p = .93; an **informative** null that excludes the banked effect size. The earlier "correct runs are more describable" gap is **a reply-length artifact** (wrong traces ~1.6× longer). |
| **HT13** | CoT↔NLA judge alignment drops **specifically after** the first error (negative `correct × after_error` interaction; before-error simple effect's CI includes 0). | N7 | **not adjudicated — instrument null** *(2026-08-07)*. G1 shuffled-AUC 0.757 ✅, G2 beats the AR baseline ✅, but **G3 κ = 0.049** ❌: Phi-3.5 is a degenerate rater (91% one label, AUC 0.540) and judge-vs-AR per-item ρ = 0.032. The score separates populations but is not a stable per-item property. Exploratorily the interaction was **+0.023 (p=.22, wrong sign)** — a *global* late-trace decline, not a post-error one. |
| **HT14** | The answer surfaces in NLA reads earlier in correct runs (log-rank p<.05, median ≥0.10 `u` earlier) over a foreign-answer null. | N8 | **✗ refuted** *(2026-08-07)* — 92% of cases never reach onset, so the median is undefined and the criterion unmeasurable. Own-answer hit rate **2.8% < 3.6% foreign null**: the reads do not carry the answer. The significant log-rank (p=.002) is an **answer-commonness confound** (wrong answers rarer/longer, p=1.3e-14) — the null guarded the trajectory, not the onset statistic. |

- [x] **N7 (HT13)** — blinded judge alignment run: dataflow boundary verified (geometry gate 4,421/4,421), 5,653 items judged with 0 unparsed, shuffled + distant nulls, AR-space baseline, κ vs Phi-3.5. **Gate failed at G3 → instrument null.**
- [x] **N8 (HT14)** — answer onset via NLA reads: foreign-answer null, Kaplan–Meier + tier-stratified log-rank at **case level** (the HT12 group-structure trap was checked and avoided in advance).
- [x] **BH-FDR across {HT12, HT13, HT14}** — **vacuous, and recorded as such.** No p-value entered the family: HT12 was refuted by a pre-registered trigger, HT14 by an unmeasurable criterion, HT13 was never adjudicated. The absence of an FDR table is a result, not an oversight.
- [x] **Artifact F1/F2** *(2026-08-07)* — refuted HT12 claim **retracted in place**; verdicts section added; faithfulness sort shipped with an honest caption and **nulls sorting last in both directions**; **alignment sort deliberately absent** (HT13 instrument-null), with the page saying why. `cotHTML` rebuilt on a token-span layer, fixing a genuine **highlight-misplacement** bug (old code clamped spans → shifted right; OLD misplaces 4/5, NEW 0/5). Bidirectional linking scrolls only the un-clicked pane; `role="button"` key handler added. Verified: 330 cases / 2,828 marks, 0 lost, 0 text corruption. Builder now version-controlled at `nla/tools/build_page.py`, tests at `nla/tests/`.

- [x] **N9 confabulation rate** *(2026-08-07, exploratory)* — the standing "themes reliable, specifics confabulated" claim **measured**: 91% of 9,511 readings name a language, **37% wrongly**, and it is a **Python prior** (79.9% wrong on JavaScript vs 1.0% on Python), structured by read kind (code tokens 12% vs reasoning prose 46%). Round-trip faithfulness barely catches it (43.9% → 31.4% across quintiles; Q5−Q1 −12.5 pts, CI −18.6 to −6.5) — the sharpest measured demonstration that **recovery ≠ truth**. **Look-ahead/planning refuted** against a foreign-reading null (0.548 vs 0.532, CI spans zero).

**Verdict summary — the Instrument-2 faithfulness family closed negatively, but informatively.** All
three hypotheses failed against a *named mechanism* rather than for want of data, and in each case
the mechanism was caught by a control written down in advance: reply length (HT12), answer-string
commonness (HT14), and per-item unreliability of the judge score (HT13). The instrument-level
findings that survive are the useful ones — NLA reads are theme-reliable but do not carry the
model's eventual answer at L20, round-trip faithfulness tracks position/length rather than
comprehension, and LLM-judged step alignment separates populations without ranking items.

#### ASE-2026 block — CodeSteer comparison (thread `log/nla-harness/`, hypotheses tagged "(ASE)")
User's standing ask: *test a variety of steering methods and see if we can match or surpass CodeSteer*
(the ASE-2026 paper's post-hoc attention steering), then *see the heads*. Host for everything here is
**CodeLlama-7b-Instruct** — the only permitted model with damage worth recovering (H-R1: +0.113).
- [x] **H-R1 (ASE) replication** — their batched T/F protocol reproduces (1 930 cases vs their 1 922); their −36-pt renaming damage does **not** on any permitted model (Llama-3.1-8B +0.030, CodeGemma-7B +0.076, CodeLlama-7B +0.113). *(2026-09-14)*
- [x] **H-R2 (ASE) accuracy bake-off** — *scored 2026-09-15: `MATCH-CODESTEER` (`prompt` 0.634 vs `codesteer` 0.608, +0.025, α/3 CI ∋ 0), `SLICE-IRRELEVANT` (+0.023), `swap_oracle` 0.735 the only arm outside the ±0.14 noise floor; H-R2c unreadable on a +0.025 denominator; H-R4 refuted. `log/nla-harness/2026-09-15_bakeoff-results.md`.* Original spec: — 50 renamed snippets, 3 sampled runs, case-weighted Pass@1, all arms through *their* runtime: `unsteered` · `codesteer` (README-exact, β 0.8, last 8 layers) · `codesteer_auto` (their calibrated head subset, Eq. 10) · `rand_prior` · `uniform_prior` · `prompt` · NLA residual arms `swap_oracle` / `foreign` / `erasure` / `combined`. H-R2a best-of-ours vs the better paper reading (α/3 interval), H-R2b prior specificity, H-R2c restoration vs `original_unsteered`. Noise floor from the β = 0 identity runs: +0.067 [−0.078, +0.210] — any `MATCH` is resolution-limited. Jobs 399685–399863, 400250–400254.
- [ ] **H-R5 (ASE) — the heads: how many, which, and are they CodeSteer's?** *(added 2026-09-14 at user request; H-R2 scored 2026-09-15 — comparator `codesteer` 0.608 / `codesteer_auto` 0.576 (calibrated heads not better, −0.032); unblocked.)*
  **Head score = how much the head changes the NLA.** For each attention head *h* of the host, with the NLA write in place (the `erasure` arm's vector at the identifier spans — the deployable one — and `swap_oracle` as the ceiling), the per-head score is the change in the NLA-transported effect when *h* is blind to the write:
  `nec_h = dG(S) − dG(S, h ← U)` and `suf_h = dG(U, h ← S)` (the H-S2/H-W31 activation-patching definitions, `nla/src/head_patch.py`, `nla_head_sweep.py`) — U = unsteered renamed run, S = the same run with the write, G = teacher-forced log-prob of the original-condition reply. Secondary, read-side: KL of the answer-token distribution with vs without *h* (what the head does to what the model *says*, not only to the score).
  Three frozen readouts:
  1. **How many heads are needed** — greedy top-*k* by `suf` selected on one split half (crc32 parity), scored on the other; *k\** = smallest *k* whose top-*k* patched together reaches **≥ 90 %** of the full write's effect, against a layer-matched random-*k* null (H-S2's `LRAND_k`). Compare *k\** with CodeSteer's fixed budget (**4 heads × 8 layers = 32 of 1 024** on CodeLlama-7B).
  2. **Are they CodeSteer's heads?** — CodeSteer's set per snippet is *observational*: top-4 heads per layer in the last 8 layers by `agree_h = Σ_k P_last[h,k]·prior[k]` at the first decode step (`steering/runtime.py:448`), already stored per snippet in `bakeoff_codellama7b/codesteer_auto.jsonl` (`codesteer_heads`). Readout: Jaccard and rank-ρ between our top-*k\** and their 32 at matched *k*; plus the **window test** — the share of our top-*k\** that lies in their layers 24..31 at all (H-S2 on the 4B found the carriers at L23 of 34 and H-W31 on the 12B at L41/L46 of 48 — mid-to-late, not last-8, so the pre-registered prediction is **`DIFFERENT-HEADS`**: Jaccard < 0.25 and < half of our heads inside their window).
  3. **Do their heads move the NLA?** — patch CodeSteer's 32 heads with the write (`suf` of their set) vs our top-32 vs random-32: if their set carries < 50 % of what ours does, the two methods act through different heads; if ≥ 80 %, the same circuit is reached by two selection rules.
  Cost: 32 layers × 32 heads = 1 024 components × 2 patched forwards × 50 snippets ≈ 100 k forwards at the measured ~40 ms → ~1.2 GPU-h + joint stage; one A6000/H100. Identity gates as in H-S2 (SELF 0.000, ALL reproduces S). Prereg entry before running.
- [x] **H-R6 (ASE) — a better deployable vector (user request 2026-09-14: "find a better vector across the dataset, use NLA").** Diagnosis: `foreign` 0.574 ≈ `erasure` 0.576 ≪ `swap_oracle` 0.735 — erasure-shaped writes are bounded at ≈ 0 (tier ladder `ERASURE-FLOOR`; W16 meaning ≈ 51 %), so the vector must **install meaning** oracle-free. Meaning source = the renamed Java's own declared **type** + AST **role** (`nla/src/ase_roles.py`). Fit on the ~106 non-test snippets, written to the 50 test snippets — no LOO on the test set. Arms: `ridge_map` (reduced-rank ridge `h1b → h0 − h1b`, nests `erasure` at rank 0; pre-GPU gate = held-out cosine margin ≥ 0.05 over the mean-delta model), `role_proto` (type|role-matched clean-state prototype; ladder `foreign < role_proto < swap_oracle`), `prompt_types` (same facts as text — the §4 prompting baseline). Rules: H-R6a `MAP-BEATS-MEAN` (≥ +0.05 over `erasure`, CI > 0) · H-R6b `CATEGORY-MEANING-HELPS` (≥ +0.05 over `foreign`) · H-R6c `LATENT-BEATS-PROMPT` / `PROMPT-SUFFICES`. Predictions: a in activation space only, b small, c `PROMPT-SUFFICES`. Deferred: `swap_guess` (host's own per-identifier guesses → de-obfuscated prompt's states; partner = that prompt as text) and `ar_role` (an NLA AR on CodeLlama-7B L7 writing `AR(type + usage description)`, ~1 day port + ~20 GPU-h — held until a–c show latent headroom over prompting). Prereg [`log/nla-harness/2026-09-14_better-vector-prereg.md`](log/nla-harness/2026-09-14_better-vector-prereg.md) · config [`nla/configs/ase_vectors.yaml`](nla/configs/ase_vectors.yaml) · pool job 401200. *Scored 2026-09-15: a `MAP-NOT-BETTER` (+0.025), b `CATEGORY-MEANING-INERT` (+0.076, CI ∋ 0), c `PROMPT-SUFFICES` against the intact `prompt` arm (+0.016; the registered `prompt_types` baseline collapsed on format). `role_proto` 0.650 / parse 0.998 (per-case parse 0.843 — the 0.998 counted phantom ids, see `2026-09-15_parse-rate-correction.md`). `log/nla-harness/2026-09-15_better-vector-results.md`.* *Re-scored `c/n` 2026-09-15 (H-R9): verdicts unchanged — a +0.002, b +0.078 [−0.002, +0.152], c `PROMPT-SUFFICES` at +0.059 — but `role_proto` 0.601 vs `unsteered` 0.596 is a wash, and its parse rate 0.998 at acc/parse 0.602 (< baseline 0.661) makes it a compliance effect. `log/nla-harness/2026-09-15_cn-rescore-results.md`.*
- [x] **H-R9 (ASE) — re-score the bake-off with the `c/n` estimator instead of run-1 `pass@1`** (no GPU; the 3 runs per program were already banked for all 16 arms). Raised by the decoding correction [`log/nla-harness/2026-09-15_pass1-decoding-correction.md`](log/nla-harness/2026-09-15_pass1-decoding-correction.md), which froze the re-read gate **before** the scoring ran: re-read the H-R2a/H-R6 verdicts on `c/n` only if the β = 0 identity arms land within ±0.05 of `unsteered`. Implementation: `--scoring {pass1,cn}` on `nla/src/ase_bakeoff_stats.py`, `load()` the only function that changes, with an identity gate that `--scoring pass1` reproduces the banked JSON byte-identically. *Resolved 2026-09-15 ✓ SUPPORTED: CI half-width ±0.145 → ±0.083 (1.75 ≈ √3), floor +0.067 → −0.027 → **gate passes**, every verdict word survives, but the ranking collapses — `unsteered` 0.596 is third of sixteen, **no arm's CI clears zero upward**, the only interval excluding zero is the harm `prompt_types` −0.217, `prompt` goes +0.101 → −0.055, `codesteer_auto` beats README-exact by +0.032, and `original − unsteered` = −0.051 (the renamed code scores higher, so H-R2c's denominator has the wrong sign). Reports now carry accuracy, not Pass@1. `log/nla-harness/2026-09-15_cn-rescore-results.md`.*
- [x] **H-R7 (ASE) — re-run the whole bake-off inside their runtime with the chat template applied (user request 2026-09-15: "need to rerun to get accurate results then change the setup to do something comparable").** Pre-registered in `log/nla-harness/2026-09-15_chat-template-rerun-prereg.md`, results in `log/nla-harness/2026-09-15_chat-template-rerun-results.md`. **One change**: `SteeredCausalLM._build_prompt` wrapped in CodeLlama's chat template (ids gated equal to `apply_chat_template` on all 50 prompts); their sampler, prior, parser, our seeds and the `c/n` scorer unchanged; pool + vectors refitted under the template (ridge held-out cos 0.3791, gate PASS). 15 arms, jobs 403728–403743 + 405637, ~6 GPU-h. **RESOLVED:** the runtime was the whole story — every arm gains **+0.108…+0.227** `c/n` and parse rises to 0.77–0.96, so the H-R2/H-R6 arm ranking was noise on a broken prompt. With the precondition met (`unsteered` **0.7143**, `original` **0.7727**, damage **`DAMAGE-WEAK` +0.0584 [−0.019, +0.140]**, sign repaired from −0.0507), the answer does not change: **`CODESTEER-INERT`** — `codesteer` 0.6974 (−0.0169 adj [−0.115, +0.086]), `codesteer_auto` 0.7120 (−0.0023) with the effect gate proving 4 096/4 096 attention rows changed. H-R7b **`MATCH-CODESTEER`** (`prompt − codesteer_auto` +0.0154), H-R7d **`SLICE-IRRELEVANT`** (+0.0276), H-R7c keeps all three H-R6 words (`MAP-NOT-BETTER` +0.0515 · `CATEGORY-MEANING-INERT` +0.0430 · `LATENT-BEATS-PROMPT` +0.1897 vs the collapsed `prompt_types`, but `ridge_map − prompt` +0.033 keeps `PROMPT-SUFFICES`), identity-floor gate **PASSES** (−0.0376). `ridge_map` **0.7604** (+0.0461 [−0.027, +0.120]) is the best steered arm and the largest positive in the table, still unable to clear zero on 50 snippets. **Every verdict is provisional** by the frozen rule (damage CI ∋ 0) → raises **H-R12** (power: ≈ 150–200 snippets to certify +0.05) and **H-R13** (`ridge_map` at a fresh seed).
- [x] **H-R14 (ASE) — the WHOLE dataset, and the paper's own numbers to compare against (user request 2026-09-16: "Can we run on the codesteer dataset and compare to paper numbers? The whole dataset?").** Pre-registered in `log/nla-harness/2026-09-16_full-corpus-prereg.md`, results in `log/nla-harness/2026-09-16_full-corpus-results.md`. Full aligned HumanEval-X Java corpus — **148 snippets / 1 742 cases, 4.0× H-R7's 50/434** — 8 arms, 14 jobs (408488–408505, peak **3** concurrent GPUs, verified from `sacct` intervals), ~18 GPU-h. `ridge_map` vectors rebuilt by **nested grouped cross-fit** (5 folds; mu / mean delta / RRR map / prototypes **and** the λ/rank choice all out-of-fold; 5/5 folds PASS at λ=100 rank=256, margins +0.209…+0.223, 4 799/4 799 spans covered), its regression gate first reproducing the banked H-R7 fit exactly. Effect gate live on every shard (2 595–3 396 effective level-2 calls/run). **RESOLVED — and it reverses the 50-snippet reading.** `codesteer_auto` **0.7179** · L0 **0.7015** · `erasure` 0.6996 · `swap_oracle` 0.6963 · `prompt` 0.6914 · `unsteered` **0.6801** · `ridge_map` **0.6770** · `codesteer` 0.6552. **a `DAMAGE-ABSENT`** +0.0214 [−0.027, +0.069] — **my own `DAMAGE-PRESENT` prediction refuted**, the n=50 +0.058 was unrepresentative (+0.016 on the 98 unseen); **b `CODESTEER-INERT`** (−0.0249 / +0.0379, α/2 CIs spanning 0); **c `NLA-MATCHES-CODESTEER` as a word but with the sign reversed** — `ridge_map − codesteer_auto` = **−0.0409**, uncorrected 95 % CI [−0.077, −0.005] **excluding zero**, held out of `CODESTEER-BEATS-NLA` only by the Bonferroni α/3 width (by 0.0049); **d `PROMPT-SUFFICES`** (−0.0144); **e `DAMAGE-FAR-WEAKER`** — ours **+2.14 pts** vs their Qwen2.5-7B renaming stratum's **+36.29** (ratio **0.059**), so the **restoration ratio is WITHHELD** by the pre-registered gate and their headline metric is not computable on this model. **The load-bearing bound:** `swap_oracle`, writing the TRUE clean state at L7, buys only **+0.0163** — every oracle-free L7 latent arm is capped there, so no better vector can produce a CodeSteer-beating result at this site. **`codesteer_auto`'s gain is about half format compliance** (Δparse +0.0293, Δacc|parsed +0.0170); plain `codesteer` *loses* accuracy purely by answering less often (parse 0.853) while its conditional accuracy rises (+0.0145). Raises **H-R15** (generation-seed noise floor — same-50 drift of ±0.02–0.04 between runs that the snippet bootstrap cannot see), **H-R16** (is the parse gain a real compliance mechanism), **H-R17** (damage gap unclosable without their model; H-R3's stronger renamer is the only lever).
- [x] **H-R13 (ASE) — does `ridge_map`'s +0.0461 survive replication? ✗ NOT REPLICATED (2026-09-16).** Absorbed into H-R14, which is a strictly stronger test than the planned seed change: the map was refitted out-of-fold on ~3× the data and scored on 98 snippets it had never seen. On the *same* 50 it is **+0.0061**; on the 98 unseen, **−0.0061**; overall **−0.0031**. Winner's curse — best of six steered arms, chosen post hoc, CI already spanning zero.
- [x] **H-R12 (ASE) — power. ABSORBED (2026-09-16).** Answered by running rather than calculating: 148 clusters cut the CI half-width 0.0739 → 0.0414 (ratio 1.78 vs √2.96 = 1.72 predicted). The +0.05 effect it was sizing does not exist.
- [x] **H-R10 (ASE) — `role_proto − foreign` as a pre-registered primary on damage-bearing stimuli.** The one oracle-free contrast that has come close: **+0.0776 [−0.0016, +0.152]** against a 0.05 bar needing CI > 0, missed by 0.0016, and both arms parse at ordinary per-case rates (0.843 vs 0.783 — the earlier 0.998/0.962 counted phantom ids) so compliance is not what the contrast measures. **Not re-readable on this corpus** — re-reading it here would be retuning after seeing data. To be pre-registered as the primary contrast on H-R7's corpus, same 0.05 bar, before that corpus is scored. **RESOLVED 2026-09-15 ✗ NOT SUPPORTED** (`log/nla-harness/2026-09-15_chat-template-rerun-results.md`): on H-R7's chat-template corpus, scored under the frozen rule, `role_proto − foreign` = **+0.0430 [−0.0218, +0.1174]** — *smaller* than the +0.0776 that raised it and short of the 0.05 bar with a CI including 0. Installing a role/type meaning is not measurably better than installing the wrong item's meaning. `role_proto` itself is 0.7281 vs `unsteered` 0.7143 (+0.0138).
- [ ] **H-R3 (ASE) stronger renamer** — held: the paper's renamer is milder than Paper 2's adversarial L1b; re-run H-R1 on CodeLlama with our decoy renamer if H-R2 shows anything worth recovering more of.
- [x] **H-R4 (ASE) prompt-format damage** — *refuted 2026-09-15: raw-prompt damage +0.025 vs chat-templated +0.113; the raw prompt hurts clean code (0.558 vs 0.800), not renaming.* Original spec: — descriptive, free from job 399863 (`original_unsteered − unsteered` in their raw-prompt runtime vs the chat-templated +0.113).

### Phase 2 — Stretch
- [ ] **E3 full** — decide whether to spend ~130–150 H100-hr on a trained CLT for Llama-3.1-8B, or stay at PoC scale.
- [ ] **E5** cross-model comparison (RSA/CKA + probing primary; train coder-model SAE if the clean version is worth it).
- [ ] Matryoshka / multi-width SAEs if E1/E6 feature separation is corrupted by absorption/splitting.

### Phase 3 — Opportunistic + synthesis
- [ ] **E8** CoT-plateau feature trajectories (reasoning models).
- [ ] **Combined verdict table** across the three instruments; resolve HT1–HT8.
- [ ] Write-up + **replication package** (configs, dictionary ids, seeds, activation-cache manifest).

---

### Phase G — Model-constraint port to Gemma *(added 2026-08-31; supersedes the Qwen host)*

**Constraint:** no Chinese-origin models going forward. This excludes `Qwen2.5-7B-Instruct` (the
Instrument-2 subject) and five of the seven panel models. Panel survivors: **Llama-3.1-8B** (Meta),
**Phi-3.5-mini** (Microsoft), **SmolLM3-3B** (HuggingFace). Judgment call outstanding:
DeepSeek-R1-Distill-Llama-8B (Llama base, DeepSeek distillation).

**Instrument 3 is barely affected.** `Llama-3.1-8B-Instruct` is the only panel model with **both**
pretrained SAEs and transcoders (Llama Scope, EleutherAI, Goodfire l19 on *Instruct*), and it is
cached. `Gemma-2-2B` remains the mandated smoke-test model and has native `circuit-tracer` support,
so [`CLAUDE.md`](CLAUDE.md) §4's small-model requirement is *easier* to satisfy, not harder.

**Instrument 2 is a port, not a rebuild.** Three of the four released NLA pairs are non-Chinese —
`kitft/nla-gemma3-12b-L32-{av,ar}`, `kitft/nla-gemma3-27b-L41-{av,ar}`,
`kitft/Llama-3.3-70B-NLA-L53-{av,ar}`. **Gemma-3-12B-IT (L32/48, d = 3840) is the recommended
successor** because it is the only host with *both* an NLA pair and a dictionary suite — which
would put Instruments 2 and 3 on one model for the first time and make E7 triangulation stronger
than it was. Llama-3.3-70B is an 8×H100 job, not an A6000 one.

**What the reference file already tells us** *(found 2026-08-31, before any GPU time)*

The vendored repo ships a worked example per released pair — `examples/gemma12b_layer32_step4000.txt`,
1,297 lines — so **G0 is available for Gemma exactly as it was for Qwen**, and in the same
`ROW_RE` format `nla/src/replicate_example.py` already parses. It supplies everything the gate
needs: prompt `'What are you hiding?'` (14 tokens), the temp-0 reply (150 tokens), full sequence
**164**, and per-token `||v||` / `mse_nrm` / `cos` / `fve_nrm`. Porting `replicate_example.py` is
parameterisation, not a rewrite — it currently hardcodes `TARGET_MODEL`, `LAYER_INDEX`,
`EXPECTED_N_TOKENS`, `END_OF_TURN` and the prompt/reply strings.

**⚠ The layer is low-variance, and this is the one finding that changes the analysis.** The
reference file warns that Gemma-3-12B at L32 has **Var(v_nrm) = 0.0302** — "activations are highly
concentrated around their mean. mse_nrm/cos compress into a narrow range (everything looks ~0.99
cos). **fve_nrm is the informative metric here.**"

`rt_cos` is the Qwen programme's self-check throughout (validated band 0.70–0.96, observed medians
0.860–0.886) and appears in **23 source files**. On Gemma L32 it would sit near 0.99 for everything
and lose its discriminative range. **But it does not need re-plumbing:** verified against the
reference's own rows, `mse_nrm = 2(1 − cos)` and `fve_nrm = 1 − mse_nrm / Var(v_nrm)`, both to
rounding. **`fve_nrm` is a pure function of `rt_cos` given one constant per model**, so capture
stays exactly as it is and only the reported metric changes — a derived column at analysis time,
not 23 files. `transfer_gate.py` already documents the same relation for Qwen.

That is also an upside for continuity: the authors call `fve_nrm` "the cross-model comparable
metric", so a Gemma port can report a number comparable with the Qwen results in a way `rt_cos`
never was.

**Two further facts from the same file.** Injection rescales every vector to **L2 = 80,000**
(`injection_scale` in `nla_meta.yaml`) against raw norms of ~80k–700k — Gemma's activation scale is
three orders of magnitude above Qwen's, consistent with the √d convention. And the **first four
positions are OOD**: datagen used `min_position=50`, so system-prompt positions were never trained
on. Our stimuli read at `last_prompt`, far past 50, so this does not bite — but it rules out any
early-position analysis.

**G-A · Instrument validation** *(blocks everything below)*
- [x] Download the AV/AR pair — **ungated**, fetched 2026-08-31 via `nla/scripts/fetch_gemma_nla.py`
      (a retry loop: unauthenticated Hub requests are rate-limited and the first attempt died on a
      504 + ReadTimeout ~9.8 GB in; `snapshot_download` resumes).
- [ ] **BLOCKED — `google/gemma-3-12b-it` is `gated: manual`** (HTTP 401 on `config.json`). Needs a
      HuggingFace account to accept Google's terms, then `HF_TOKEN` exported or written to
      `$HF_HOME/token`. No token exists on this host. Gemma-2-2B (the mandated circuit-tracer smoke
      model) is almost certainly gated the same way, so one token covers both. **Do not work around
      this** — an ungated mirror routes around the licence.
- [x] **Fixed `_layers()` in BOTH `nla/src/extract.py` and `nla/src/steer.py`** (2026-08-31) — — Gemma-3 loads as
      both now probe `model.language_model.layers` and `model.model.language_model.layers` in
      addition to the original three. Both spellings, because transformers exposes
      `language_model` at different depths across versions; most-nested last so a plain causal LM
      still resolves first. Verified byte-identical between the two files, with a comment in each
      saying they must stay that way.
- [ ] Run the **layer-indexing gate** (`nla/src/p04_gate.py --layers …`) at L32 — it verifies the
      read site and write site are the same block, which is exactly what the two-file edit risks.
- [ ] **G0 replication gate** on the native pair. *This is the decision point.* A failure here is a
      finding about the released Gemma checkpoints, **not** a repeat of B1 (which used a Qwen NLA
      on a different model; this is the supported native configuration).
- [ ] α = 0 identity test for the write hook — byte-identical to unsteered.
- [ ] Confirm the `sqrt_d_model` embedding convention (`arch_adapters.py`: `gemma3`) does not break
      the AV/AR interface. The steering hook itself is scale-free (unit-normalised direction ×
      local ‖h‖), so injection should be unaffected.

**G-B · Behavioural baseline** — 5 tiers × 60 items × **10 draws** = 3,000 generations, five jobs
in parallel (~½ day). Ten draws, not one: graded k/N labels are what made curve shapes identifiable
(`log/nla-harness/2026-08-30_p1b-graded-labels.md`).
- [ ] Validate `build_call` on Gemma's tokenizer for all five tiers before generating.
- [ ] Record `l0_reply_chars` **and** `l1b_reply_chars` (the omission that blocked a tier result).
- [ ] Run at `--max-new-gen 2048`; at 1,100 the parse rate is 0.8833 and ~12% of items are scored
      wrong for not finishing.

**G-C · Read-side battery** *(~1 day; every script exists)* — dense probes with per-tier length
baselines, the five-tier ladder, the span probe with its repetition control, position × depth.
48 layers instead of 28 gives finer depth resolution than the Qwen run had.

**G-D · Causal arm** *(2–3 days)* — V1–V5 sweep (~1,800 generations) plus the P0.1–P0.4
equivalents (~2,700).

**Estimate ≈ one week**, mostly unattended, conditional on G0 passing first time. Throughput
anchor: 7B at a 2,048 budget ran ~7.5 gen/min on H200 this week; 12B scales to ≈4/min.

**Not re-done:** stimuli, scripts, frozen decision rules, the nineteen-item failure list, and the
whole measurement stack (graded labels, per-tier baselines, selection-free statistics, the
reproducibility-floor protocol). **No number from the Qwen programme survives a host change** — the
host, the layer (20 → 32) and d_model (3584 → 3840) all differ, so G0 gates everything.


## 4. Per-experiment "definition of done" (discipline gate)
Applies to every experiment before a result is "kept" ([`CLAUDE.md`](CLAUDE.md) §4):
- [ ] Deterministic **seed** set and recorded; variance across ≥2 seeds/conditions noted.
- [ ] **Dictionary identity** logged (pretrained id+rev, or trained checkpoint + training config).
- [ ] **Baseline included** where applicable — dense probe (E4/E5), dense-steering + prompting (E2), random-feature ablation (E2). *SAEs are discovery tools, not measurement.*
- [ ] **Causal validation** in the original model for any feature claim (not just correlation).
- [ ] **Error-node mass** reported for every attribution graph (E3).
- [ ] Silent-failure checks: dead features, absorption/splitting, auto-interp illusion, off-target steering, base→Instruct transfer, degenerate reconstruction.
- [ ] **Triangulated** against attention + NLA where the hypothesis allows (E7 logic).
- [ ] Dated **`log/` entry** written; thread README + `log/README.md` indexes updated.

---

## Changelog
- **2026-09-16a** — **H-R14 (ASE) added and closed the same day** (full 148-snippet / 1 742-case HumanEval-X corpus vs the paper's own Table 2 / Table 4 rows, rules frozen before submission): `DAMAGE-ABSENT`, `CODESTEER-INERT`, `NLA-MATCHES-CODESTEER` with the sign reversed, `PROMPT-SUFFICES`, `DAMAGE-FAR-WEAKER`, restoration withheld. **H-R13 closed ✗ NOT REPLICATED** and **H-R12 closed ABSORBED**. New: **H-R15** (seed noise floor), **H-R16** (parse-compliance mechanism), **H-R17** (damage gap). CruxEval-X packs built (698 snippets / 1 396 cases / 2.00 per snippet vs their 1.97) but no GPU arm run there yet.
- **2026-09-14b** — Moved from `docs/CHECKLIST.md` to the project base `CHECKLIST.md` (user request); links re-based.
- **2026-09-14c** — H-R6 (ASE) added at user request: oracle-free meaning-installing vectors (`ridge_map` / `role_proto` / `prompt_types`, deferred `swap_guess` / `ar_role`), rules frozen before the pool capture.
- **2026-09-14** — ASE-2026 block added to the Instrument-2 sidebar: H-R1 (done), H-R2 (running) and, at user request, **H-R5 — the head comparison against CodeSteer** (how many heads the NLA effect needs, whether they are CodeSteer's calibrated heads, whether CodeSteer's heads move the NLA), with heads scored by how much they change the NLA-transported effect (`nec`/`suf` activation patching) and a pre-registered `DIFFERENT-HEADS` prediction. Runs after H-R2 is scored.
- **2026-08-31b** — Phase G started: `_layers()` fixed in both files, AV/AR pair downloaded (ungated). **Blocked on an HF token** — `google/gemma-3-12b-it` is `gated: manual`. Reference-file findings recorded: **G0 is available for Gemma**, and the L32 layer is **low-variance (Var = 0.0302)** so `rt_cos` compresses to ~0.99 — but `fve_nrm` is a pure function of `rt_cos` given that constant, so the fix is a derived column, not 23 files.
- **2026-08-31** — **Phase G added**: model-constraint port to Gemma. No Chinese-origin models going forward removes `Qwen2.5-7B-Instruct` (the Instrument-2 host) and five of seven panel models. Instrument 3 is barely affected — `Llama-3.1-8B` is the only panel model with both SAEs and transcoders, and `Gemma-2-2B` (circuit-tracer native) still serves the mandated smoke test. Instrument 2 ports rather than dies: three of four released NLA pairs are non-Chinese, and **Gemma-3-12B-IT is the recommended successor** as the only host with both an NLA pair and a dictionary suite. Two integration facts recorded before any GPU time: `_layers()` will fail on `Gemma3ForConditionalGeneration`, and Gemma normalises embeddings by √d. Estimate ≈ 1 week, gated on G0.
- **2026-08-03** — Phase-0 progress recorded: scaffold tree + provenance/seed/gpu discipline **done**; `environment.yml` written + extraction harness built and smoke-tested **[~]** (env not created; batching/GQA/sink-tracking + `apply_dictionary` pending); annotated the remaining boxes with their landing zones (`configs/data.yaml`, `configs/dictionaries.yaml`). Status → Phase 0 in progress.
- **2026-08-07c** — N9 exploratory addendum: confabulation rate measured (Python prior), planning refuted against a null.
- **2026-08-07b** — Confirmatory family closed: HT13 not adjudicated (instrument null), HT14 refuted; BH-FDR recorded as vacuous. Verdict summary added.
- **2026-08-07** — Instrument-2 sidebar expanded: banked corpus, N4/N5 instruments and the HT12–HT14 pre-registration recorded; **HT12 refuted** and the 2026-08-06 faithfulness↔correctness gap reclassified as a reply-length artifact. HT9–HT11 remain parked.
- **2026-07-24** — Created. Formalized HT1–HT8 from the E1–E8 confirm/falsify conditions in `experiment_menu.md`; added the external attention H1–H4 that E7 triangulates; laid out the phased task list (Phase 0 infra → Phase 1 committed E1/E2/E3-PoC/E7+E4 → Phase 2 stretch → Phase 3 synthesis) and the discipline gate.
- **2026-09-03** — E3: recorded the **token-length-matching prerequisite** for any patching-based attribution work, measured on the banked L0/L1b corpus (0/60 pairs admit a 1:1 token map; median +27-token inflation). Noted that the separate question — whether that inflation confounds the L1b effect — was tested and refuted (ρ_partial = −0.159), so the prerequisite constrains the method, not the stimuli.
- **2026-09-15** — H-R2 and H-R4 (ASE) boxes closed with their verdicts; H-R5 unblocked with its comparator numbers.
- **2026-09-15b** — H-R6 (ASE) box closed with its verdicts; H-R7 (damage-bearing stimuli) and H-R8 (compliance vs meaning) to be added when pre-registered.
- **2026-09-15c** — H-R9 (ASE) added and closed the same day (the `c/n` re-score, rule frozen in the decoding correction before it ran): verdicts survive, the ranking does not, nothing beats `unsteered`. H-R6's box carries the re-scored numbers. **H-R10 opened** (`role_proto − foreign`, blocked on H-R7). Accuracy replaces Pass@1 in the reporting.
- **2026-09-15d** — Parse-rate correction (`log/nla-harness/2026-09-15_parse-rate-correction.md`): H-R6/H-R10 parse figures annotated (per-case, not `parsed_frac`); H-R8 withdrawn as posed; **H-R7 re-scoped** to a chat-template re-run of six arms through their `SteeredCausalLM` (on the same 50 the chat template gives 0.780 → 0.714, their raw prompt 0.545 → 0.596); **H-R11 raised** (does their fixation diagnostic fire on CodeLlama-7B at all).
- **2026-09-15e** — H-R7 (ASE) box added and closed the same day (chat-template re-run of all 15 arms inside their runtime, rules frozen before submission): `DAMAGE-WEAK` +0.058, `CODESTEER-INERT`, `MATCH-CODESTEER`, `SLICE-IRRELEVANT`, all three H-R6 words survive, `ridge_map` 0.760 best steered arm with CI ∋ 0; **H-R10 closed ✗ NOT SUPPORTED** (`role_proto − foreign` +0.043 on the pre-registered corpus); **H-R12** (power/corpus size) and **H-R13** (`ridge_map` seed replication) raised.
