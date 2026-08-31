# Transcoders — Hypotheses, Experiments & Task Checklist

*Last updated: 2026-08-03*
**Status:** Phase 0 in progress (scaffold built, smoke passing; no science runs yet) · **Instrument 3** (SAE + transcoder feature/circuit analysis)

The living plan for the `transcoders/` sub-project. Master hypotheses ledger + experiment
tracker + phased to-do list. Source of truth for *what* and *why*:
[`experiment_menu.md`](experiment_menu.md) (full E1–E8 menu), the proposal deck in this
folder, [`../papers/REFERENCES.md`](../papers/REFERENCES.md) (Papers 1–3), and
[`../CLAUDE.md`](../CLAUDE.md) §3–§4. Per-thread progress lives in
[`../log/`](../log/); this file is the master list those threads resolve against.

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
- [x] Create `src/`, `configs/`, `data/` (large artifacts → `data/`, never `$HOME` — [`../CLAUDE.md`](../CLAUDE.md) §2). *(2026-08-03: full tree + `scripts/smoke.sh`; see `src/README.md`.)*
- [x] Pin conda env `/data/jvl210002/conda_envs/transcoders-mi`; write `environment.yml`. *(2026-08-03: created + frozen (`environment.yml` exact pins + `environment.lock.txt`); circuit-tracer 0.5.2 installed (downgrades transformers→4.57 — documented); `pip check` clean; smoke green in-env; `scripts/env.sh` handles the CXXABI `LD_LIBRARY_PATH` fix.)*
- [x] Pull the reused **stimuli**: Dataset A + Dataset B. *(2026-08-04: located, symlinked, AND **converted** — `src/convert_stimuli.py` → 350 Snippet-JSONL rows with per-span classes and the cross-tier-derived decoy↔true `rename_map` (100% pairing on A-L1b; alignment parquets proved to be a different generation round — see `DATA_SOURCES.md`). Span→token resolution **1.0000** on Llama-3.1 + Qwen3 tokenizers.)*
- [~] Ingest the **behavioral tables** from Papers 2–3. *(2026-08-04: **located + symlinked** — `paper2_trials.parquet` (31,711 rows; `is_core==1` → 29,546), `paper2_adv_features.parquet` (ISF etc.), `paper2_dispatcher_cf.csv`, `paper3_human_graded.csv`, `paper3_model_results.xlsx`. Gotcha: HCI is **derived**, not stored — recipe in `DATA_SOURCES.md`.)*
- [x] Obtain **pretrained dictionaries** — identity + revision recorded. *(2026-08-04: all 14 registry entries pinned with HfApi-verified shas — Llama Scope LXR/LXTC (+widths), Llama Scope R1, EleutherAI ×2, Goodfire l19 (Instruct-trained!), AIRI, Qwen-Scope (official `Qwen/SAE-Res-*`), BluelightAI CLT, mwhanna PLTs (post-trained Qwen3 match), Chanin, Gemma Scope res/tc + circuit-tracer shim. Weights not yet downloaded — that happens per-experiment.)*
- [~] Build the **activation-extraction harness** (batched forward passes; cache under `data/`; GQA-aware; sink tokens tracked separately). *(2026-08-03: `src/extract_activations.py` built + smoke-tested end-to-end on tiny-gpt2 — config merge/seed/capture/safetensors/manifests all pass. Still TODO: real batching, GQA-aware attention handling, sink-token tracking, `apply_dictionary` once repo ids are pinned.)*
- [ ] Build the **auto-interp** feature-labeling harness (recall-biased — pair with causal/specificity validation).
- [ ] Build the **steering/intervention** harness (SAELens / circuit-tracer) + the **dense-steering + prompting baselines**.
- [ ] Build the **probing** harness (linear + SAE-feature probes) for E4.
- [ ] Wire results into the **existing GLMM stack** (binomial GLMMs, crossed random effects snippet×model, Wilson CIs, BH-FDR).
- [x] Seed + provenance logging scaffold (script sha256, GPU id, timestamp, dictionary id) — [`../CLAUDE.md`](../CLAUDE.md) §4. *(2026-08-03: `src/seedutil.py` + `src/provenance.py` → `run_manifest.json` per run; `src/gpu.py` enforces idle-GPU pinning before torch import.)*

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
so [`../CLAUDE.md`](../CLAUDE.md) §4's small-model requirement is *easier* to satisfy, not harder.

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
Applies to every experiment before a result is "kept" ([`../CLAUDE.md`](../CLAUDE.md) §4):
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
- **2026-08-31b** — Phase G started: `_layers()` fixed in both files, AV/AR pair downloaded (ungated). **Blocked on an HF token** — `google/gemma-3-12b-it` is `gated: manual`. Reference-file findings recorded: **G0 is available for Gemma**, and the L32 layer is **low-variance (Var = 0.0302)** so `rt_cos` compresses to ~0.99 — but `fve_nrm` is a pure function of `rt_cos` given that constant, so the fix is a derived column, not 23 files.
- **2026-08-31** — **Phase G added**: model-constraint port to Gemma. No Chinese-origin models going forward removes `Qwen2.5-7B-Instruct` (the Instrument-2 host) and five of seven panel models. Instrument 3 is barely affected — `Llama-3.1-8B` is the only panel model with both SAEs and transcoders, and `Gemma-2-2B` (circuit-tracer native) still serves the mandated smoke test. Instrument 2 ports rather than dies: three of four released NLA pairs are non-Chinese, and **Gemma-3-12B-IT is the recommended successor** as the only host with both an NLA pair and a dictionary suite. Two integration facts recorded before any GPU time: `_layers()` will fail on `Gemma3ForConditionalGeneration`, and Gemma normalises embeddings by √d. Estimate ≈ 1 week, gated on G0.
- **2026-08-03** — Phase-0 progress recorded: scaffold tree + provenance/seed/gpu discipline **done**; `environment.yml` written + extraction harness built and smoke-tested **[~]** (env not created; batching/GQA/sink-tracking + `apply_dictionary` pending); annotated the remaining boxes with their landing zones (`configs/data.yaml`, `configs/dictionaries.yaml`). Status → Phase 0 in progress.
- **2026-08-07c** — N9 exploratory addendum: confabulation rate measured (Python prior), planning refuted against a null.
- **2026-08-07b** — Confirmatory family closed: HT13 not adjudicated (instrument null), HT14 refuted; BH-FDR recorded as vacuous. Verdict summary added.
- **2026-08-07** — Instrument-2 sidebar expanded: banked corpus, N4/N5 instruments and the HT12–HT14 pre-registration recorded; **HT12 refuted** and the 2026-08-06 faithfulness↔correctness gap reclassified as a reply-length artifact. HT9–HT11 remain parked.
- **2026-07-24** — Created. Formalized HT1–HT8 from the E1–E8 confirm/falsify conditions in `experiment_menu.md`; added the external attention H1–H4 that E7 triangulates; laid out the phased task list (Phase 0 infra → Phase 1 committed E1/E2/E3-PoC/E7+E4 → Phase 2 stretch → Phase 3 synthesis) and the discipline gate.
