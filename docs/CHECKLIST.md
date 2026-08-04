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

### Phase 2 — Stretch
- [ ] **E3 full** — decide whether to spend ~130–150 H100-hr on a trained CLT for Llama-3.1-8B, or stay at PoC scale.
- [ ] **E5** cross-model comparison (RSA/CKA + probing primary; train coder-model SAE if the clean version is worth it).
- [ ] Matryoshka / multi-width SAEs if E1/E6 feature separation is corrupted by absorption/splitting.

### Phase 3 — Opportunistic + synthesis
- [ ] **E8** CoT-plateau feature trajectories (reasoning models).
- [ ] **Combined verdict table** across the three instruments; resolve HT1–HT8.
- [ ] Write-up + **replication package** (configs, dictionary ids, seeds, activation-cache manifest).

---

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
- **2026-08-03** — Phase-0 progress recorded: scaffold tree + provenance/seed/gpu discipline **done**; `environment.yml` written + extraction harness built and smoke-tested **[~]** (env not created; batching/GQA/sink-tracking + `apply_dictionary` pending); annotated the remaining boxes with their landing zones (`configs/data.yaml`, `configs/dictionaries.yaml`). Status → Phase 0 in progress.
- **2026-07-24** — Created. Formalized HT1–HT8 from the E1–E8 confirm/falsify conditions in `experiment_menu.md`; added the external attention H1–H4 that E7 triangulates; laid out the phased task list (Phase 0 infra → Phase 1 committed E1/E2/E3-PoC/E7+E4 → Phase 2 stretch → Phase 3 synthesis) and the discipline gate.
