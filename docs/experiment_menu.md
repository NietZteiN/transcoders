# SAE & Transcoder Experiments for Obfuscated-Code Comprehension: A Candidate-Experiment Menu

## TL;DR
- **The single most valuable third instrument is a feature-level "semantic capture" analysis of adversarial renaming (L1b) plus causal feature steering, backed by transcoder attribution graphs on matched L0/L2/L3 snippets** — because these map directly onto your strongest existing findings (the HCI collapse to 21.25%, the r=−0.196 dispatcher-complexity effect) and give you a *causal* test the attention and NLA instruments cannot.
- **Feasibility is asymmetric across your panel.** Llama-3.1-8B has off-the-shelf SAEs *and* transcoders (Llama Scope), DeepSeek-R1-Distill-Llama-8B has SAEs (Llama Scope R1), and Qwen3 has SAEs (Qwen-Scope) — but your reasoning-tuned core (DeepSeek-R1-Distill-Qwen-7B/1.5B, SmolLM3-3B) and all three coder models have **no** public SAEs/transcoders and would need training from scratch (~300–500M activation tokens, ~10–30 GPU-hours per residual-stream SAE per layer).
- **The literature is fast-moving and partly adversarial to SAEs.** 2025–2026 results (Kantamneni et al.; Wu et al./AxBench) show SAEs underperform simple baselines on probing/steering, and error nodes carry real causal weight in attribution graphs. Any SAE claim must be triangulated against your attention and NLA instruments and validated causally — treat SAEs as *discovery* tools, not measurement tools.

## Key Findings
1. **SAE architectures have converged on a sparsity–fidelity–interpretability frontier with known tradeoffs.** BatchTopK and JumpReLU give the best reconstruction; Matryoshka SAEs give the best *feature disentanglement* (best on absorption, sparse probing, SCR) despite worse reconstruction. There is no free lunch: optimizing reconstruction (TopK/JumpReLU) worsened feature absorption.
2. **Transcoders — especially cross-layer transcoders (CLTs) — are the right tool for your circuit questions**, because they model the *computation* of MLP layers rather than reconstructing a single activation, enabling attribution graphs (Anthropic's circuit-tracing line). Transcoders also slightly *beat* SAEs on probing in the one head-to-head extension available.
3. **Availability is the binding feasibility constraint.** Only part of your panel has pretrained dictionaries; the reasoning-Qwen and coder models do not.
4. **Prior code-specific interpretability exists but is thin** — code-correctness SAE directions, variable-binding circuits, entity/state tracking — giving you real methods to build on and a genuine gap to claim.
5. **SAEs are contested.** You must build in causal validation and cross-instrument triangulation from the start.

## Details

### 1. Current state of SAE methods

**The core idea and the standard architecture.** An SAE learns an over-complete, sparse dictionary that reconstructs a model activation `x ≈ Σ f_i(x) d_i + b`, where feature activations `f_i` are sparse. The founding work is Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (arXiv:2309.08600) and Anthropic's Bricken et al. (2023). The motivation is superposition/polysemanticity (Elhage et al., "Toy Models of Superposition," 2022).

**Architecture family and tradeoffs:**
- **Vanilla/ReLU (L1-penalty) SAEs** — original; suffer "shrinkage" from the L1 penalty biasing activations toward zero.
- **Gated SAEs** (Rajamanoharan et al., 2024) — decouple feature *detection* from *magnitude* to fix shrinkage.
- **TopK SAEs** (Gao et al., "Scaling and evaluating sparse autoencoders," arXiv:2406.04093) — enforce exactly k active latents; directly control L0. This work trained the flagship 16-million-latent autoencoder on GPT-4 activations for **40 billion tokens** (verbatim from the paper).
- **BatchTopK** (Bussmann et al., arXiv:2412.06410) — relax the top-k constraint to the batch level; "consistently outperform TopK SAEs at reconstructing activations from GPT-2 Small and Gemma 2 2B, and achieve comparable performance to the state-of-the-art JumpReLU SAE."
- **JumpReLU SAEs** (Rajamanoharan et al., arXiv:2407.14435) — discontinuous activation with a learned threshold; state-of-the-art reconstruction fidelity; used by Gemma Scope. Downside: L0 cannot be fixed to a target value.
- **Matryoshka SAEs** (Bussmann et al., 2025) — nested dictionaries trained at multiple widths simultaneously; on SAEBench they "substantially outperform other architectures on feature disentanglement metrics; moreover, this advantage grows with SAE scale," at the cost of slightly worse reconstruction. They mitigate feature absorption and feature splitting.
- **Newer variants**: Matching-Pursuit SAEs (best reconstruction, poor latent quality on SynthSAEBench), AbsTopK/bidirectional, AdaptiveK (complexity-driven k), variational SAEs.

**Known problems and critiques:**
- **Feature absorption** (Chanin et al., "A is for Absorption," arXiv:2409.14507) — a general feature (e.g., "starts with S") gets absorbed into a more specific latent, creating gerrymandered features. Worsened by TopK/JumpReLU.
- **Feature splitting** — as width grows, one concept splits into many narrower ones (Bricken et al.; confirmed in Llama Scope).
- **Feature hedging** (Chanin et al., arXiv:2505.11756) — correlated features break narrow SAEs.
- **Feature composition/occlusion** and **dead features**.
- **"SAEs don't find canonical units"** (Bussmann et al., arXiv:2502.04878) — different widths find different decompositions.
- **The downstream-utility critique (most important for you):** Kantamneni, Engels, Rajamanoharan, Tegmark & Nanda, "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (ICML 2025, arXiv:2502.16681): **"SAE probes underperform the baseline of logistic regression in each regime when taking the mean across datasets"** — tested on 113 binary-classification datasets across data scarcity, class imbalance, label noise, and covariate shift. Wu, Arora, Geiger et al., "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders" (ICML 2025 spotlight, arXiv:2501.17148), on Gemma-2-2B and 9B: **"For steering, we find that prompting outperforms all existing methods, followed by finetuning."** Google DeepMind's interpretability team publicly de-prioritized SAEs for downstream tasks. A rebuttal — "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" — argues SAEs are for *discovery* of unknown concepts, not for acting on prespecified ones. There is even a random-baseline critique (Korznikov et al., "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" arXiv:2602.14111) and Heap et al. showing SAEs on *random* models still yield "interpretable" features.

**Evaluation methodology:**
- **SAEBench** (Karvonen et al., ICML 2025; saebench.xyz) — 8 evals: Feature Absorption, AutoInterp, L0/Loss Recovered, RAVEL, Spurious Correlation Removal (SCR), Targeted Probe Perturbation (TPP), Sparse Probing, Unlearning.
- **Sparse probing** (Gurnee et al., "Finding Neurons in a Haystack," 2023; extended to SAEs by Gao et al.) — k=1 probing: can a single latent classify a concept?
- **Automated interpretability** (Bills et al.; EleutherAI's open auto-interp, blog.eleuther.ai/autointerp) — an LLM generates and scores explanations. Warning: recall-biased, creates an "illusion of interpretability" (Bolukbasi et al., 2021, cited in Gao et al.).
- **SynthSAEBench / CE-Bench / MCC** — synthetic ground-truth evaluations.

### 2. Current state of transcoders

**How transcoders differ from SAEs.** An SAE reconstructs one activation `x → x̂`. A transcoder approximates the *computation* of an MLP sublayer: it takes the MLP *input* and predicts the MLP *output* with a wide, sparsely-activating hidden layer. This lets you interpret not just what a layer outputs but *how it computes it*. Foundational: Dunefsky, Chlenski & Nanda, "Transcoders Find Interpretable LLM Feature Circuits" (arXiv:2406.11944, NeurIPS 2024) — circuits "neatly factorize into input-dependent and input-invariant terms," reverse-engineering GPT-2's greater-than circuit.

- **Skip transcoders** (Paulo et al., "Transcoders Beat Sparse Autoencoders for Interpretability," arXiv:2501.18823) — add a linear skip connection; lower reconstruction error, more interpretable latents.
- **Per-layer transcoders (PLTs)** — one transcoder per MLP layer.
- **Cross-layer transcoders (CLTs)** (Ameisen, Lindsey et al., "Circuit Tracing," transformer-circuits.pub 2025) — each feature reads at one layer and writes to *all subsequent* MLP layers, greatly simplifying circuits. Per the paper: **"Remarkably, we can substitute our learned CLT features for the model's MLPs while matching the underlying model's outputs in ~50% of cases."**

**Attribution graphs & circuit tracing.** Anthropic's two-paper line: "Circuit Tracing: Revealing Computational Graphs in Language Models" (methods) and "On the Biology of a Large Language Model" (Lindsey et al., applying it to Claude 3.5 Haiku). Nodes = token embeddings, active CLT features, error nodes, output logits; edges = linear attributions via the frozen-nonlinearity backward Jacobian. Findings include two-step reasoning (Dallas→Texas→Austin), multilingual shared circuits, planning in poetry, and entity-recognition/hallucination circuits — each validated by feature interventions.

**Open-source ecosystem & replications:**
- **circuit-tracer** (Anthropic Fellows; github.com/safety-research/circuit-tracer and decoderesearch/circuit-tracer) — supports Gemma-2-2B, Llama-3.2-1B, and now Qwen3-4B; started with PLTs, CLT support added; feature interventions supported.
- **Neuronpedia** hosts the attribution-graph frontend; on-demand graph generation, feature interventions, sharable graphs.
- **EleutherAI "Attribute"** — independent CLT attribution library; EleutherAI's **Sparsify** repo trains CLTs at scale.
- **Goodfire** replicated CLT attribution graphs on GPT-2's greater-than circuit ("Replicating Circuit Tracing for a Simple Known Mechanism").

**Relationship to other circuit-discovery methods** (for positioning):
- **Activation patching / path patching** (Wang et al. IOI, 2022; Goldowsky-Dill et al.) — causal but labor-intensive, node-level.
- **ACDC** (Conmy et al., 2023) — automates patching, but needs a forward pass per edge (slow).
- **Edge Attribution Patching (EAP)** (Syed et al., BlackboxNLP 2024) — linear gradient approximation; "discovering the IOI circuit in GPT2-small in 4.1 seconds compared to ACDC's 8 minutes"; EAP-IG (Hanna et al.) improves faithfulness.
- **Sparse Feature Circuits** (Marks et al., ICLR 2025, features.baulab.info) — SAE features as causal circuit nodes with SAE-error nodes included.

Transcoder attribution graphs sit at the "interpretable-nodes, weights-based" end: more interpretable nodes than attention-head circuits, but requiring trained dictionaries and inheriting reconstruction error.

### 3. Pretrained-dictionary availability for your panel (the feasibility crux)

| Panel model | Pretrained SAEs? | Pretrained transcoders? | Source |
|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **YES** (Llama Scope 256 SAEs on base; EleutherAI sae-llama-3.1-8b-32x/64x; Goodfire l19 on *Instruct*) | **YES** — Llama Scope includes a transcoder ("TC") site on all 32 layers | Llama Scope (arXiv:2410.20526); EleutherAI; Goodfire |
| **DeepSeek-R1-Distill-Llama-8B** (adjacent, not in panel) | **YES** — Llama Scope R1 (OpenMOSS) | Partly | Neuronpedia |
| **DeepSeek-R1-Distill-Qwen-7B / 1.5B** | **NO public** (research k-SAE on R1-Distill-Qwen-1.5B via EleutherAI; AIRI SAE-Reasoning targets Llama-8B/Qwen-1.5B/Qwen-7B) | **NO** | AIRI-Institute/SAE-Reasoning; EleutherAI |
| **Qwen3-0.6B** | **YES** (Qwen-Scope covers Qwen3-1.7B/8B; Qwen3 SAEs on Neuronpedia); for 0.6B specifically, use BluelightAI CLTs | **YES (CLT)** — BluelightAI Qwen3-0.6B/1.7B CLTs (circuit-tracer-compatible) | Qwen-Scope (arXiv:2605.11887); BluelightAI |
| **SmolLM3-3B** | **NO public** | **NO** | — |
| **Phi-3.5-mini-instruct** | **NO public** | **NO** | — |
| **Qwen-7B / Qwen2.5-Coder-7B / CodeLlama-7B / DeepSeek-Coder-6.7B** | **NO public** (a Qwen2.5-7B-Instruct SAE exists via David Chanin/Neuronpedia) | **NO** | Neuronpedia |

**Training-from-scratch cost.** Gemma Scope (Lieberum et al., arXiv:2408.05147v2) is the canonical "huge compute" suite: it "contains more than 400 sparse autoencoders in the main release, with more than 30 million learned features in total… trained on 4–16B tokens of text each. We used over 20% of the training compute of GPT-3 (Brown et al., 2020), saved about 20 Pebibytes (PiB) of activations to disk." (Note: that is ~20% of GPT-3 *pretraining* compute for the *entire 400+ SAE suite*, not per-SAE.) OpenAI's flagship trained a 16M-latent SAE on GPT-4 activations for **40 billion tokens** (Gao et al., arXiv:2406.04093). Llama Scope (arXiv:2410.20526) trained 256 SAEs on Llama-3.1-8B using SlimPajama at 1024-token context in bfloat16 (exact per-SAE token count not stated verbatim; commonly ~8B).

For *your* scale, the realistic figures are much smaller:
- **One residual-stream SAE, one 7–8B-model layer:** ~300–500M activation tokens, ~10–30 GPU-hours on an 80GB GPU. Feature-Hedging and e2e-SAE studies train one SAE on a single H100; typical research SAEs use 200–400M tokens.
- **One full cross-layer transcoder on a ~1B model:** CLT-Forge trained a CLT on Llama-3.2-1B (65,536 features/layer × 16 layers) on 300M OpenWebText tokens in **17 hours on one 8×H100 node (~136 H100 GPU-hours)** — the cleanest wall-clock anchor.
- Activation extraction (forward passes of the base 8B model) is often the dominant cost.

Within your envelope (4×A6000 + one 8×H100 allocation, 16 weeks): training SAEs for a handful of layers on 2–3 currently-uncovered models is feasible; training the full 10-model panel of SAEs, or CLTs on all 8B models, is not.

### 4. Prior work on SAEs/transcoders for code and on variable/state binding

- **Code-correctness SAEs**: "Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders" (arXiv:2510.02917). SAE predictor directions detect incorrect code at **F1 = 0.821**, while steering shows a fix/corrupt tradeoff (**4.04% fixed, 14.66% corrupted**). Mechanistically, "successful code generation depends on attending to test cases rather than problem descriptions," and base-model directions "retain their effectiveness after instruction-tuning." This is a directly transplantable methodology (t-statistic direction selection, steering, weight orthogonalization) and its base→instruct transfer result is reassuring for your panel.
- **Variable binding**: Wu, Geiger & Millière, "How Do Transformers Learn Variable Binding in Symbolic Programs?" (ICML 2025, arXiv:2505.20896) — a Transformer dereferences variable-assignment chains up to 4 hops; "the model learns to exploit the residual stream as an addressable memory space, with specialized attention heads routing information across token positions." Davies et al., "Discovering Variable Binding Circuitry with Desiderata" (arXiv:2307.03637); Feng & Steinhardt, "How do language models bind entities in context?" (ICLR 2024, binding-ID vectors / binding subspace).
- **State tracking**: Li, Guo & Andreas, "(How) Do Language Models Track State?" (ICML 2025, arXiv:2503.02854) — permutation word problems reduce to finite-automata simulation; LMs learn "associative scan" or "parity-associative" mechanisms; uses prefix-patching + probing signatures. Entity tracking across state changes (arXiv:2605.30233; finding that models do *not* track incrementally but aggregate at the query); retrieval-conditioned rebinding circuit (arXiv:2606.08644). Kim et al. note "code pretraining improves entity tracking."
- **Syntax vs semantics**: Temporal SAEs (arXiv:2511.05541) disentangle local syntactic vs long-range semantic features.

## The Experiment Menu (the core deliverable)

Eight experiments (E1–E8), ordered by value/feasibility. Each maps to a Block Model level and a prior finding, with method + metric + confirm/falsify + feasibility + risks. I recommend the **Tier-1 set (E1, E2, E3)** as the committed third instrument, with E4–E6 as stretch and E7–E8 as opportunistic cross-instrument validation.

---

**E1 — Semantic-capture feature diff under adversarial renaming (L0 vs L1b).** *[Tier 1, flagship]*
- **What it measures:** whether SAE features encoding the *decoy* semantics (e.g., "area/geometry/temporal-smoothing" for `smoothArea`) activate on the renamed identifiers while the *true-semantics* features (e.g., "recurrence/accumulator/Fibonacci") are suppressed, relative to L0.
- **Block Model level:** Atoms (identifier tokens) × Text-surface and Function dimensions.
- **Maps to prior finding:** Paper 2's HCI collapse (accuracy 21.25%, HCI 10.0%) at the high-semantic-displacement × high-ISF intersection; Paper 3's "Stroop-like interference."
- **Method/metric:** On matched L0/L1b snippet pairs, extract SAE feature activations at identifier token positions (residual-stream SAE, mid-layer). Auto-interp label features; identify "decoy-domain" and "true-domain" feature sets. Define a **Semantic Capture Score** = (Δ decoy-feature mass) − (Δ true-feature mass) from L0→L1b at identifier positions. Correlate SCS with per-item HCI/accuracy from Paper 2.
- **Confirms if:** SCS is significantly positive on trap items and predicts HCI (e.g., top SCS quartile shows the accuracy collapse). **Falsifies if:** decoy features don't activate above L1 baseline, or SCS is uncorrelated with HCI.
- **Feasibility:** ✅ Strong on **Llama-3.1-8B** (Llama Scope) and **Qwen3** (Qwen-Scope). Forward-pass only; fits 4×A6000. ~1–2 weeks.
- **Risks/confounds:** auto-interp illusion (validate causally in E2); base-trained SAE may transfer imperfectly to Instruct; decoy/true feature sets may not separate cleanly (absorption/splitting) — mitigate with Matryoshka or multiple-width SAEs.

---

**E2 — Causal feature steering: ablate decoy / amplify true semantics.** *[Tier 1, the causal payoff]*
- **What it measures:** whether ablating the decoy feature (or amplifying the true-semantics feature) *recovers output accuracy* on L1b trap items — a cleaner causal test than PASTA attention steering, and directly comparable to the planned NLA-derived steering (Instrument 2, E4).
- **Block Model level:** Atoms → Function (does fixing the atom-level representation restore purpose comprehension?).
- **Maps to prior finding:** HCI failures (Paper 2); causal complement to E1.
- **Method/metric:** Using E1 feature sets, intervene (set decoy feature to 0, or scale true feature up) during the forward pass and measure Δaccuracy and ΔHCI on trap items. Report a dose-response curve vs steering coefficient. Compare against a random-feature-ablation control **and a dense steering-vector / prompting baseline (mandatory given AxBench)**.
- **Confirms if:** decoy ablation / true amplification produces monotone accuracy recovery exceeding random-feature ablation. **Falsifies if:** no recovery, or the dense/prompt baseline dominates — which, given the AxBench finding that "prompting outperforms all existing methods," is itself a publishable result if pre-registered.
- **Feasibility:** ✅ Llama-3.1-8B, Qwen3. Intervention tooling exists (SAELens, circuit-tracer). ~2 weeks.
- **Risks:** steering off-target effects (validate with RAVEL-style specificity); pre-register the baseline comparison so a null is still informative.

---

**E3 — Transcoder attribution graphs on matched L0/L2/L3 snippets: does the dispatcher circuit break?** *[Tier 1, transcoder core]*
- **What it measures:** whether the computational graph for the *same underlying program* changes shape under control-flow flattening; whether an identifiable "dispatcher/state-tracking" circuit exists under L2 and *breaks* at L3.
- **Block Model level:** Relations & Macro Structure × Program-execution dimension.
- **Maps to prior finding:** the r = −0.196 (q = 3.08e-23) dispatcher-complexity effect (Paper 2); GEE OR 0.57 for L2 (Paper 3).
- **Method/metric:** Use **circuit-tracer** with CLTs. On **Llama-3.1-8B** use Llama Scope per-layer transcoders (or train a CLT — see feasibility); alternatively run the whole instrument on a supported small model (Qwen3-4B, Llama-3.2-1B, Gemma-2-2B) as a proof-of-concept. Generate attribution graphs at the output-prediction token for L0/L2/L3 variants. Quantify graph divergence (node/edge overlap, path length to the answer logit, presence of a dispatcher-token→state-feature→branch path). Correlate a "dispatcher-circuit integrity" score with dispatcher state count and accuracy.
- **Confirms if:** a reproducible dispatcher-state circuit appears under L2 and degrades (more error-node mass, longer/broken paths) at L3, tracking the r=−0.196 gradient. **Falsifies if:** graphs are dominated by error nodes or show no systematic L2→L3 change.
- **Feasibility:** ⚠️ Mixed. Small supported models (Gemma-2-2B, Llama-3.2-1B, Qwen3-4B): Colab-scale, ✅. Llama-3.1-8B with a *trained* CLT: on the order of ~130–150 GPU-hours on your 8×H100 allocation — feasible but a real chunk of budget. Full panel: not feasible.
- **Risks:** error nodes carrying causal weight (see §6) undermine graph completeness; attribution graphs are per-prompt and labor-intensive to interpret; matched-snippet graph comparison is methodologically novel (a contribution, but risky).

---

**E4 — Transcoder/probe state-binding fidelity across dispatcher hops.** *[Tier 2]*
- **What it measures:** whether the model maintains a feature representing the current dispatcher state value (s = 0/1/2) across hops in the flattened while-if state machine, and whether that feature's fidelity predicts correctness.
- **Block Model level:** Relations × Program-execution.
- **Maps to prior finding:** r=−0.196 dispatcher complexity; connects to Wu/Geiger variable binding and Li/Andreas state tracking.
- **Method/metric:** Train a linear or SAE-feature probe for the current-state variable `s` at each dispatch site (following Li–Andreas prefix-patching + probing signatures and Wu–Geiger dereferencing analysis). Track probe accuracy across hop depth; correlate decay with output accuracy and dispatcher state count.
- **Confirms if:** state-feature fidelity decays with hop count and predicts errors. **Falsifies if:** state is not linearly decodable or decoding is unrelated to accuracy — which would echo the entity-tracking finding that models aggregate at the query rather than track incrementally, itself interesting.
- **Feasibility:** ✅ Probing is cheap and needs no pretrained dictionary (probes trained on your data) — works on *any* panel model, including the coder and reasoning-Qwen models. ~2 weeks.
- **Risks:** probing ≠ causal use (amnesic-probing caveat); linear decodability may reflect information *present* but not *used*.

---

**E5 — Cross-model-type feature comparison (reasoning vs instruct vs coder).** *[Tier 2]*
- **What it measures:** whether human-aligned models (reasoning-tuned, ρ=0.30–0.47) share features that non-aligned (coder/instruct, ρ≈0) models lack.
- **Block Model level:** all levels; explanatory for the Paper-2 alignment split.
- **Maps to:** Paper 2's ρ=0.30–0.47 vs near-zero human-alignment split.
- **Method/metric:** Compare feature dictionaries / feature activation profiles on identical obfuscated snippets across model types. Because pretrained dictionaries exist only for some models, restrict the *clean* comparison to models with SAEs (Llama-3.1-8B Instruct vs Qwen3 reasoning vs a coder model you train an SAE for) OR use model-agnostic measures (probing accuracy for "difficulty," representational similarity/CKA between families).
- **Confirms if:** reasoning models exhibit a "task-difficulty" or "trace-length" feature absent in coder models. **Falsifies if:** no consistent feature-level difference tracks the alignment split.
- **Feasibility:** ⚠️ Cross-model SAE comparison is confounded by *different dictionaries*; cleanest version needs SAEs trained on matched data per model (coder models: train from scratch, ~10–30 GPU-hr each). ~3–4 weeks.
- **Risks:** dictionaries are not directly comparable across models ("SAEs don't find canonical units"); prefer RSA/probing baselines as primary, SAE features as illustration.

---

**E6 — Cross-tier feature stability: what survives L0→L1 but breaks at L1b?** *[Tier 2]*
- **What it measures:** which features are robust to *uninformative* renaming (L1) but break under *misleading* renaming (L1b) — isolating semantic vs lexical dependence.
- **Block Model level:** Atoms × (Text-surface vs Function).
- **Maps to:** the L1 vs L1b distinction central to your design; Paper 3's finding that L1b's pooled effect cancels by item.
- **Method/metric:** For each feature active at identifier positions on L0, measure activation persistence across L1 and L1b (Jaccard/cosine of activation patterns). Classify features as lexical (break at L1), semantic-robust (survive both), or decoy-captured (flip at L1b). Relate the decoy-captured fraction to ISF and HCI.
- **Confirms if:** a well-defined feature class flips specifically at L1b and correlates with ISF/HCI. **Falsifies if:** L1 and L1b are indistinguishable at the feature level.
- **Feasibility:** ✅ Forward-pass only; Llama-3.1-8B/Qwen3. Natural companion to E1. ~1–2 weeks.
- **Risks:** same transfer/absorption caveats as E1.

---

**E7 — Cross-instrument validation: SAE features vs ISF and A_id/A_struct.** *[Tier 3, triangulation — highest value-per-cost]*
- **What it measures:** whether SAE feature-space measures corroborate the attention-based A_id/A_struct and the generation-uncertainty ISF metrics — a convergent-validity check across all three instruments.
- **Block Model level:** cross-cutting.
- **Maps to:** ISF (Paper 2); the H1–H4 attention hypotheses (Instrument 1).
- **Method/metric:** Per item, compute correlations among (a) identifier-feature mass from SAEs, (b) A_id/A_struct from attention, (c) ISF from generation, (d) NLA verbalizer flags. Test whether SAE feature-space deltas Δ(L3−L2), Δ(L3−L1b) discriminate H1–H4 independently of attention (e.g., does identifier-feature mass rise (H1) or fall (H2) at L3?).
- **Confirms if:** the three instruments agree on the H1–H4 verdict (strong result) OR disagree in an interpretable way. **Falsifies if:** measures are mutually uncorrelated noise.
- **Feasibility:** ✅ Pure analysis on already-collected activations; cheap. ~1 week. **This is what makes the SAE work a genuine third instrument rather than a standalone — run it regardless.**
- **Risks:** correlation ≠ shared mechanism; pre-register which agreement pattern supports which hypothesis.

---

**E8 — Feature-level correlates of the CoT reasoning plateau (~2048 tokens).** *[Tier 3, opportunistic]*
- **What it measures:** whether feature-space trajectories saturate at the same ~2048-token System-2 budget where Paper 3 found accuracy saturation, and whether NLA-read features diverge from displayed CoT (feature-level CoT faithfulness).
- **Block Model level:** Macro Structure × Function.
- **Maps to:** Paper 3's ~2048-token System-2 plateau; NLA E5 (internal vs displayed thinking).
- **Method/metric:** On reasoning models (DeepSeek-R1-Distill, Qwen3), track dispatcher-state-feature fidelity (E4) and decoy/true-feature mass (E1) as a function of generated reasoning-token count; test for a plateau. Cross-check whether internally active features match the displayed CoT content.
- **Confirms if:** feature-fidelity plateaus co-locate with the accuracy plateau. **Falsifies if:** no plateau or no relationship.
- **Feasibility:** ⚠️ Reasoning-Qwen models lack pretrained SAEs — needs the R1-Distill-Qwen-1.5B EleutherAI k-SAE or training; long-generation forward passes are costlier. ~3 weeks.
- **Risks:** confounds between token budget and problem difficulty; SAEs on reasoning traces are under-explored.

---

**Recommended committed instrument:** **E1 + E2 + E3 (proof-of-concept on a circuit-tracer-supported small model) + E7.** This gives you (i) a descriptive feature diff, (ii) a causal steering test, (iii) a circuit-level structural test, and (iv) cross-instrument triangulation — all mapped to your strongest prior findings and all feasible on Llama-3.1-8B plus a supported small model within budget. E4 is a cheap, panel-agnostic add (no dictionary needed) that strengthens the dispatcher story.

### 6. Risks, limitations & critiques to anticipate

1. **The SAE-utility critique.** Kantamneni et al. (ICML 2025: "SAE probes underperform the baseline of logistic regression in each regime") and Wu et al./AxBench (ICML 2025: "prompting outperforms all existing methods") show SAEs losing to baselines on probing and steering. **Mitigation:** frame SAEs as *discovery* tools; always include dense-probe / dense-steering / prompting baselines (E2, E5); a negative result vs baselines is publishable in this contested area.
2. **Interpretability illusion.** Auto-interp is recall-biased (Bills et al.; Bolukbasi et al. 2021); features on *random* models look interpretable (Heap et al.). **Mitigation:** causal validation (E2), specificity tests (RAVEL), neighbor/precision scoring, human spot-checks.
3. **Error nodes and "dark matter."** SAE error accounts for 1–15% of variance (Marks et al.); in attribution graphs error nodes "carry a non-trivial share of causal influence — especially in complex, rare, or out-of-distribution prompts." Obfuscated code is exactly that OOD regime. **Mitigation:** report error-node mass per graph; treat high-error graphs as low-confidence; use skip transcoders (lower error).
4. **Features may not be causal mechanisms.** "Causality is Key for Interpretability Claims to Generalise" (arXiv:2602.16698); counterfactuals are ambiguous (arXiv:2407.04690). **Mitigation:** intervene in the *original* model and check output effects (Anthropic's validation protocol).
5. **Feature absorption/splitting/hedging** corrupt the clean decoy/true-feature separation E1 relies on. **Mitigation:** Matryoshka SAEs (best disentanglement on SAEBench), multiple widths, absorption-metric reporting.
6. **Base-vs-fine-tuned transfer.** Most pretrained SAEs are on *base* models; your panel is Instruct/reasoning-tuned. Llama Scope explicitly studies this transfer, and the code-correctness SAE work found base-model directions "retain their effectiveness after instruction-tuning" — reassuring, but still a confound. **Mitigation:** validate feature labels on the actual checkpoint; report reconstruction quality on your data.
7. **Cross-model dictionary incomparability** (E5). **Mitigation:** RSA/CKA and probing as primary; SAE features as illustration.
8. **Fast-moving, single-lab results.** CLTs, Matryoshka SAEs, the AxBench/Kantamneni critiques, and NLAs are all 2024–2026 and mostly single-lab or preliminary. **Mitigation:** flag maturity explicitly (§7).

**Triangulation design.** Treat the three instruments as convergent evidence: attention (where the model looks), NLAs (what it says it represents), SAE/transcoders (what features/circuits are active and causal). E7 formalizes this. A hypothesis (e.g., H1 identifier-focused) is strongly supported only if attention (A_id ↑), NLA (verbalizes identifier reliance), and SAE (identifier-feature mass ↑, causally load-bearing via E2) agree. Disagreements are diagnostic, not failures.

### 7. Maturity / uncertainty flags
- **Well-established:** SAEs find more monosemantic features than raw neurons; the sparsity–fidelity tradeoff; TopK/JumpReLU/BatchTopK reconstruction rankings; attribution-graph *methodology* replicates across labs (Anthropic, EleutherAI, Goodfire).
- **Contested / preliminary:** whether SAEs beat baselines on downstream tasks (strong 2025–2026 negative results); whether SAE features are "canonical" or causal; the role/irreducibility of error nodes; Matryoshka superiority (holds at 65k width on Gemma-2-2B, may not at small scale).
- **Single-lab / very new:** CLTs and the "Biology of an LLM" findings (Anthropic); NLAs (Anthropic 2026); matched-snippet attribution-graph comparison (E3) is, to my knowledge, novel — a genuine contribution but unproven.
- **Resolved during this research:** the Gemma Scope "over 20% of GPT-3 training compute" figure and the 4–16B-tokens-per-SAE figure are confirmed verbatim in arXiv:2408.05147v2; the OpenAI 16M-latent SAE was trained on 40B tokens (arXiv:2406.04093). Llama Scope's exact per-SAE token count is not stated verbatim in the paper (SlimPajama data, 1024-token context, bfloat16 are confirmed) — verify before quoting a hard number.

### Key references (for follow-up)
- Cunningham et al. 2023, arXiv:2309.08600 · Gao et al. 2024, arXiv:2406.04093 · Bussmann et al. (BatchTopK) 2024, arXiv:2412.06410 · Rajamanoharan et al. (JumpReLU) 2024, arXiv:2407.14435 · Chanin et al. (Absorption) arXiv:2409.14507; (Hedging) arXiv:2505.11756 · Karvonen et al. (SAEBench) ICML 2025 · Kantamneni et al. arXiv:2502.16681 · Wu et al. (AxBench) arXiv:2501.17148 · Dunefsky et al. (Transcoders) arXiv:2406.11944 · Paulo et al. (Skip transcoders) arXiv:2501.18823 · Ameisen/Lindsey et al. (Circuit Tracing / Biology of an LLM) transformer-circuits.pub 2025 · Marks et al. (Sparse Feature Circuits) ICLR 2025 · Conmy et al. (ACDC) 2023 · Syed et al. (EAP) BlackboxNLP 2024 · Lieberum et al. (Gemma Scope) arXiv:2408.05147 · He et al. (Llama Scope) arXiv:2410.20526 · Qwen-Scope arXiv:2605.11887 · Code-correctness SAEs arXiv:2510.02917 · Wu/Geiger/Millière (Variable binding) arXiv:2505.20896 · Li/Guo/Andreas (State tracking) arXiv:2503.02854 · Feng & Steinhardt (Entity binding) ICLR 2024 · circuit-tracer: github.com/safety-research/circuit-tracer · Neuronpedia.org.