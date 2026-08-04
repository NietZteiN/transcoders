# Foundational Papers — transcoders project

The three papers this research project builds on. PDFs live in this folder;
BibTeX keys are in [`references.bib`](references.bib). Cite by the **key** column.

| # | Key | Title | PDF |
|---|-----|-------|-----|
| 1 | `nguyen2026obfuscation` | The Effect of Code Obfuscation on Human Program Comprehension | [paper1](paper1_Nguyen2026_ObfuscationHumanComprehension_arXiv-2603.07668.pdf) |
| 2 | `le2026machines` | Do Machines Struggle Where Humans Do? LLM and Human Comprehension of Obfuscated Code | [paper2](paper2_Le2026_LLMvsHumanObfuscatedCode_arXiv-2606.31725.pdf) |
| 3 | `dualprocess2026` | Fast Errors or Slow Effort? Dual-Process Signatures in Human and LLM Code Understanding | [paper3](paper3_DualProcess_FastErrorsSlowEffort.pdf) |

---

## Paper 1 — `nguyen2026obfuscation`
**The Effect of Code Obfuscation on Human Program Comprehension**
Anh H. N. Nguyen\*, Jack Le\*, Ilse Lahnstein Coronado, Tien N. Nguyen — University of Texas at Dallas
arXiv:2603.07668v1 [cs.SE], 8 Mar 2026 · 21 pp · *(\* equal contribution)*
Original file: `2603.07668v1 (paper 1).pdf`

The **human baseline**. A controlled output-prediction study of how obfuscation
affects human program comprehension across obfuscation tiers — identifier renaming,
adversarially misleading identifiers, control-flow modification, and combinations —
applied to function-level Python and JavaScript. Measures correctness, response time,
and self-reported experience. Key finding: obfuscation generally raises reasoning time
and lowers accuracy, but the strength→difficulty relationship is **not monotonic** and
varies by language (JavaScript follows the expected trend; some Python renamings match
or beat the unobfuscated baseline). Response-time analysis suggests obfuscation shifts
readers from fast heuristic reasoning toward slower deliberate reasoning.

## Paper 2 — `le2026machines`
**Do Machines Struggle Where Humans Do? LLM and Human Comprehension of Obfuscated Code**
Jack Le\*, Anh H. N. Nguyen\*, Tien N. Nguyen — University of Texas at Dallas
arXiv:2606.31725v1 [cs.SE], 30 Jun 2026 · 13 pp · *(\* equal contribution)*
Original file: `2606.31725v1 (paper 2).pdf`

Extends Paper 1 to **machines**. Evaluates several LLM classes (reasoning-tuned, coder,
instruction-tuned) on obfuscated Python/JavaScript across five obfuscation tiers
(identifier renaming, adversarial renaming, control-flow flattening, combinations),
analyzing both output-prediction performance and reasoning traces. Interprets failures
through **Schulte's Block Model** (atom / block / relational / macro levels) and asks
whether LLM difficulty patterns mirror the human patterns from Paper 1 — including
effort that scales with difficulty and confident misinterpretation under misleading
identifiers. Uses Paper 1 as the human baseline for alignment.

## Paper 3 — `dualprocess2026`
**Fast Errors or Slow Effort? Dual-Process Signatures in Human and LLM Code Understanding**
Anonymous (double-blind review) · 12 pp
Original file: `Model_Human_Obfuscation_Code_Reasoning (paper 3).pdf`

Synthesizes and deepens Papers 1 & 2 (cited internally as its `[1]` and `[3]`). Tests a
**dual-process account**: adversarial identifier renaming acts as Stroop-like
interference (detected and suppressed at the cost of a strategy switch) while
control-flow obfuscation causes System-2 overload (degradation through sustained
tracing) — distinct cognitive routes rather than one mechanism. Study of 73 participants
across JavaScript/Python experience measuring accuracy, completion time, confidence,
calibration, and answering strategy across tiers (L0, L1b, L2) under time-pressure and
concurrent-cognitive-load manipulations, then tests whether LLMs reproduce the same
accuracy, confidence, and failure patterns.

---

## How the transcoders project builds on these
The `transcoders/` project is **Instrument 3 (SAE + transcoder feature & circuit analysis)**
of the mechanistic follow-up *"Opening the Black Box of Obfuscated-Code Comprehension"* — it
moves from the **behavioral** findings of Papers 1–3 to the **internal mechanism**. Full goal
in [`../CLAUDE.md`](../CLAUDE.md) §3; experiment menu in
[`../docs/experiment_menu.md`](../docs/experiment_menu.md).

- **Papers 1 → 2 → 3** form one line: human comprehension of obfuscated code (1),
  human↔LLM alignment on the same task (2), and a dual-process mechanistic account of
  *how* obfuscation breaks comprehension in both (3).
- **Shared experimental scaffold reused directly:** obfuscation tiers
  (**L0 / L1 / L1b / L2 / L3**), Python + JavaScript function-level tasks, output-prediction
  as the comprehension probe, and the Papers 2–3 model panel + GLMM statistical stack.
- **Interpretive lens:** **Schulte's Block Model** (Paper 2) is the shared coordinate system;
  **dual-process / System 1–2 theory** (Paper 3) frames the two failure routes.
- **Specific prior findings the mechanistic instrument targets:** the L1b **HCI collapse**
  (acc 21.25 % / HCI 10.0 %) → E1/E2 (semantic-capture feature diff + causal steering); the
  dispatcher-complexity **r = −0.196** under control-flow flattening → E3 (attribution graphs)
  and E4 (state-binding probe); the **ρ = 0.30–0.47** reasoning-vs-≈0 alignment split → E5;
  the **~2,048-token** System-2 plateau → E8.
