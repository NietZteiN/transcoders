### Target Date: 2026-07-24 (Register foundational papers)
- **Hypotheses / what we're testing:** None — organizational. Goal: register the three papers this project builds on so they are citable from experiment notes and any write-up.
- **Setup:** Working dir `transcoders/` on `csr-94608.utdallas.edu`. No GPU used (bookkeeping only). Actions:
  - Created `papers/` and moved the three source PDFs into it with clean, referenceable filenames (paper number + citation key + arXiv id).
  - Extracted bibliographic metadata with `pdfinfo` / `pdftotext`.
  - Wrote [`../../papers/references.bib`](../../papers/references.bib) (BibTeX) and [`../../papers/REFERENCES.md`](../../papers/REFERENCES.md) (bibliography + summaries + how the project builds on each).
- **Results:** Three foundational papers registered:

  | # | Cite key | Title | arXiv / status |
  |---|----------|-------|----------------|
  | 1 | `nguyen2026obfuscation` | The Effect of Code Obfuscation on Human Program Comprehension | arXiv:2603.07668 (8 Mar 2026) |
  | 2 | `le2026machines` | Do Machines Struggle Where Humans Do? LLM and Human Comprehension of Obfuscated Code | arXiv:2606.31725 (30 Jun 2026) |
  | 3 | `dualprocess2026` | Fast Errors or Slow Effort? Dual-Process Signatures in Human and LLM Code Understanding | under double-blind review |

  File moves (original → new):
  - `2603.07668v1 (paper 1).pdf` → `papers/paper1_Nguyen2026_ObfuscationHumanComprehension_arXiv-2603.07668.pdf`
  - `2606.31725v1 (paper 2).pdf` → `papers/paper2_Le2026_LLMvsHumanObfuscatedCode_arXiv-2606.31725.pdf`
  - `Model_Human_Obfuscation_Code_Reasoning (paper 3).pdf` → `papers/paper3_DualProcess_FastErrorsSlowEffort.pdf`
- **What worked / hypothesis verdict:** N/A (organizational). Papers are now citable by BibTeX key.
- **Observations:** The three papers form one research line: human comprehension of obfuscated code (1) → human↔LLM alignment on the same task (2) → dual-process mechanistic account of *how* obfuscation breaks comprehension (3). Paper 3 is anonymized for review but internally cites Paper 1 (`[1]`) and Paper 2 (`[3]`), confirming the lineage. Shared scaffold across all three: obfuscation tiers (L0 / renaming / adversarial renaming / control-flow / combined), Python + JavaScript function-level tasks, output-prediction as the comprehension probe.
- **New questions / new hypotheses:** How does the transcoder / mechanistic-interpretability angle attach to this line — what internal signals to probe under each obfuscation tier? (Answered the same day — see [`2026-07-24_charter-and-scaffold.md`](2026-07-24_charter-and-scaffold.md).)
- **Next Steps:** Write the project charter (`CLAUDE.md`) once the goal artifacts are in hand.
