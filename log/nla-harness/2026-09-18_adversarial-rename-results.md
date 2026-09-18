### Target Date: 2026-09-18 (H-R22 RESULTS — deranging every identifier costs 1.2 points, and renaming is closed as a stimulus)

- **Hypotheses / what we're testing:** as pre-registered in
  [`2026-09-18_adversarial-rename-prereg.md`](2026-09-18_adversarial-rename-prereg.md). Predictions on
  record: **a** `CORPUS-OK` · **b** `DAMAGE-SUFFICIENT` (≥ +0.05 with CI excluding 0).

- **Setup:** `ase_rename_swap.py` (sha `37d3f13c5a96f6ac…`; `ase_rename_java.py` untouched) deranges each
  file's own declared identifiers — Sattolo, one cycle, **no fixed point** — including single-character
  names and the **method name**, which appears inside every case expression. Packs rebuilt on the variants
  with the artifact's own execution-validating `build_case_pack` (job 410780, CPU, 23 min), then the
  PACKS-PAIRED gate. One greedy arm (job 410796, 13 min, g-07-12). Scored against the **already-banked**
  H-R18 greedy L0, restricted to the matched subset. Seed 20260724, `N_BOOT 10 000`.

- **Results:**

  **H-R22a `CORPUS-OK`** — 148/148 deranged, 0 skipped; **837 identifiers renamed, of which 365 are
  single-character** (the old renamer renamed **zero** of those); **0 compile or execution failures** on
  any of the 148, which is the empirical confirmation that a derangement is semantics-preserving;
  **146 of 148** survive PACKS-PAIRED (2 dropped on case count). Rebuilt packs give **11.73 cases per
  snippet against the paper's 11.72**, so the harder stimulus did not distort case construction.
  Frozen set: **146 snippets / 1 722 cases**.

  **H-R22b `DAMAGE-PRESENT-BUT-SMALL` — my `DAMAGE-SUFFICIENT` prediction is REFUTED.** Greedy,
  deterministic, matched 146 snippets, per-case parse **1.0000 in all three conditions**:

  | condition | accuracy | damage vs L0 |
  |---|---|---|
  | L0 original | 0.8531 | — |
  | L1b weak (pooled random decoys) | 0.8461 | +0.0070 [−0.0203, +0.0371] |
  | **L1b SWAP (derangement)** | **0.8409** | **+0.0122 [−0.0125, +0.0382]** |

  The derangement is harder than the weak renamer by **+0.0052 [−0.0222, +0.0354]** — a CI comfortably
  containing zero. Against the paper's Qwen2.5-7B renaming drop of **+0.3629**, the ratio is **0.034**.

  **The diagnostic that explains it** (both conditions deterministic, so this is exact, not noisy):

  | | |
  |---|---|
  | per-item agreement, L0 vs derangement | **0.9286** |
  | per-item agreement, L0 vs weak rename | 0.9292 |
  | of 1 469 L0-**correct** cases, still correct | 0.9510 |
  | of 253 L0-**wrong** cases, now correct | 0.2016 |
  | total label flips | 123 (**7.1 %**) for a net −1.2 points |

- **What worked / hypothesis verdict:**
  - **H-R22a SUPPORTED.** The renamer works, is validated by execution, and is strictly stronger than its
    predecessor on every count that was diagnosed as weak.
  - **H-R22b REFUTED my prediction.** I argued the derangement would clear +0.05 because it attacks
    state-tracking and the call site at once and is adversarial by construction rather than by vocabulary.
    It moved the damage from +0.0070 to +0.0122 — **an increase of half a point, with the CI spanning
    zero.** The reasoning was wrong.
  - **The pre-committed consequence is honoured: renaming is CLOSED as a stimulus for this model.** No
    steering arm will be run on an identifier-renaming corpus for CodeLlama-7B-Instruct. **H-R3 is also
    closed by this** — it proposed exactly "a stronger renamer" as the way to reach their damage, and the
    strongest renaming available (vocabulary-preserving, everything deranged) gets 3.4 % of their drop.

- **Observations:**
  - **This is the substantive finding, not the null.** The model gives the **same answer 92.9 %** of the
    time after every identifier in the program has been rewired, and the 7.1 % that flip are
    **near-symmetric** (it loses 4.9 % of what it had right and rescues 20.2 % of what it had wrong, for a
    net −1.2 points). **CodeLlama-7B-Instruct is essentially not using identifier semantics for
    output prediction on this corpus — it traces the code.** That is a positive claim about the model, and
    it is measured on the cleanest possible manipulation: the *set* of identifier names is identical in
    both conditions, so nothing here can be attributed to unfamiliar or out-of-distribution tokens.
  - **It also explains every null in this block at once.** If names carry ~1 point of accuracy, then no
    identifier-targeted intervention — ours or CodeSteer's — can recover more than ~1 point, which is
    exactly what the oracle bound said independently (+0.0052 for writing the *true* clean state at L7).
    Two unrelated measurements now agree on the ceiling.
  - **Agreement with the weak renamer is 0.9292 vs the derangement's 0.9286** — statistically the same.
    The model is equally indifferent to random decoy names and to a full role permutation, which is
    stronger evidence of indifference than either number alone.
  - **What this does NOT establish.** Their generator is still not ours, this is one model outside their
    panel, and only renaming was tested — control-flow flattening (L2/L3) is untouched and attacks a
    different route (relational overload, hidden-state simulation) that this result says nothing about.
    **Their 36-point drop is not contradicted**; what is now well supported is that it does not
    generalise to this model, and that the mechanism it is attributed to (reliance on lexical cues) is
    absent here.
  - Reusing the banked L0 arm meant the primary question cost **13 minutes of GPU**. The expensive part
    was CPU pack validation.

- **New questions / new hypotheses:**
  - **H-R21 RESOLVED in the "different stimulus or host" direction.** Renaming cannot host a steering
    experiment on this model. Two options remain, and they are not equivalent: **(i)** control-flow
    flattening (L2/L3), which attacks relational overload rather than atom-level interference and is the
    route the charter's H-R21/E3 actually cares about — but the generator is **not** in the artifact and
    would have to be written and equivalence-checked, a substantial build; **(ii)** a different host from
    the permitted panel, which tests whether the paper's effect exists on *any* model we may run.
  - **H-R23 (cheap, and now the highest-value next run):** (ii) before (i). Run L0 vs the derangement on
    the permitted panel — Llama-3.1-8B, CodeGemma-7B, Phi-3.5-mini, SmolLM3-3B — greedy, one arm each,
    ~1 GPU-h total. If **no** permitted model shows a renaming deficit, the ASE replication is finished as
    far as we can take it and the honest write-up is "their effect does not reproduce on any model we are
    allowed to run"; if one does, that model becomes the host for the steering arms and the whole
    programme moves there. This subsumes H-R17 and is a precondition for spending a flattening build.
  - **H-R20 unchanged** and now more interesting: the ~16-point greedy gain dwarfs every stimulus effect
    measured here.

- **Next Steps:** H-R23 (panel damage screen under greedy) before any flattening build or further
  steering work.
