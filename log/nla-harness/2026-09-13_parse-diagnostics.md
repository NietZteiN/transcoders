### Target Date: 2026-09-13 (H-A4 + H-S15, no GPU — where the tier gradient and the `edit_single` harm actually live)

- **Hypotheses / what we're testing:** two free follow-ups raised by today's two results entries, both answerable
  from rows already on disk.
  - **H-A4** ([`2026-09-13_tier-ladder-results.md`](2026-09-13_tier-ladder-results.md)): is the tier ladder a
    comprehension effect or an **output-budget** effect? Predictions: prompt length predicts parse rate *within
    item*; and if truncation drives H-A3, dropping the items that never parse should **weaken** `ROUTES-COMPOUND`.
  - **H-S15** ([`2026-09-13_accuracy-results.md`](2026-09-13_accuracy-results.md)): are the 12–13 items
    `edit_single` broke **unparseable output** or **parseable wrong answers**? Prediction not recorded in advance
    beyond the question; the two possibilities mean different things — formatting damage vs corrupted computation.
  - **Both are diagnostics on published verdicts, not new rules. No frozen threshold is touched and no verdict is
    revised**; where they change the *emphasis* of an already-published entry that is said explicitly below, in
    this new entry, per the append-only protocol.

- **Setup:** no GPU, no model load. Inputs are the banked rows from the two jobs reported earlier today:
  `data/nla/ml/gemma4b/gate/tier_ladder/tier_rows.jsonl` (job 392094) and
  `data/nla/ml/gemma4b/gate/accuracy{,_greedy}/accuracy_rows.jsonl` (job 391968). Bootstrap reused unchanged from
  `nla/src/dose_score.py` (`boot`, N_BOOT 10 000, seed 20260724); Spearman via `scipy.stats`. Run on the login node
  with `TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1` (the rayon thread-pool panic guard).

- **Results.**

  **H-A4a — prompt length tracks parse failure, and it survives removing item identity:**

  | tier | mean prompt_len | sampled parse rate | rate | rate \| parsed |
  |---|---|---|---|---|
  | L0 | 230.3 | 0.867 | 0.588 | 0.658 |
  | L1 | 270.9 | 0.771 | 0.504 | 0.647 |
  | L1b | 265.4 | 0.785 | 0.540 | 0.627 |
  | L2 | 362.9 | 0.760 | 0.498 | 0.593 |
  | L3 | 394.6 | 0.698 | 0.438 | 0.598 |

  Pooled Spearman(prompt_len, parse_rate) over the 300 tier-items: **ρ = −0.483, p = 5.9 × 10⁻¹⁹**.
  **Within-item centred** (each item's own 5 tiers, so item identity — 56 % of the variance in this corpus per
  H-W16 — is differenced out): **ρ = −0.305, p = 6.8 × 10⁻⁸**.

  **H-A4b — dropping the never-parse items does not weaken `ROUTES-COMPOUND`; it strengthens it.** Excluding the
  10 items whose L2 or L3 sampled parse rate is exactly 0:

  | contrast | all 60 | kept 50 |
  |---|---|---|
  | `L3 − L2` | −0.060 [−0.110, −0.013] | **−0.075 [−0.133, −0.020]** |
  | `L0 − L2` | +0.090 [+0.023, +0.165] | +0.085 [+0.010, +0.165] |
  | `L1 − L1b` | −0.035 [−0.100, +0.027] | −0.030 [−0.105, +0.038] |

  **H-S15 — the `edit_single` harm is wrong answers, not unparseable output.** Per arm on the sampled pass:

  | arm | sampled parse rate | Δ parse vs `noop` | rate | rate \| parsed |
  |---|---|---|---|---|
  | `noop` | 0.781 | — | 0.527 | 0.634 |
  | `c3_band` | 0.779 | −0.002 [−0.054, +0.044] | 0.521 | 0.636 |
  | `edit_band` | 0.783 | +0.002 [−0.038, +0.040] | 0.535 | 0.680 |
  | `foreign_band` | 0.802 | +0.021 [−0.010, +0.052] | 0.460 | 0.539 |
  | **`edit_single`** | **0.802** | +0.021 [−0.031, +0.073] | 0.406 | **0.468** |
  | `prompt` | 0.787 | +0.006 [−0.019, +0.031] | 0.537 | 0.643 |

  Of the items `edit_single` broke: **7 of 12 parseable-but-wrong** in the sampled pass's greedy, **10 of 13** in
  the standalone greedy pass. Comparison arms broke fewer and almost never by non-parse: `c3_band` 5 broken
  (1 unparseable), `edit_band` 5 (1), `foreign_band` 10 (1), `prompt` 3 (**0**).

- **What worked / hypothesis verdict:**
  - **H-A4a SUPPORTED.** Prompt length predicts parse failure both pooled (ρ = −0.483) and, decisively, *within
    item* (ρ = −0.305, p = 6.8 × 10⁻⁸) — so it is not a between-item artifact. L2/L3 prompts run 363/395 tokens
    against L0's 230, and the parse rate falls monotonically with them.
  - **H-A4b REFUTED in the direction that matters, and this is good news for a published verdict.** If truncation
    manufactured `ROUTES-COMPOUND`, dropping the never-parse items should have collapsed it; instead `L3 − L2`
    goes from −0.060 to **−0.075** and both other contrasts are unchanged inside their CIs. **`ROUTES-COMPOUND` is
    not carried by the items that fail to answer.**
  - **H-S15 answered: corrupted computation, not formatting.** `edit_single` has the *highest* parse rate of any
    arm (0.802, Δ +0.021 with the CI containing 0) while its accuracy **among parsed output** falls to 0.468
    against `noop`'s 0.634, and 7/12 (10/13 in the other pass) of the items it broke were parseable and wrong.
    **Full single-layer replacement makes the model answer confidently and incorrectly** — it damages the
    computation, not the model's ability to emit an answer. Same shape for `foreign_band` (0.539 \| parsed).

- **Observations:**
  - **Emphasis correction to [`2026-09-13_tier-ladder-results.md`](2026-09-13_tier-ladder-results.md), stated here
    rather than by editing it.** That entry's post-hoc table used *greedy*-conditioned parsing and read the
    accuracy-among-parsers column as "flat within noise (0.619–0.708)". On the sampled per-item ratio the column
    is **not flat**: it declines 0.658 → 0.598 from L0 to L3. Decomposing the L0→L3 rate drop of −0.150: parse
    rate contributes −0.169 of *rate capacity* while conditional accuracy contributes −0.060. So the honest
    statement is **roughly two-thirds output-budget, one-third comprehension** — not "mostly parse failure".
    The numbers in that entry are correct as computed; the reading of them was too strong.
  - **The H-A5 requirement is downgraded, not cancelled.** That entry said H-A5 (L3 at `MAX_NEW_GEN` 2 048)
    "must run before `ROUTES-COMPOUND` is cited anywhere". H-A4b has now tested the concern the cheap way and
    `ROUTES-COMPOUND` survived it, so the verdict may be cited with the length caveat attached. H-A5 remains the
    clean test — dropping zero-parse items is not the same as removing truncation *pressure*, since partially
    truncating items stay in — and it is still worth 0.5 GPU-h, but it is no longer a gate.
  - **The two jobs fail in different ways, which is the useful contrast.** Obfuscation makes the model *not
    answer* (parse rate 0.867 → 0.698 with prompt length); steering it with a full-replacement write makes it
    *answer wrong* (parse rate highest of any arm, conditional accuracy lowest). Anything that reports a single
    accuracy number for both conflates two mechanisms.
  - `edit_band`'s conditional accuracy (0.680) is the highest of any arm including `noop` (0.634) and `prompt`
    (0.643). **Not claimed as an effect** — conditioning on parse success is a collider, the item sets differ per
    arm, and the unconditional paired delta was +0.008 [−0.038, +0.056]. Noted only because it is the one place
    the β-band write looks better than baseline, and H-S13 will touch it.

- **New questions / new hypotheses:**
  - **H-A7:** length is now a measured confound in the *tier* ladder, so it is one in Papers 2–3's behavioural
    tables too, where L2/L3 prompts are likewise longer. A length-matched control (truncate/pad the L0 prompt to
    the L2 token count with semantically inert filler) would separate "flattening is hard" from "flattening is
    long". No GPU for the design; ~0.5 GPU-h to run. **This is a claim about the measurement instrument and it
    belongs in the paper whichever way it lands.**
  - **H-S16:** `edit_single` answers confidently and wrongly — so is it *decoy-consistent* wrongness? Regrade its
    7–10 parseable-wrong items against the **decoy** semantics rather than the truth. If the model returns the
    decoy's answer, full replacement is installing the wrong belief rather than destroying the computation, which
    is the programme's originating question ("wrong belief or missing one?") answered from the steering side. Free.
  - Does the within-item length→parse slope differ by tier family (rename vs flatten)? Free, and it decides
    whether H-A7 needs one control or two.

- **Next Steps:** job **392723** (erasure vector) is running; report it when it lands. **H-S16** is free and is the
  most interesting question in this entry. H-A5 and H-A7 are ~0.5 GPU-h each and no longer block anything.
