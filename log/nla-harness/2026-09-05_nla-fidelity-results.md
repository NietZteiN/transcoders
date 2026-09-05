### Target Date: 2026-09-05 (H-W7 — the NLA round trip preserves **98.3 %** of an activation's causal power, and beats prompting: the programme's first)

**Thread:** nla-harness · **Job:** 377835 (h200, **1:13:28**, COMPLETED, 0 errors) · **Host:** `gemma12b`
L32/48 · **Resolves:** [`2026-09-04_nla-fidelity-prereg.md`](2026-09-04_nla-fidelity-prereg.md).

- **Hypotheses:** H-W7a (C3pure beats prompting), H-W7b (causal fidelity `C3pure/P_patch` ≥ 0.50),
  H-W7c (C3pure beats the round-trip null by ≥ +12.11 nats).

- **Setup:** three arms at the **375 locatable spans**, identical positions, `PositionReplacer`
  (norm-matched, direction-only, **no α**). `C3pure` = AR(AV(h_L0)); `P_patch` = h_L0 itself;
  `C1r` = AR(banked stage-0 L1b read). n = **49 items / 1,206 positions** — 11 of 60 items have no
  locatable span at all. Seed 20260724, 10,000 bootstrap, `--deterministic` OFF. Artifacts
  `data/nla/p0/fidelity/gemma12b/fidelity_{rows.jsonl,stats.json}`.

- **Results.** The `G_sum` unit **recomputed on these 49 items is +111.57 nats** (the 60-item value
  is +121.13); every percentage below uses the 49-item unit, and the banked benchmarks are
  recomputed on the same 49.

  | arm | G_sum | 95 % CI | % of prompt swap |
  |---|---|---|---|
  | **C3pure** — clean state → English → back | **+44.94** | [+40.25, +49.90] | **40.3 %** |
  | **P_patch** — clean state written directly | **+45.71** | [+40.82, +50.88] | 41.0 % |
  | C1r — round-trip null (L1b read) | +9.30 | [+7.45, +11.25] | 8.3 % |
  | *banked* prompting, same 49 | *+27.66* | — | *24.8 %* |
  | *banked* random @ id_spans, same 49 | *+20.04* | — | *18.0 %* |

  **Causal fidelity `C3pure/P_patch` = 0.983 [0.967, 0.999].** Paired `C3pure − P_patch` =
  **−0.77 nats** [−1.55, −0.02]; per-item ratio median 0.989 (IQR 0.944–1.032).
  `C3pure − C1r` = **+35.64** [+31.15, +40.35].
  Behaviour: baseline acc 0.714 / parse 0.857; C3pure 0.694/0.878, P_patch 0.714/0.918,
  C1r 0.673/0.857 — **no arm trips the veto.**
  **Verdict `W7-CHANNEL-STRONG-AND-FAITHFUL`; all three gates pass.**

- **What worked / hypothesis verdict.**
  - **H-W7a ✓** — C3pure's CI lower bound **+40.25** clears prompting (+27.66) outright. **This is
    the first intervention in this entire programme to beat the prompting baseline**, which has
    dominated every steering arm since Paper 2.
  - **H-W7b ✓, far above the bar** — 0.983 against a frozen 0.50. **Turning an activation into
    English and reconstructing it costs ~1.7 % of its causal effect** (0.77 of 45.71 nats). The
    loss is statistically detectable (CI just excludes 0) and practically negligible.
  - **H-W7c ✓** — +35.64 against a +12.11 bar. The effect is the *content*, not the operator:
    the same generator (AR), same positions, same operator, fed a description of the **obfuscated**
    state instead of the clean one, yields **+9.30** — a fifth as much.

- **Observations, including what this does NOT show.**
  - **The headline restated carefully:** the NLA can carry an activation's causal content through
    natural language almost losslessly. The released pair is scored on `MSE = 2(1−cos)` —
    *reconstruction* fidelity — and nothing in the NLA literature scores **causal** fidelity. This
    is that number, for one layer, one model, one regime.
  - **⚠️ 98.3 % causal fidelity and stage 0's raw cosine 0.9877 are the same fact viewed twice, not
    independent confirmations.** `PositionReplacer` writes `‖h‖·unit(v)`, so C3pure and P_patch
    differ only by the angle between `AR(AV(h))` and `h` — about 9° in raw space. A near-lossless
    causal effect is what that angle *predicts*. The result is real but it is not a second,
    independent line of evidence, and it must never be reported as one.
  - **⚠️ The decisive null is still missing.** A **foreign clean activation** — `h_L0` from a
    *different* item at these positions — is not in this design. `C1r` covers part of the worry (an
    AR output from the same generator gives 8.3 %), but C1r differs in *content* as well as in
    *source*. Until the foreign-clean arm runs, "the NLA preserves item-specific causal content"
    outruns the evidence; what is established is "clean-state-derived directions work and
    obfuscated-state-derived ones do not".
  - **⚠️ n = 49, and it is the easier subset.** 11 items have no locatable span. Baseline accuracy
    on these 49 is **0.714** against 0.633 on the full 60, and the swap unit is 111.57 vs 121.13.
    Nothing here is directly comparable to stage 1's 60-item numbers.
  - **The dissociation is the finding for the paper.** A **41 %-of-swap** likelihood shift toward the
    clean reply moves accuracy from 0.714 to **0.694**. Restoring the clean identifier
    representations — 98 % faithfully, at every locatable span — **does not restore correctness**.
    Partly the measurement: only **3 of 49** items are flippable, so a perfect rescue is +0.061 and
    this design cannot see it. But the direction is telling, and it says the adversarial trap's
    damage is **not confined to identifier positions** — consistent with L3's `decoy_overlap` result
    and with Papers 2–3 describing L1b as semantic displacement rather than local mislabelling.

- **New questions / new hypotheses.**
  - **H-W9 (now the top priority, was H-W6):** foreign clean activation at the same positions. If it
    matches C3pure, the effect is "any clean-code direction", not item-specific content, and the
    fidelity claim collapses to a statement about the operator.
  - **H-W10:** if H-W9 clears, the paper's frame is **"the NLA channel is causally faithful; what
    fails is editing in language"** — the natural next test is *where* the edit fails, using H-W3's
    finding that reads describe these positions in surface terms and reach the identifier mainly by
    quoting it.
  - **H-W11:** run the accuracy question on a corpus with a real flippable denominator (Dataset B).
    Every accuracy statement in this thread is bounded by 3–6 flippable items.

- **Next Steps:** pre-register and run **H-W9**. No claim about item-specific fidelity leaves this
  thread until it lands.
