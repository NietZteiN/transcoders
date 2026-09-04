### Target Date: 2026-09-03 (R2 results — R2-GENERIC, as predicted; belief writes move the clean trace the wrong way)

Result for the rule frozen in [`2026-09-03_clean-half-prereg.md`](2026-09-03_clean-half-prereg.md).
No GPU was used: the staged V-spec pass was never needed, because nothing reached it.

- **Hypotheses / what we're testing:** **H-R2** — with the corrupt-trace term removed and damage
  vetoed, does a **belief-shaped** write at the banked site move the model toward its own correct
  reply? CONFIRM = a belief arm is a mover, separates from matched random by non-overlapping CI, and
  clears both vetoes.

- **Setup:** `nla/src/r2_score.py` (sha256 f553fa8e77b4…), CPU only, env `nla-mi`, seed 20260724,
  10,000-resample percentile bootstrap. Input = the banked `llr_rows.jsonl` from R (job 375789),
  clean-trace rows only; behavioural veto joins `p0/{nla_steer,coverage,steerv2/run}/gemma12b/steer_results.jsonl`
  (25 of 28 arms) against `steerv2/gemma12b/run/baseline.jsonl`. Output
  `data/nla/p0/trace_llr/gemma12b/r2_stats.json`. Command:
  `python nla/src/r2_score.py` (all paths defaulted).

- **Results:**
  - **Unit and sanity:** mean **G = +0.3071** nats/token [+0.2643, +0.3549], **G > 0 on 100 %** of
    60 items → sanity **OK**. Mover threshold = max(0.10 × G, F2) = **+0.0307**.
  - **The pre-registered floor is degenerate and the fallback was used.** `noop#1 − noop#2` on the
    clean half is **exactly 0.0 on all 60 items** (max |d| = 0.0, SD = 0) — two unsteered forward
    passes are bit-identical here, so the bf16 floor the rule was designed around does not exist on
    this path. F2 fell back to **0.0100**, as the frozen rule specified. It was **not binding**:
    0.10 × G = 0.0307 > 0.0100. *(R's floor of 0.010 was the same fallback, for the same reason —
    recorded here because both entries print it as if measured.)*
  - **Verdict: `R2-GENERIC`.** Two movers, **neither belief-shaped**:

    | arm | mean M2 | CI95 | % of G | mover | separates | V-behav | banked acc |
    |---|---|---|---|---|---|---|---|
    | `P_prompt` | **+0.0694** | [+0.0544, +0.0856] | **23 %** | ✓ | n/a | ✓ | 0.63 |
    | `R_random@0.149 id_spans` | **+0.0555** | [+0.0437, +0.0685] | **18 %** | ✓ | — | ✓ | 0.58 |
    | `V3_taskvec@8.0 id_spans` | +0.0291 | [+0.0147, +0.0440] | 9 % | ✗ | ✓ | **✗** | 0.52 |
    | `V1_gloss@0.149 id_spans` | +0.0149 | [+0.0110, +0.0192] | 5 % | ✗ | ✗ | ✓ | 0.63 |
    | `V3_taskvec@0.149 id_spans` | +0.0071 | [+0.0050, +0.0096] | 2 % | ✗ | ✗ | ✓ | 0.63 |

  - **Every belief arm at the banked `last_prompt` site is negative** — the write makes the model's
    own correct reply *less* likely: V1@1.0 **−0.0300**, V2 **−0.0286**, V4@1.0 **−0.0181**,
    V3@1.0 −0.0204, V3@4.0 −0.0559, V1@4.0 **−0.1330** (−43 % of G). Only two arms in the whole
    28-arm table are positive at that site, both ≈ 0 (`V5_replace` +0.0004, `R_random@0.1` +0.0004).
  - **R's "supporting" arm inverts.** `V1_gloss@8.0 id_spans` — the arm that produced `R-1` at
    ΔM +1.486 — has **M2 = −0.4845** [−0.6574, −0.3301], **−158 % of G**, with banked accuracy
    **0.25 vs 0.633 baseline**. Its matched random control is worse still (−1.2961, −422 % of G,
    accuracy 0.20). Both fail V-behav.
  - **Matched-α comparison, the one that matters:** at α = 0.149071 on `id_spans`,
    random **+0.0555** vs V1 **+0.0149** and V3 **+0.0071** — random is **3.7×** the best belief arm.
  - **V-spec (GPU) was never run:** `pending_spec` is empty, so the staged pass had nothing to
    adjudicate. Zero GPU-hours spent on this experiment.

- **What worked / hypothesis verdict:**
  - **H-R2 ✗ REFUTED.** No belief arm is a mover; the two arms that clear the threshold are a
    **prompt change** and a **random vector**. Movement at this site is generic perturbation, and the
    belief-shaped writes are worse than doing nothing — negative at every magnitude at `last_prompt`.
  - **The pre-registered predictions held.** I predicted R2-NULL or R2-GENERIC with R2-NULL more
    likely, and stated the tie-break in advance: *"P_prompt clears the mover threshold iff mean
    G < 0.69."* **G came out 0.3071 and P_prompt cleared** — GENERIC rather than NULL, by the
    mechanism named beforehand. I also predicted R2-BELIEF would surprise me; it did not occur.
  - **The prompting baseline wins on a third independent readout.** `P_prompt` is the single best
    arm at 23 % of a prompt swap, ahead of every vector including the oracle. That is now the fourth
    time prompting has beaten steering in this thread, consistent with AxBench (Wu et al., ICML 2025).

- **Observations:**
  - The three metrics now agree and say the same thing in three different units: **accuracy** (no arm
    ever rescued > 2 of 60), **the confounded LLR** (movement tracked damage, not belief), and **the
    damage-controlled clean half** (belief arms negative, random and prompting positive). The
    disagreement between R and R2 on the *same rows* is entirely explained by the corrupt-trace term.
  - The sign pattern is the informative part. If these writes were merely too weak, belief arms would
    sit at ~0. They sit **below** 0 and get more negative with α — the write is off-manifold and costs
    the model likelihood on its own correct answer. `V1@4.0` at −43 % of G is a *large* effect in the
    wrong direction.
  - Bit-identical noop passes mean this pipeline has **no measurable run-to-run floor** on likelihood
    readouts, so likelihood CIs here are item-sampling only. That is a nice property to have
    established, and it makes the 0.01 fallback conservative rather than wrong.
  - Bounds unchanged: one host (Gemma-3-12B-it), one layer (32), one tier (L1b), n = 60, greedy
    decoding. A negative bounds the **site**, not the hypothesis that some other depth carries belief.

- **New questions / new hypotheses:**
  - **H-R3:** the `id_spans` positions are the only ones where anything is positive at small α
    (V1 +0.0149, V3 +0.0071, random +0.0555 — all > 0, unlike every `last_prompt` arm). Is that a
    genuine site difference, or just the smaller per-position perturbation that spreading α over
    45 tokens implies? Testable from banked rows by re-normalising per delivered energy — no GPU.
  - **H-R4:** random at matched energy beating the informed direction 3.7× is the same ordering
    Papers 2–3 report for dense vs sparse interventions. Worth stating as a *positive* claim about
    generic perturbation rather than only as a null about belief.
  - Retired: the "belief writes are too weak" reading. They are not too weak; they point the wrong way.

- **Next Steps:**
  1. **Do not run V-spec.** Nothing reached it; running it now would be a fishing expedition.
  2. `patch_map.py` alignment fix is the only remaining GPU item from the R/S/T family, and it is
     independent of everything here — T asks *where* the corruption lives, which no steering result
     bears on.
  3. The flippable-denominator power gate (6/60) still binds every accuracy-level claim; growing the
     corpus from Dataset B remains the prerequisite for any rescue experiment.
