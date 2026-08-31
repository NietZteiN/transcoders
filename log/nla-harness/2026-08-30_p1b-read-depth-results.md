### Target Date: 2026-08-30 (P1b read-depth probe — the tier contrast is vacuous, and correctness runs the other way)

Pre-registration: [`2026-08-30_p1b-read-depth-prereg.md`](2026-08-30_p1b-read-depth-prereg.md),
frozen before any activation was extracted. **The pre-registered primary rule returned
NOT REPORTABLE, and the reason was a defect in a control I wrote.** That is recorded first below,
before any result, because it conditions how the rest should be read.

- **Setup:**
  ```
  probe   job 359103, h200 g-07-08, 00:01:36 · nla/src/p1b_read_probe.py
  perm    job 359112, normal partition, 00:00:44 · nla/src/p1b_perm_control.py (corrected control)
  acts    one forward pass per prompt, output_hidden_states, final prompt token
          60 pairs x 2 tiers = 120 vectors x 28 layers x 3584 dims -> data/nla/p0/p1b/acts.npz
  probe   logistic, L2, C = 1.0 frozen · GroupKFold(5) on snippet_id · bootstrap over snippets
  labels  correctness reused from the P0.4 L20 baseline, so this is purely read-side
  ```
  Reads are bit-exact ([`2026-08-30_reads-exact-truncation-ruled-out.md`](2026-08-30_reads-exact-truncation-ruled-out.md)),
  so this measurement carries no reproducibility caveat.

- **The control defect, stated plainly.**

  The frozen rule required the **maximum** single-draw permutation AUC over **28 layers** to stay
  below 0.60. That is a threshold calibrated for one draw applied to a max-over-28 statistic — a
  multiple-comparisons error in the control itself, and mine. It fired on L17 at 0.6353 and
  declared the primary NOT REPORTABLE.

  The 28 draws had mean **0.4827**, sd **0.0616**. The corrected control (200 permutations through
  the identical grouped-CV pipeline) confirms this is ordinary variance, not a leak:

  | layer | observed | null mean ± sd | null p95 | null max | p |
  |---|---|---|---|---|---|
  | 13 | 0.9994 | 0.4913 ± 0.0662 | 0.6008 | 0.6922 | 0.005 |
  | 20 | 0.9897 | 0.4967 ± 0.0654 | 0.6133 | 0.6943 | 0.005 |

  **My 0.60 threshold sat almost exactly at the 95th percentile of a single draw**, so ~1.4 of 28
  layers were expected to exceed it; exactly 1 did. **Verdict: NO LEAK.** The grouping works and
  the pipeline is sound. This corrects a badly-built control; it does **not** renegotiate the
  primary rule, and it cannot rescue the primary, for the reason immediately below.

- **Results:**

  **H-R1 → UNINFORMATIVE, and this branch was pre-declared.** The tier probe is at ceiling in
  **28 of 28 layers** (0.9733–1.0000), which the pre-registration named in advance as the outcome
  where depths cannot be ranked.

  **The diagnostic number is layer 0: AUC 0.9953.** Adversarial renaming changes the literal
  tokens, so L0-vs-L1b is linearly separable at the *embedding* layer, before a single decoder
  block has run. **The tier probe measures surface token identity, not semantic representation.**
  This is a property of the stimulus set, and per the frozen rule it is explicitly **not** to be
  read as "layer 20 is fine".

  **H-R2 → NO READ-DEPTH ADVANTAGE AT 13, and the gradient points the other way.**

  | layer | 0 | 8 | 13 | 16 | 20 | 24 | **27** |
  |---|---|---|---|---|---|---|---|
  | correctness AUC | 0.6205 | 0.6641 | **0.6775** | 0.7098 | **0.7299** | 0.7511 | **0.7612** |
  | tier AUC | 0.9953 | 0.9994 | 0.9994 | 0.9950 | 0.9897 | 0.9831 | 0.9775 |
  | norm-only AUC | 0.5944 | 0.6461 | **0.7922** | 0.7144 | 0.6453 | 0.6189 | 0.6547 |

  **L13 − L20 = −0.0524** [−0.1669, +0.0683] — layer 13 is *worse*, and the CI spans zero, so
  formally this is a null at n = 60 with 32 positives. The structure that is not null is the
  **monotone rise from L12 to the final layer**, argmax **L27 at 0.7612**.

  **H-R3 → the curve.** Tier argmax is L11 at 1.0000 and means nothing at ceiling. Correctness
  argmax is **L27**. One incidental result worth keeping: **the norm-only probe peaks at exactly
  L13 (0.7922)** — whatever distinguishes layer 13 shows up in activation *magnitude* — while the
  full probe beats norm-only there by **+0.2072**, so the tier signal is not magnitude.

- **What worked / hypothesis verdict:**
  - **H-R1 — UNINFORMATIVE (pre-declared branch).** Not "no advantage": the instrument cannot rank
    depths on this contrast at all, because the contrast is solved at the embedding layer.
  - **H-R2 ✗ — no read advantage at layer 13**, direction reversed (L20 > L13 by 0.052), CI
    spanning zero.
  - **H-R3 ✓ reported** — correctness decodability rises with depth and peaks at the last layer.

  **Taken with P0.4, the "wrong depth" explanation is retired.** P0.1's coherence peak at layer 13
  has no consequence for **writing** (P0.4: the oracle nets 0.0000 there) and none for **reading**
  (here: 0.052 *worse* than layer 20). A measured property of the representation turned out to
  predict nothing causal and nothing decodable. That is a clean, pre-registered closure of the last
  apparatus-blaming explanation available to Phase 0.

- **Observations:**
  - **The ceiling result is the more useful of the two.** Any future claim of the form "a probe /
    an SAE feature / a read detects obfuscation" on this stimulus set is vacuous — a linear probe
    on the embedding layer does it at 0.995. A meaningful version needs token-matched controls
    (same identifiers, different structure) or the L2/L3 tiers, where the surface cue is absent.
    **This retires a whole family of experiments that would have looked successful.**
  - **The dense baseline the charter demands now exists, and it is the number to beat.**
    Correctness from the raw residual stream: **AUC 0.7612**. For calibration: reply length alone
    gives 0.693 (2026-08-29) and N13's answer instability gives 0.869. **The best internal read
    still loses to a free behavioural signal** — the programme's signature pattern, now with a
    dense probe rather than an autoencoder.
  - The NLA's layer-20 choice costs little on the readout: 0.7299 against 0.7612 at L27, a gap of
    0.031. So "the instrument is at the wrong depth" is not merely unsupported — even the best
    available depth would buy about three AUC points.
  - Writing the UNINFORMATIVE branch into the pre-registration is what made the ceiling result
    reportable instead of embarrassing. Without it the honest options would have been to bury a
    28/28 ceiling or to invent a post-hoc reason it was fine.

- **New questions / new hypotheses:**
  - **Re-run the tier probe on L2/L3 (flattening, stacked)**, where obfuscation is structural and
    there is no lexical giveaway. If the probe stays at ceiling there too, obfuscation-detection is
    trivial in general; if it drops toward chance, the L1b ceiling is specifically a renaming
    artefact and the tiers are not interchangeable as probe targets.
  - **Is the depth gradient in correctness just "closer to the logits"?** L27 is one block from the
    output, so a correctness probe there may be reading the model's own emerging answer rather than
    a representation of difficulty. Control: probe at L27 on the *prompt's* final token versus a
    mid-generation position, and compare against the answer-line result from 2026-08-29 (`rt_cos`
    at `answer_line`, q = 8e-05) — which was also a signal that only appeared near the answer.
  - The `answer_line` lead and this depth gradient may be the same phenomenon seen twice. That is
    now the most promising thread for the readout paper and it is a **read**-side question, which
    is the side that survives the reproducibility floor intact.

- **Next Steps:** the L2/L3 tier-probe control; then decide whether the readout paper's positive is
  the answer-line/late-layer effect. P0.3-ext still blocked on `adversarial_rename`.
