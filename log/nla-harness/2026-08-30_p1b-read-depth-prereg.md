### Target Date: 2026-08-30 (PRE-REGISTRATION — depth for a *read*: the per-layer dense probe)

**Frozen before any activation is extracted.** Follows
[`2026-08-28_p04-depth-results.md`](2026-08-28_p04-depth-results.md) (depth for a *write*:
NOT DEPTH-LIMITED) and [`2026-08-30_reads-exact-truncation-ruled-out.md`](2026-08-30_reads-exact-truncation-ruled-out.md)
(reads are bit-exact, so this is a deterministic measurement carrying no reproducibility caveat).

- **Hypotheses / what we're testing:**

  P0.1 found the task direction `h_L0 − h_L1b` is most coherent at **layer 13 (0.543)** and least
  at the instrument's **layer 20 (0.459)**, with magnitude running the other way. P0.4 asked
  whether 13 is a better place to **write** and the answer was no — the oracle nets exactly 0.0000
  there. **This asks the other half: is 13 a better place to READ?** Those are different questions
  and only the second is the readout paper's.

  It is also the dense baseline this project's charter requires (`CLAUDE.md` §4: every feature or
  steering claim ships with a dense-probe baseline). No SAE, no NLA, no dictionary — a linear
  decoder on the raw residual stream, which is the thing any interpretability claim has to beat.

  - **H-R1 (primary).** Does a linear probe decode **L0 vs L1b** better at layer 13 than at 20?
  - **H-R2 (secondary).** Does it decode **correctness** on L1b items better at 13 than at 20?
    This is the one the readout paper actually wants and it is underpowered at n = 60, so it is
    secondary by construction, not by hindsight.
  - **H-R3 (reported either way).** The full 28-layer curve, which is the read-side analogue of
    P0.1's coherence curve.

- **Setup:**
  ```
  acts    nla/src/layer_rotation.all_layer_acts — ONE forward pass per prompt with
          output_hidden_states=True, read at the final prompt token. layer K == hidden_states[K+1],
          the correspondence P0.1's gate verified at 0.0 relative error.
  corpus  the same 60 L0/L1b pairs as B4 / P0.1 / P0.2 / P0.4 => 120 vectors x 28 layers
  model   Qwen/Qwen2.5-7B-Instruct, bf16 · seed 20260724 · d = 3584
  probe   logistic regression, L2, C = 1.0, on standardized features
  cv      GroupKFold(5) grouped by snippet_id
  ```

  **The grouping is the load-bearing methodological choice.** Each snippet contributes both an L0
  and an L1b vector. Ordinary k-fold would put a snippet's two tiers in train and test, letting the
  probe memorise the snippet instead of the tier and inflating AUC toward 1.0. Grouping by
  `snippet_id` makes every fold's test snippets unseen. Any AUC reported here without grouping
  would be meaningless, and this is written down now so the grouped number cannot later be
  swapped for a prettier ungrouped one.

  **C = 1.0 is frozen, not tuned.** With d = 3584 and n = 120 the problem is p >> n, so the result
  depends on the regularisation strength; tuning C against the same CV that reports the AUC would
  be a leak, and tuning it per layer would be a forking path across 28 layers.

  **Two controls, both mandatory:**
  1. **Label permutation** per layer — the same grouped CV on shuffled labels. Must sit near 0.5.
     Anything above 0.60 means the CV leaks and no curve is reportable.
  2. **Norm-only probe** per layer — logistic on `‖h‖` alone. P0.1 found relative magnitude peaks
     at exactly layer 20, so a probe that merely tracks activation size would manufacture a
     depth effect out of scale. The semantic claim requires the full probe to beat it.

- **Decision rules — frozen:**

  Statistic: **grouped-CV AUC**, with a percentile bootstrap CI over snippets (the resampling unit
  is the snippet, not the vector, because the two tiers of a snippet are not independent).

  **H-R1 primary.**
  - **L13 IS A BETTER READ SITE** iff AUC(13) − AUC(20) **≥ +0.05** with a bootstrap CI excluding
    zero, **and** the full probe beats the norm-only probe at layer 13 by ≥ +0.05.
  - **NO READ-DEPTH ADVANTAGE** iff the contrast is under +0.05 or its CI spans zero. Combined
    with P0.4 this would say layer 20 is not a bad place to read *or* write, and the P0.1 coherence
    peak has no decodable consequence — which retires the "wrong depth" explanation entirely.
  - **UNINFORMATIVE** iff AUC ≥ 0.95 at *every* layer. A contrast that is trivially decodable
    everywhere cannot rank depths, and this outcome is to be reported as a property of the
    stimulus set, **not** as "layer 20 is fine".

  **H-R2 secondary.** The identical rule on correctness. Underpowered at n = 60 and 27 positives;
  a null here is not evidence of absence and is to be reported with that caveat attached.

  **H-R3.** The 28-layer curve and its argmax are reported whatever the primary says. If the
  argmax is neither 13 nor 20, that is the finding.

  **Standing constraints:** full sample only, no partial curve; the permutation control is
  reported alongside every AUC, not in an appendix; and this cannot revisit P0.4 — a read
  advantage at 13 would *not* mean the write result was wrong, it would mean read and write
  depths dissociate, which is itself the interesting outcome.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** `nla/src/p1b_read_probe.py`, one GPU, ~10 min.
