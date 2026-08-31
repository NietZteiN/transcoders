### Target Date: 2026-08-31 (the ladder — a route-specific positive that survives its controls)

The five-tier read probe. Follows the 2026-08-30 tier retraction, whose lesson — measure each tier
against **its own** length baseline — is built into the design here.

- **Hypotheses / what we tested:**

  `last_prompt` ≈ reply length had been shown on L0 and L1b only, and both sit on the **atom-level**
  route. The study's frame has two documented failure routes — atom-level interference (L1 nonsense
  renaming, L1b adversarial renaming) and **relational overload** (L2 control-flow flattening, L3
  stacked) — and the relational half had never been probed. Either the residual stream ties length
  at every tier, making the item-level null ledger fully general, or it beats length on one route
  and not the other.

- **Setup:**
  ```
  run     array 359698 (h200), 5 tiers x 5 draws x 60 items = 1,500 generations, 35-43 min each
          nla/src/p1b_ladder.py · uniform regime: all tiers generated fresh at max_new_gen 2048
          rather than reusing banked L0/L1b draws — mixing regimes produced the 08-30 retraction
  target  graded k/5 · Ridge alpha = 1.0 frozen · GroupKFold(5) on snippet_id
  acts    extracted once per tier (reads are bit-exact, 2026-08-30)
  score   nla/src/p1b_ladder_score.py, then nla/src/p1b_ladder_null.py for the honest nulls
  ```

- **Two scoring errors of mine, found and corrected before the result was believed:**
  1. `beats_length_by` first used the **argmax over 28 layers** — a maximum over 28 correlated
     statistics compared against a single-number baseline, which inflates the gap.
  2. The first permutation **fixed the layer at the observed argmax**, asking "is this layer
     unusual?" when the claim was "is the best of 28 unusual?".

  Both are corrected below: the **mean over layers** is the primary, selection-free statistic, and
  the null is taken over the same statistic the claim uses. **The correction matters —
  the max-statistic null has mean +0.17 to +0.18, i.e. taking the best of 28 layers of pure noise
  reaches ≈ 0.18 on average.**

- **Results:**

  | tier | acc | length baseline | mean ρ | **beats length (mean)** | max ρ | beats length (max) |
  |---|---|---|---|---|---|---|
  | L0 clean | 0.667 | +0.4486 | +0.3892 | **−0.0593** | +0.5323 | +0.0837 |
  | L1 rename | 0.573 | +0.3608 | +0.0224 | **−0.3383** | +0.2003 | −0.1604 |
  | L1b adversarial | 0.580 | +0.2803 | +0.2175 | **−0.0628** | +0.3539 | +0.0736 |
  | **L2 flattening** | 0.617 | +0.2710 | **+0.4105** | **+0.1395** | +0.4899 | +0.2188 |
  | **L3 stacked** | 0.567 | +0.3195 | **+0.4133** | **+0.0938** | +0.5276 | +0.2081 |

  **By route (selection-free): clean −0.059 · atom −0.201 · relational +0.117.**

  **Permutation nulls, both statistics:**

  | tier | mean-stat vs null | p | max-stat vs null | p |
  |---|---|---|---|---|
  | L1b | +0.2175 vs −0.0266 ± 0.1139 | 0.030 | +0.3539 vs +0.1759 ± 0.1316 | **0.114** |
  | **L2** | +0.4105 vs −0.0223 ± 0.1339 | **0.005** | +0.4899 vs +0.1692 ± 0.1386 | **0.005** |
  | **L3** | +0.4133 vs −0.0199 ± 0.1251 | **0.005** | +0.5276 vs +0.1805 ± 0.1364 | **0.025** |

  **The correction earned its keep: L1b's apparent +0.0736 max-statistic advantage is NOT
  significant once selection is accounted for (p = 0.114).** Had the first scoring stood, L1b would
  have been reported as a weak positive. Both relational tiers survive both nulls.

  **The finding:** the residual stream at `last_prompt` carries item-level correctness information
  that reply length does not — **under relational obfuscation, and not under atom-level obfuscation
  or on clean code.** L1's mean ρ of **+0.0224** is the extreme case: at no depth does the residual
  stream predict correctness on nonsense-renamed code.

- **What worked / hypothesis verdict:**
  - **Route-specific positive, surviving selection-free scoring and both nulls.** L2 clears the
    +0.10 bar written into the script before the run (+0.1395); L3 is marginal on that statistic
    (+0.0938) but significant on both nulls.
  - **The item-level null ledger has a scope, and it is the atom route.** Every item-level read in
    this programme was taken on L0 or L1b — the clean anchor and the atom route — which is exactly
    where the residual stream ties or loses to a token count. **B4, B5, N11 and P0.4 were run on
    the one route where there was nothing to read.**
  - **The effect is broad in depth, not a spike.** L2 ranges +0.29 to +0.49 across all 28 layers;
    L3 rises +0.24 → +0.53. L1 sits at +0.02 throughout.

- **Observations:**
  - **This is exploratory, not confirmatory, and the distinction matters this week.** The +0.10
    threshold and the route grouping were written into the script before it ran, but the ladder was
    designed, run and scored in one sitting on data I could inspect. **Two attractive positives
    died to controls in the last 48 hours.** The correct status is: *a route-specific effect that
    survived every control applied to it so far, awaiting a pre-registered replication on
    independent draws.*
  - **The atom/relational split is exactly the frame the study was built on**, which is a reason
    for confidence and a reason for suspicion in equal measure — it is the result we would most
    like to see. A replication with the rule frozen in advance is the only thing that settles it.
  - **L0's negative sign is itself informative:** on clean code, reply length beats the residual
    stream (−0.0593). Verbosity is a better correctness signal than the representation, until the
    code is structurally obfuscated.
  - The mechanism this suggests is testable: relational obfuscation forces hidden-state simulation
    (the dispatcher/state-tracking account from Papers 2–3), so a state being tracked would be
    *in* the residual stream in a way that a lexical decoy is not. That is what E3's attribution
    graphs were commissioned to look for, and this is the first read-side evidence pointing there.

- **New questions / new hypotheses:**
  - **Pre-register and replicate on independent draws.** Five fresh draws for L2/L3 and L1b, rule
    frozen, before this is called a finding. ~1 h of GPU. **This is the next thing to run.**
  - **Split-half the existing draws** as an immediate cheap check on label-noise sensitivity —
    the depth curve died to exactly that on 2026-08-30.
  - If it replicates: probe *what* is decodable at L2/L3 — dispatcher state, loop counter, hop
    index — which turns a correctness correlation into a mechanistic claim and connects to E3.

- **Next Steps:** pre-registered replication of the relational effect; then the split-half check.
  P0.3-ext still blocked on `adversarial_rename`.
