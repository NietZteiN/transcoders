### Target Date: 2026-08-31 (PRE-REGISTRATION — replication of the relational read effect)

**Frozen before a single replication draw exists.** Discovery entry:
[`2026-08-31_ladder-relational-positive.md`](2026-08-31_ladder-relational-positive.md), whose
numbers are quoted below and are **not** renegotiated here.

- **Hypotheses / what we're testing:**

  The ladder found that the residual stream at `last_prompt` beats each tier's own reply-length
  baseline under **relational** obfuscation and not under **atom-level** obfuscation:

  | route | tiers | beats length (mean ρ over 28 layers) |
  |---|---|---|
  | clean | L0 | −0.0593 |
  | atom | L1, L1b | **−0.2006** |
  | relational | L2, L3 | **+0.1167** |

  It survived selection-free scoring and permutation nulls on both statistics (L2 p = 0.005/0.005,
  L3 p = 0.005/0.025). But it was **designed, run and scored in one sitting on inspectable data**,
  and it is precisely the result this study was built to want. Two attractive positives died to
  controls in the 48 hours before it. This replication is what decides whether it is a finding.

- **Setup:**
  ```
  draws   five NEW draws per tier (indices 5-9) for all five tiers, same script, same regime:
          p1b_ladder.py --draws 10 resumes by (draw, snippet_id) and adds only the new ones
  scoring draws 5-9 ONLY. The discovery draws 0-4 are not pooled in — a replication that
          includes its own discovery data is not a replication.
  target  graded k/5 on the new draws · Ridge alpha = 1.0 frozen · GroupKFold(5) on snippet_id
  stat    mean rho over all 28 layers minus that tier's own length baseline, computed from the
          same new draws. Selection-free: no argmax anywhere in the primary.
  ```

- **Decision rules — frozen:**

  **PRIMARY — the route contrast, because that is what the claim is.** Testing each tier against an
  absolute bar would let a single tier's noise decide a claim about routes, and L3's discovery value
  (+0.0938) sits close enough to the +0.10 line that an absolute rule would be near a coin flip.

  Let Δ = mean(beats-length over L2, L3) − mean(beats-length over L1, L1b).

  - **REPLICATED** iff **Δ ≥ +0.15** *and* the relational mean is **> 0**.
    The discovery value is Δ = +0.3178; **+0.15 is half of it**, the standard replication
    convention, and is set here rather than after seeing the number.
  - **NOT REPLICATED** iff Δ < +0.15, or the relational mean is ≤ 0.
    Then the discovery is recorded as a false positive of the same family as the depth gradient and
    the tier effect, and the item-level null ledger stands without a route exception.

  **SECONDARY, reported whatever the primary says:**
  - Per-tier permutation nulls on the mean statistic, 200 draws, for L1b, L2, L3.
    Supportive if L2 and L3 each reach p < 0.05 and L1b does not.
  - **L1 must stay near zero.** Its discovery value was +0.0224 mean ρ — no signal at any depth.
    If L1 now shows a large effect, the pipeline changed and neither run is interpretable.
  - **Split-half over the pooled ten draws** (discovery + replication) as a label-noise check, the
    control that killed the 2026-08-30 depth gradient.

  **Standing constraints:**
  - Full sample only; no partial tier is scored.
  - The selection-free mean statistic is the verdict. Max-over-layers is reported as secondary and
    carries its own null, since the max of 28 layers of pure noise averages **+0.17–0.18**.
  - This cannot revive anything Phase 0 closed. A replicated relational effect would mean the
    item-level nulls have a **scope** (the atom route), not that they were wrong.
  - **Both outcomes are publishable and are written down as such now**, so there is no incentive to
    prefer one when the numbers arrive.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** `nla/scripts/p1b_ladder_sbatch.sh` with `--draws 10`, then
  `nla/src/p1b_ladder_replication.py`.
