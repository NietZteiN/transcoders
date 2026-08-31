### Target Date: 2026-08-31 (PRE-REGISTRATION — is the L2 signal more than static dispatcher complexity?)

**Frozen before the probe is run.** Follows
[`2026-08-31_ladder-replication-results.md`](2026-08-31_ladder-replication-results.md), where the
relational read effect replicated (Δ = +0.3195 vs a +0.15 bar; L2 beats its own length baseline by
+0.1253, p = 0.005).

- **Hypotheses / what we're testing:**

  L2 in this corpus is **dispatcher indirection**, not switch-based flattening: 0 of 70 items
  contain a `switch`, and every item carries a lookup object of wrapper functions with **2–12
  dispatcher sites** (`dispatcher_spans`) named by `meta.dispatcher_state_var`. That matches the
  charter's own phrasing — *"dispatcher indirection that forces hidden-state simulation"*.

  The replicated result says the residual stream predicts correctness beyond **reply length** on
  this tier. It does not say what it is predicting *with*. There is an obvious deflationary
  candidate, and it is already a documented behavioural finding: **dispatcher complexity correlates
  with accuracy at r = −0.196 (q = 3.1×10⁻²³) in Papers 2–3.** If the residual stream is merely
  encoding *how many dispatcher sites this program has*, the effect is that known static
  correlation re-measured internally — real, but not evidence of state tracking.

  - **H-L2a (primary).** Does the residual stream beat a baseline that already knows reply length
    **and** the code's static complexity?
  - **H-L2b (control).** The identical test on L1b, where the residual stream does **not** beat
    reply length. It should not beat the combined baseline either; if it does, the combined
    baseline is mis-specified rather than the L2 result being special.

- **Setup:**
  ```
  acts    data/nla/p0/p1b/ladder/{L2,L1b}/acts.npy — already on disk, reads are bit-exact
  target  graded k/10 over all ten draws
  probe   Ridge alpha = 1.0 frozen · GroupKFold(5) on snippet_id · mean rho over 28 layers
          (selection-free — no argmax anywhere, per the 08-31 correction)
  static  n_dispatcher_spans · code_chars · n_lines · has_while · is_javascript
  base    combined = reply length + the five static features, one Ridge, same folds
  ```

  **Language is in the baseline on purpose.** The corpus is 40 Python / 30 JavaScript at L2, and
  N9 found the verbaliser names the wrong language on 37% of reads with a 79.9%/1.0% JS/Python
  split. A probe that separated languages would look like a mechanism.

- **Decision rules — frozen:**

  - **BEYOND STATIC COMPLEXITY** iff mean ρ (residual) − ρ (combined baseline) **≥ +0.10** *and*
    the residual's permutation p < 0.05 on the mean statistic (200 draws).
    Then the residual stream carries item-specific information about this program's traversal that
    its static shape does not supply — the state-tracking reading, and the first mechanistic claim
    the programme would have.
  - **THE SIGNAL IS STATIC COMPLEXITY** otherwise. Then the replicated L2 effect is the
    r = −0.196 dispatcher-complexity finding surfacing in the residual stream. **Still worth
    reporting** — it would be the first internal correlate of a documented behavioural effect in
    this programme — but it is *not* evidence of hidden-state simulation and must not be described
    as such.

  **Standing constraints:** full sample only; the combined baseline is fitted with the same folds
  and regularisation as the residual probe, so neither is advantaged; and this cannot revisit the
  replication, which was about length alone and stands either way.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** `nla/src/p1b_l2_mechanism.py`, CPU.
