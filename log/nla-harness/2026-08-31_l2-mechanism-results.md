### Target Date: 2026-08-31 (the L2 signal is static dispatcher complexity, not state tracking)

Pre-registration: [`2026-08-31_l2-mechanism-prereg.md`](2026-08-31_l2-mechanism-prereg.md), frozen
before the probe ran and naming this outcome in advance. Tests the *interpretation* of
[`2026-08-31_ladder-replication-results.md`](2026-08-31_ladder-replication-results.md); the
replication itself was about reply length and **stands unchanged**.

- **Setup:** `nla/src/p1b_l2_mechanism.py`, job 360373, CPU, 2 min. Graded k/10 over all ten draws,
  Ridge alpha = 1.0 frozen, GroupKFold(5) on `snippet_id`, mean ρ over 28 layers (selection-free).
  Static features: `n_dispatcher_spans`, `code_chars`, `n_lines`, `has_while`, `is_javascript`.

  **A label correction first.** L2 in this corpus is **dispatcher indirection**, not
  switch-based control-flow flattening: **0 of 70 items contain a `switch`**, and every item carries
  a lookup object of wrapper functions with 2–12 dispatcher sites (mean 7.3). Earlier entries in
  this thread called it flattening; the charter's own phrase — *"dispatcher indirection that forces
  hidden-state simulation"* — is the accurate one.

- **Results:**

  | tier | spans | ρ length | ρ static | **ρ combined** | ρ residual | **beats combined** | perm p |
  |---|---|---|---|---|---|---|---|
  | **L2** | 7.3 | +0.2982 | **+0.2989** | **+0.3846** | +0.4091 | **+0.0245** | 0.005 |
  | L1b | 0.0 | +0.2833 | +0.1198 | +0.2920 | +0.2537 | **−0.0383** | 0.005 |

  **VERDICT: THE SIGNAL IS STATIC COMPLEXITY.** The residual stream adds **+0.0245** over a
  baseline that already knows reply length and the code's static shape — against a frozen bar of
  +0.10. The permutation p = 0.005 says the residual signal is real *versus chance*; it is not
  incremental *versus the code's own visible structure*.

  **The route contrast now has a deflationary explanation, and it is a complete one.** Static
  features alone predict correctness at **ρ = +0.2989 on L2** and only **+0.1198 on L1b** — because
  L1b has no dispatcher spans at all (mean 0.0). So the replicated "relational beats length" result
  is this: **L2 carries an extra *static* predictor that reply length does not fully capture, and
  the residual stream encodes it.** No hidden state required.

  **The control behaved.** L1b does not beat its combined baseline (−0.0383), so the baseline is
  not simply strong enough to absorb everything.

- **What worked / hypothesis verdict:**
  - **H-L2a ✗ — not beyond static complexity.** The state-tracking reading is **refused**. The
    replicated L2 effect must not be described as evidence of hidden-state simulation.
  - **H-L2b ✓ — the control held.**
  - **What the result IS, and it is not nothing:** static dispatcher complexity predicts correctness
    at ρ = +0.2989, and the residual stream encodes that complexity. **This is the first internal
    correlate in this programme of a documented behavioural finding** — the dispatcher-complexity
    effect, r = −0.196 (q = 3.1×10⁻²³) in Papers 2–3. The model represents *how tangled this
    program is*. It does not represent *how the tangle resolves*.

- **Observations:**
  - **The replication stands; its interpretation is now bounded.** "Under dispatcher indirection the
    residual stream beats reply length" remains true and pre-registered. What is withdrawn is the
    inference that this indicates state tracking, which was never tested until now — and which I
    had flagged as the reason to run this before making a mechanistic claim.
  - **Fourth control-driven deflation of the week, and the first one anticipated in advance.** The
    depth gradient, the tier effect and L1b's argmax advantage were all discovered to be artefacts
    after being reported. This one was named as a possible outcome *before* the probe ran, so
    nothing had to be retracted — the difference between a pre-registration and a post-mortem.
  - **The programme's shape is now consistent at every level.** Theme-level content is represented
    (malice, complexity, language); item-level content is not (which capability, which algorithm,
    whether this run was misled, how this dispatcher resolves). The ladder result briefly looked
    like an exception and is not one — it is theme-level content (structural complexity) showing up
    on the one tier where a theme-level property happens to predict item-level outcomes.
  - **This is a better result for the paper than the mechanistic claim would have been**, because it
    is measured rather than asserted, and because it ties an internal representation to a
    behavioural effect already published in Papers 2–3.

- **New questions / new hypotheses:**
  - **Probe the static complexity directly.** If the residual stream encodes `n_dispatcher_spans`,
    that is measurable head-on (regression on the span count, not on correctness) and would make
    the claim direct rather than inferred. Cheap, arrays already on disk.
  - **Does anything survive on L2 after residualising out static complexity?** +0.0245 is small but
    permutation-significant against chance; the honest test is whether the *residualised* target
    (correctness with the combined baseline's prediction removed) is decodable at all.
  - The state-tracking question is not answered, only unsupported by *this* probe at
    `last_prompt`. A dispatcher's state is a thing that changes *during* execution simulation, so
    the reply positions — where the position × depth grid found the mid-reasoning region empty on
    L1b — are where it would live on L2. That tier has never been probed across positions.

- **Next Steps:** direct probe of `n_dispatcher_spans`; residualised-target test; position sweep
  on L2. P0.3-ext still blocked on `adversarial_rename`.
