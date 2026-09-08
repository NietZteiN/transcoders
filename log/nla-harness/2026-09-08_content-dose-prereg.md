### Target Date: 2026-09-08 (H-W39 — is route similarity smooth in content similarity, or does it snap? Frozen before running.)

**Thread:** nla-harness · **Job:** not yet submitted · **Raised by:**
[`2026-09-08_specificity-results.md`](2026-09-08_specificity-results.md) ·
**Uses:** the banked `data/nla/p0/heads/gemma12b/{vectors.npz,spans.jsonl}`; no AV, no AR.

- **Why.** H-W35 found route similarity graded by content — Spearman against `C3pure`'s profile was
  0.847 (same item, other span), 0.743 (other item), 0.283 (no content). But those are **three points
  chosen for other reasons**, so "graded" is an impression, not a measured curve. Interpolating
  between a span's own clean state and a foreign one turns the content difference into a dial and
  asks whether the route degrades **smoothly** or **snaps** at some point.

- **The measure, and why it is necessity.** Both `suf` and `nec` were computed for every arm in
  H-W31/H-W35; comparing them across the five banked arms shows **necessity discriminates about
  twice as hard**:

  | comparison | Spearman on `suf` | Spearman on `nec` |
  |---|---|---|
  | `C3pure` vs `P_patch` | 0.838 | 0.763 |
  | `C3pure` vs `N_sibling` | 0.847 | 0.766 |
  | **`C3pure` vs `N_foreign`** | **0.743** | **0.396** |
  | **`C3pure` vs `N_random`** | **0.283** | **−0.158** |

  Sufficiency compresses the interesting contrast into 0.85 → 0.74; necessity spreads the same
  contrast across 0.77 → 0.40 → −0.16. Choosing `suf` here would repeat H-W36a's mistake — freezing
  a rule on a measure with too little range to fail informatively. **Primary is `nec`; `suf` is
  reported as a secondary.** This choice is made from banked data before any dose arm exists, and is
  recorded here so it cannot be mistaken for a post-hoc pick.

- **Setup.** Three new arms interpolate each span's own clean state toward **the same foreign vector
  `N_foreign` already used**, so the α = 1 endpoint is the banked arm rather than a fresh draw:

      v(α) = normalise( (1 − α)·h0_own + α·h0_foreign ),   α ∈ {0.25, 0.50, 0.75}

  with α = 0 ≡ `P_patch` (+41.52) and α = 1 ≡ `N_foreign` (+12.99) both banked. `PositionReplacer`
  norm-matches on write, so α moves **direction only**. Because a linear blend of two near-orthogonal
  vectors is not uniform in angle, **α is not assumed to be the dose**: the achieved
  `cos(v(α), h0_own)` is measured per span and is the x-axis of every curve. Same 471 spans / 60
  items / 1,459 positions, same identity gates, seed 20260724, N_BOOT 10000.

- **Hypotheses and frozen rules.** Let ρ(α) = Spearman between the **necessity** profile at α and the
  necessity profile at α = 0, over all 255 components. Endpoints: ρ(0) = 1 by construction,
  ρ(1) = **0.316** (banked `P_patch` vs `N_foreign` on `nec`).
  - **H-W39a — shape.** Write `mid = ρ(0.5)` and let `lin = 0.5·(ρ(0) + ρ(1)) = 0.658` be the
    linear-interpolation expectation.
    **`W39-SMOOTH`** if ρ is monotone non-increasing in α **and** `|mid − lin| ≤ 0.15`.
    **`W39-THRESHOLD-LATE`** if `mid ≥ lin + 0.15` — the route survives most of the way and collapses
    near the end, i.e. a component only stops carrying once the content is almost entirely wrong.
    **`W39-THRESHOLD-EARLY`** if `mid ≤ lin − 0.15` — the route degrades as soon as the content is
    perturbed at all. Non-monotone → **`W39-NON-MONOTONE`**, reported as a failure of the design
    rather than a finding about the model.
  - **H-W39b — magnitude, descriptive.** `dG_S(α)` against α and against achieved cosine. There is no
    threshold and no verdict word; the point is whether *effect size* and *route* decay together or
    come apart. Coming apart would be the more interesting outcome: it would mean a write can keep
    using the right heads while delivering less, or use different heads while delivering the same.
  - **H-W39c — the top-8, descriptive.** Jaccard of the top-8-by-necessity at each α against α = 0,
    and specifically whether `L41H4` stays first.

- **Stated priors.** On H-W39a I expect **`W39-THRESHOLD-LATE`**, weakly. The carrying heads are
  selected by attention geometry, and a half-corrupted vector still points substantially at the true
  state, so I expect the route to be robust until the content is mostly wrong. I was wrong about the
  last prior in this family (H-W35 GENERIC → INTERMEDIATE), which is a reason to state this one
  plainly rather than to hedge it. On H-W39b I expect magnitude to fall **faster** than route
  similarity — the effect should degrade before the path does.

- **What this cannot settle.** A blend of two clean states is not the same thing as a *partially
  wrong description*, which is what the NLA edit actually produces; this dial moves through vectors
  that may be off-manifold in a way neither endpoint is. So H-W39 characterises the *measurement*'s
  response to content distance, and licenses no direct claim about edited text.

- **Results / verdict:** not yet run.
