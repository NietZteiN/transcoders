### Target Date: 2026-09-08 (H-W39 — `W39-SMOOTH`: the route degrades smoothly with content, it degrades *before* the effect does, and cosine is nearly useless as a content measure here)

**Thread:** nla-harness · **Job:** 383270 (g-07-13, sweep 1.74 h, 93,660 forwards at 67 ms) ·
**Resolves:** [`2026-09-08_content-dose-prereg.md`](2026-09-08_content-dose-prereg.md) a/b/c ·
**Artifacts:** `data/nla/p0/w39/gemma12b/heads_rows.jsonl` (180 rows), scored against the banked
`P_patch` (α = 0) and `N_foreign` (α = 1) rows of jobs 382366 / 383160.

- **Setup.** Each span's own clean state interpolated toward **the same foreign vector `N_foreign`
  uses**, α ∈ {0.25, 0.50, 0.75}; the endpoints are banked, so only the middles were run. Primary
  measure **necessity**, chosen from banked data before these arms existed. Identity gate exact
  again: `SELF` 0.0000, `ALL` 0.0000, `ko_gap` 0.0000.

- **Results.**

  | arm | α | achieved cos to own | dG_S | ρ(nec) vs α=0 | ρ(suf) | Jaccard top-8 | top by nec |
  |---|---|---|---|---|---|---|---|
  | `P_patch` | 0.00 | 1.0000 | +41.52 | 1.000 | 1.000 | 1.00 | `L47H4` |
  | `D_25` | 0.25 | **0.9977** | +39.21 | 0.800 | 0.902 | 0.78 | `L47H4` |
  | `D_50` | 0.50 | **0.9906** | +31.39 | **0.759** | 0.877 | 0.60 | `L41H4` |
  | `D_75` | 0.75 | **0.9789** | +16.29 | 0.485 | 0.731 | 0.23 | `L41H4` |
  | `N_foreign` | 1.00 | — | +12.99 | 0.316 | 0.696 | 0.14 | `L47H15` |

- **Hypothesis verdicts.**
  - **H-W39a → `W39-SMOOTH`.** ρ(nec) is monotone, and `|ρ(0.5) − linear| = |0.759 − 0.658| =
    **0.101**`, inside the 0.15 the rule allowed. **My prior was `W39-THRESHOLD-LATE` and it does not
    fire** — though the deviation is in that direction, so the prior was right about the sign and
    wrong about the size. That is the third prior I have missed in this family; the rules keep
    catching it, which is the point of freezing them.
  - **H-W39b — descriptive, and my prediction is wrong in an interesting way.** I predicted magnitude
    would fall *faster* than route. As fractions of α = 0:

    | | α=0.25 | α=0.50 | α=0.75 | α=1 |
    |---|---|---|---|---|
    | effect `dG_S` | **0.94** | 0.76 | **0.39** | 0.31 |
    | route ρ(nec) | **0.80** | 0.76 | **0.49** | 0.32 |

    **They cross at α = 0.5.** At small perturbations the *route* moves first while the effect is
    nearly intact — 6 % of the effect lost against 20 % of the route. At large perturbations it
    inverts: the effect collapses to 0.39 while half the route survives. So "how much it delivers"
    and "which heads it uses" are not two views of one quantity; they come apart in both directions.
  - **H-W39c — descriptive.** Jaccard of the top-8 by necessity falls 1.00 → 0.78 → 0.60 → 0.23 →
    0.14, and the leading component changes hands: `L47H4` at α ≤ 0.25, **`L41H4` at 0.5 and 0.75**,
    `L47H15` at α = 1. The head that leads H-W31's sufficiency ranking (`L41H4`) is not the one that
    leads necessity at α = 0 (`L47H4`) — a reminder that the two measures rank differently even
    where they correlate at 0.70.

- **The observation I did not expect, and it is the most useful line here.** Look at the achieved
  cosine. Moving **75 % of the way** to a *different program's* activation moves the cosine to the
  original from 1.0000 only to **0.9789**. Three quarters of the way to entirely wrong content is
  two hundredths of a cosine. The pre-registration insisted the achieved cosine be measured rather
  than assuming α was the dose, and this is why that mattered: **cosine is nearly useless as a
  measure of content in this representation space.**

  It also explains H-W36a rather than merely agreeing with it. There, `cos(h0, c3)` had sd
  **0.0017** across items and predicted nothing about the causal shortfall, which I reported as "a
  fired rule on a measure with no range". This says *why* the range is absent: the residual stream at
  layer 32 is anisotropic enough that even a 75 %-wrong vector sits at cos 0.98. Any measure of
  "how close is this reconstruction" built on raw cosine is reading a scale where the entire
  interesting interval is 0.98–1.00. Two entries, one cause.

- **Observations.**
  1. **The gradient H-W35 saw in three points is a genuine curve**, and it is smooth rather than
     stepped. There is no threshold at which a component stops carrying; contribution degrades
     continuously as the written content moves away from the truth.
  2. **Necessity was the right primary and the choice is visible in the numbers.** Across the same
     five arms ρ(suf) moves 1.00 → 0.70 while ρ(nec) moves 1.00 → 0.32. Had this been frozen on
     sufficiency, the whole curve would have been squeezed into a 0.30 band and `W39-SMOOTH` would
     have been near-unfalsifiable.
  3. **A caution for the write-up.** `D_75` delivers +16.29 nats — still above the +12.11 support
     threshold and comparable to the random-vector benchmark of +19.60 — while being 75 % wrong
     content. Magnitude alone does not certify that a write carried the right thing; that is the same
     lesson as W1's +19.47 landing on a random vector, arriving from the other direction.

- **New questions.**
  - **H-W41:** a content metric with range. Cosine is flat here; centred cosine (which rescued the
    stage-0 gate, +0.68 vs +0.05 where raw cosine gave +0.99 vs +0.97) is the obvious candidate, and
    recomputing the dose x-axis in centred cosine costs nothing.
  - **H-W42:** the route moves before the effect does at low dose. If that holds, a *small* edit
     should be detectable in the necessity profile while leaving `G_sum` almost unchanged — which
     would give the NLA-edit question a far more sensitive readout than accuracy or even `G_sum`.
     This is the first thing in the family that suggests a better instrument for the original
     question rather than another characterisation of the channel.

- **Bounds.** Rank correlations over item-mean profiles; no CI on the Spearman values, so the
  ordering is measured and the *differences* between adjacent α are not tested. The interpolation
  passes through vectors that may be off-manifold in ways neither endpoint is, so this characterises
  the measurement's response to content distance and licenses no direct claim about edited text.
  **The joint stage of job 383270 crashed** — `TypeError: 'Component' object is not subscriptable`.
  **Defect #13, mine:** the H-W35 patch introduced a span list named `pool` inside `stage_joint`,
  which already bound `pool` to the list of **components** for the random-k draws; the later
  assignment won and `item_targets` was handed Components to subscript. Renamed to `span_pool`
  throughout, and the per-arm selection profile — which still read the module constant `ARMS` — now
  follows `--arms` as well. Unlike defect #12 this one **crashed loudly rather than producing an
  empty result**, which is the behaviour to want. H-W39's verdicts depend only on the sweep and are
  unaffected; the per-arm joint localisation for the dose arms is owed and is ~7 minutes of GPU.
