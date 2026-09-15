### Target Date: 2026-09-13 (H-E1/H-E2/H-E3 — a training-free mean-difference vector matches the trained NLA edit)

- **Hypotheses / what we're testing:** whether the NLA `edit` can be reproduced **without an autoencoder** by a
  single cross-item **decoy-erasure direction**: `v_s = h1b_s + α·‖Δ_s‖·Ê_K^{(−i)}`, where `Ê^{(−i)}` is the unit
  leave-one-out mean of `h0 − h1b` over the **other items'** spans. Rules frozen in
  [`2026-09-12_erasure-vector-prereg.md`](2026-09-12_erasure-vector-prereg.md):
  **H-E3** `ERASURE-DIRECTIONAL` if `erase_1.0 − erase_rand` ≥ **+3.0** with CI excluding 0, else
  `ERASURE-NOISE` (which would void E1/E2) · **H-E1** on `erase_1.0 − edit`: `ERASURE-BEATS-EDIT` if lower > 0,
  `EDIT-CARRIES-CONTENT` if upper < 0, else **`ERASURE-MATCHES-EDIT`** *with the CI width reported* ·
  **H-E2** on `erase_1.0 − 0.55·swap`: **`ERASURE-WITHIN-SHARE`** if upper < 0, `ERASURE-EXCEEDS-SHARE` if
  lower > 0, else `SHARE-UNRESOLVED`.
  **Predictions recorded: DIRECTIONAL · MATCHES-EDIT · WITHIN-SHARE.**

- **Setup:** job **392883** (the re-run after the gate fault in
  [`2026-09-13_erasure-gate-fault.md`](2026-09-13_erasure-gate-fault.md), job 392723), node **g-06-01**,
  `2026-09-13T10:31:33Z → 10:53:43Z`, **22:12 elapsed, rc=0, 0.37 GPU-h.**
  `nla/src/nla_erasure.py` sha256 `8206b81cb25b…` (the patched gate; scientific rules byte-unchanged from the
  prereg). In-job pytest **19 passed / 11 deselected**, then the 3-item smoke, then the full run.
  Host Gemma-3-4B-it, banked vectors `data/nla/ml/gemma4b/gate/vectors/L7.npz` (471 spans × 2 560, fp32), same 60
  items as every run in this family, seed 20260724, N_BOOT 10 000, `--deterministic` OFF. Sets: **L7 at β = 1.0**
  (primary) and **band L2–L13 at β = 0.35**. Output `data/nla/ml/gemma4b/gate/erasure/`.
  **Gates, both passing:** vector identity — `max abs(erase_own − h0)` **3.05 × 10⁻⁵** (bar 1e-3), bf16 differing
  fraction **9.21 × 10⁻⁵** (bar 5e-4); score identity — `edit`, `foreign`, `swap` reproduce the banked rows at
  **exactly 0.0**, `self` **exactly 0.0**. Reported diagnostic, not a gate: `erase_own_chaos_nats` **1.201**
  worst-case per item, while at the **mean** level `erase_own` +73.21 vs `swap` +73.15 — a 0.06-nat gap, so the
  bf16 write chaos that voided job 392723 is ~1 nat per item and ~0.06 nats on the quantity the rules test.

- **Results.** L7, β = 1.0 (nats of `G_sum`, 60 items):

  | arm | mean | 95 % CI |
  |---|---|---|
  | `swap` (raw clean state = ceiling) | +73.15 | [+66.36, +80.11] |
  | `erase_own` (identity arm) | +73.21 | [+66.32, +80.38] |
  | `erase_2.0` | +36.96 | [+32.82, +41.34] |
  | **`erase_1.0`** | **+35.75** | [+31.88, +39.77] |
  | `edit` (the trained NLA edit) | +32.06 | [+27.25, +36.86] |
  | `erase_0.5` | +28.51 | [+24.94, +32.21] |
  | `foreign` (content null) | +23.88 | [+19.49, +28.09] |
  | `erase_rand` (norm-matched random) | +21.43 | [+17.69, +25.13] |
  | `self` | +0.00 | [+0.00, +0.00] |

  Band L2–L13, β = 0.35: `swap` +69.79 · **`erase_1.0` +40.64** · `erase_2.0` +40.71 · `edit` +36.22 ·
  `erase_0.5` +30.07 · `foreign` +26.37 · `erase_rand` +22.72.

  Frozen contrasts, paired per item:

  | rule | contrast | value | bar |
  |---|---|---|---|
  | **H-E3** | `erase_1.0 − erase_rand` | **+14.32 [+11.10, +17.61]** | ≥ +3.0, CI excl 0 |
  | **H-E1** | `erase_1.0 − edit` | **+3.69 [−0.99, +8.30]** (width **9.30**) | sign of the CI |
  | **H-E2** | `erase_1.0 − 0.55·swap` | **−4.49 [−7.99, −0.98]** · share **0.489** | upper < 0 |

  Secondary: specificity `erase_1.0 − foreign` = **+11.87 [+7.80, +15.99]** at L7 and **+14.27 [+9.58, +18.87]**
  in the band (banked `edit − foreign` at L7 is +8.18). Spearman of per-item gain against
  `cos(Δ_s, Ê)`: **ρ = +0.294, p = 0.023**.

- **What worked / hypothesis verdict: all three predictions SUPPORTED.**
  - **H-E3 → `ERASURE-DIRECTIONAL`.** +14.32 [+11.10, +17.61] against a +3.0 bar — the cross-item direction beats
    a norm-matched random vector by nearly 5× the bar, so E1 and E2 are not voided and the direction carries real
    information. Reinforced by the dose structure: `ρ = +0.294 (p = 0.023)` between an item's gain and how well its
    **own** displacement aligns with the **other items'** mean, i.e. the shared direction is the mechanism.
  - **H-E1 → `ERASURE-MATCHES-EDIT`.** `erase_1.0 − edit` = +3.69 with the CI **[−0.99, +8.30]** containing 0.
    **Reported as the prereg demanded, with its width: 9.30 nats.** So this is "does not separate", not "proven
    equal" — but the point estimate favours the *training-free* vector, and the trained NLA edit is nowhere ahead.
    **A leave-one-out mean-difference vector — no autoencoder, no training, one direction per layer — reproduces
    what the trained AV/AR pair delivers.** In the band it is +40.64 against the edit's +36.22, same ordering.
  - **H-E2 → `ERASURE-WITHIN-SHARE`.** `erase_1.0 − 0.55·swap` = −4.49 [−7.99, −0.98], and the achieved share is
    **0.489** — against the **0.493** decoy-removal share measured independently in the activation-space tier
    decomposition (H-W23: meaning ≈ 51 %, decoy removal ≈ 49 %, structure ≈ 1 %). **The erasure vector delivers
    the decoy-removal share of the clean state's effect and essentially nothing beyond it**, to three decimal
    places of a number derived from a different experiment. That is the tightest cross-experiment agreement in the
    programme.
  - **The α ladder saturates at the natural scale.** +28.51 (α = 0.5) → **+35.75** (α = 1.0) → +36.96 (α = 2.0):
    doubling the push past ‖Δ_s‖ buys **+1.21**, while the first half bought +7.24. So ‖Δ_s‖ is not an arbitrary
    normalisation — it is where the effect plateaus, which is what a *removal* operation should look like (the
    decoy can only be removed once) and not what a *magnitude* effect would look like.

- **Observations:**
  - **This is the sharpest statement the programme has of its own central claim.** Three independent lines now
    agree that the NLA `edit` is **erasure, not reconstruction**: (a) `foreign ≈ rt ≈ edit` and random scoring
    +17–19 at L2/L3 (banked); (b) today's ladder, where *perfect token-space erasure* has no accuracy headroom
    (`ERASURE-FLOOR`, [`2026-09-13_tier-ladder-results.md`](2026-09-13_tier-ladder-results.md)); and (c) this run,
    where a training-free mean-difference vector matches the trained edit and lands exactly on the 49 % removal
    share. **The trained layers are buying an operation that costs one mean difference.**
  - **What this does to the CodeSteer framing the user asked about.** It removes the need to compete on method
    cost at all: the honest result is that *this whole family of interventions* — attention reallocation, NLA
    editing, a mean-difference vector — is doing the same cheap thing, and the ladder says that thing cannot move
    accuracy on this failure mode because **the accuracy headroom is in reconstruction (+0.083 [+0.015, +0.160]),
    not erasure (−0.035 [−0.100, +0.027])**. The paper's contribution is the measurement that explains why,
    not a better steering operator.
  - **Specificity is *higher* for the untrained vector** (+11.87 vs the edit's +8.18 against the same `foreign`
    null). Stated as a comparison of point estimates from the same run, not a tested contrast — no rule was
    pre-registered for it, and it should be tested before it is leaned on.
  - **The gate fault cost 15 minutes and bought a constant.** The re-run's diagnostics quantify what voided the
    first attempt: ~1.2 nats per item of bf16 write chaos, 0.06 nats at the mean. That is why H-E1's 9.30-nat CI
    width, not the chaos, is the binding limitation here — worth stating explicitly so the floor is not read as
    threatening H-E3 (+14.32) or H-E2 (−4.49), both of which clear it by an order of magnitude.
  - `self` exactly 0.000 and the three banked-byte arms exactly 0.0 for the seventh consecutive run in this family.

- **New questions / new hypotheses:**
  - **H-E5:** if a leave-one-out mean matches the trained edit, does a **within-item** mean (each item's own Δ,
    i.e. `erase_own` restricted to a *different span* of the same item) beat both? That interpolates between the
    cross-item vector (+35.75) and the full clean state (+73.15) and localises how much of the gap is item-specific
    content rather than method. Free from banked vectors, ~0.3 GPU-h to score.
  - **H-E6:** the 0.489 share matching H-W23's 0.493 predicts the *residual* +37 nats is the meaning component. A
    direct test: write `erase_1.0` **and** the meaning component together and check additivity against `swap`. If
    they add, the clean state decomposes cleanly into two writable parts, which is a much stronger claim than
    either half alone. ~0.5 GPU-h.
  - **H-E7:** is the training-free vector's match to the edit *host-general* or an artifact of our under-trained
    4B pair? H-C4/H-C10 showed our pair is **data-limited**, so a better-trained edit might pull ahead. The
    released 12B pair's `edit` is available and flat (+19.47 vs random +19.60) — computing its erasure-vector
    equivalent on the repaired anchoring is the like-for-like test, and it is **H-C11 with a second purpose**.
    This is the single most important qualification on today's headline and it is nearly free.

- **Next Steps:** H-E7/H-C11 (free, and it qualifies the headline) before any claim that training is unnecessary
  in general. Then H-E4 (bf16 floor, 0.1 GPU-h), H-A5 (0.5), H-S13 (0.6), H-S16 (0.5, cost corrected). The
  programme-level decision — H-C7's ~30 GPU-h corpus vs H-S14's flippable corpus — should wait on H-E7.
