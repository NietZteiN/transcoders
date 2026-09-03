### Target Date: 2026-09-03 (Pre-registration: continuous readout, optimised-vector site test, causal patching map)

**Status:** frozen before any run · **Host:** `gemma12b` (permitted) · **Thread:** nla-harness ·
Filed after [`2026-09-03_flippable-denominator.md`](2026-09-03_flippable-denominator.md) showed that
the rescue ceiling on this host (+0.10) equals the day's support threshold. Nothing below has been
run; no number below was seen before the rules were written.

- **Hypotheses / what we're testing:** Three diagnostics, run in order, each cheap, that together
  decide whether there is *anything* rescuable at the current site before any further steering
  recipe (directional ablation, generation-time writes, SAE-feature clamping, host switch) is
  spent. They share one new metric.

  **Metric M — clean-trace log-likelihood ratio.** For item *i* with corrupt (L1b) prompt
  `x_i`, let `y_i^clean` be the model's own greedy L0 reply and `y_i^corr` its own greedy L1b reply
  (both already on disk in `baseline.jsonl`). Under an intervention θ,
  `M_i(θ) = [log P(y_i^clean | x_i, θ) − log P(y_i^corr | x_i, θ)] / |y_i^clean|`, teacher-forced,
  per-token. Positive = the write makes the model *more willing to produce its own clean trace*
  than its corrupt one. Continuous, per item, defined on all 60 items (not 6), needs one forward
  pass per reply, and asks the belief question directly. `ΔM_i(θ) = M_i(θ) − M_i(∅)`.
  Floor for M: `M(∅)` computed twice (bf16 non-determinism) → floor = 2 × SD of the paired
  difference; if that SD is 0 use 0.01 nats/token.

  **Experiment R — re-score existing arms on M.** All 1,920 banked rows' interventions are
  re-applied (same vectors, α, positions, layer) but scored by M instead of generation.
  - **R-1 ✓** iff any belief-shaped arm (V1/V2/V3/V4/V5) has mean ΔM ≥ +0.05 nats/token, paired
    bootstrap CI excluding 0, **and** `R_random` at the same α does not match it. → the site was
    delivering a belief-direction effect that greedy accuracy could not see.
  - **R-0 ✗** iff no belief-shaped arm clears the floor while `R_random` at α ≥ 1 does (sensitivity
    clause) → the site is live for M but no direction we have moves the trace.
  - **R-UNINF** iff nothing, including `R_random@8` at id_spans (which destroyed 27 items), moves M
    → M is not a sensitive readout; abandon M, fall back to accuracy on an enlarged corpus.

  **Experiment S — optimised-vector site test.** Learn `δ ∈ R^3840` by gradient ascent on
  `Σ_train M_i(δ)` at L32/`last_prompt`/prefill-only (the exact banked site), 30 train items,
  30 held-out, split by seed 0, **5 splits**; ‖δ‖ constrained to the V3 norm at α=1 (so it is a
  *direction* search, not a magnitude search). Controls: (a) `δ_shuf` optimised with `y^clean`/
  `y^corr` swapped on train; (b) `δ` evaluated on the 30 held-out **L0** prompts (must not lower
  `log P(y^clean)` there — specificity).
  - **S-LIVE** iff held-out mean ΔM(δ) ≥ +0.05, CI excluding 0, on ≥ 4 of 5 splits, **and**
    `δ_shuf` does not. → the site can carry belief information; our directions were wrong. Report
    cos(δ, V1), cos(δ, V3), cos(δ, V4) — if all < 0.1 the NLA/contrastive directions were
    orthogonal to the working one.
  - **S-DEAD** iff train ΔM rises (optimisation works) but held-out ΔM is inside the floor on ≥ 3
    of 5 splits → no direction at this site generalises; **single-layer late-token steering is
    closed on this host regardless of recipe.**
  - **S-UNINF** iff train ΔM itself does not rise → optimiser failure; report and fix before
    reading anything.
  - Secondary, reported never gating: greedy accuracy and rescue count on the held-out flippable
    items under δ.

  **Experiment T — causal patching map.** Clean-run (L0 prompt) residuals patched into the
  corrupt run (L1b prompt) at cells (layer ℓ × position class *c*), scored by fraction of the
  clean–corrupt M gap recovered: `rec(ℓ,c) = ΔM(patch ℓ,c) / [M(L0 prompt) − M(L1b prompt)]`.
  Layers ℓ ∈ {0,4,8,…,44} ∪ {32} (13 cells). Position classes: **id** (identifier-span tokens),
  **code** (non-identifier code tokens), **instr** (prompt scaffold), **last** (last prompt token),
  **all**. Token alignment L0↔L1b via `difflib` on token ids with identifier spans forced to
  align span-to-span (span *k* → span *k*; unequal span lengths → patch the span's last token and
  mean-broadcast the rest; count and report items where alignment fails, skip them).
  - **T-LOCAL** iff some cell recovers ≥ 0.50 of the gap → there is a site; report argmax (ℓ*, c*).
    If (ℓ*, c*) ≠ (32, last), the banked site was demonstrably not the load-bearing one, and the
    49% coverage ceiling gets its answer (depth if ℓ* < 32, position if c* ≠ last at ℓ = 32).
  - **T-DISTRIBUTED** iff no single cell ≥ 0.50 but `all` at some ℓ ≥ 0.50 → the decoy
    semantics are spread across positions at one depth; single-position steering of any kind is
    bounded, all-position recipes (directional ablation) are the only live option.
  - **T-DEEP** iff even `all` never reaches 0.50 at any single ℓ → interference is spread across
    depth *and* position; only weight-level (orthogonalisation) or embedding-level interventions
    remain, and they are the next family.
  - Sanity: `all` at ℓ=0 must recover ≈ 1.0 (patching the full input is the clean run). If it
    does not, alignment is broken and T is void.

- **Setup (to be executed exactly):**
  - Scripts to write: `nla/src/trace_llr.py` (M), `nla/src/site_optimise.py` (S),
    `nla/src/patch_map.py` (T); sbatch wrappers in `nla/scripts/`; sha256 of each recorded in the
    results entry. Seeds: split seed 0–4 for S; optimiser seed = split seed. bf16, `gemma12b`,
    L32, `TOKENIZERS_PARALLELISM=false`. No `--deterministic`.
  - Order: R first (validates M and its floor; ~40 min, 1 H200), then S (~1 h), then T (~1.5 h).
    **S and T are not run if R returns R-UNINF.**
  - Power gate, applied to any future accuracy-level rescue experiment on this host: refuse to
    launch unless flippable ceiling ≥ 2 × floor (≥ 9 flippable items at n=60).

- **Decision table (frozen):**

  | R | S | T | reading | next |
  |---|---|---|---|---|
  | R-1 | — | — | greedy hid a real effect | re-run day's arms scored by M; the accuracy nulls are re-scoped |
  | R-0 | S-LIVE | any | directions wrong, site fine | build directions from δ; test on Dataset B |
  | R-0 | S-DEAD | T-LOCAL elsewhere | wrong site | move the write to (ℓ*, c*), re-run V3/V1 there |
  | R-0 | S-DEAD | T-DISTRIBUTED | wrong recipe | directional ablation, all positions, all layers ≥ ℓ* |
  | R-0 | S-DEAD | T-DEEP | no activation-level fix | embedding swap / orthogonalisation / host switch to Gemma-2-9B-it for SAE features |
  | R-UNINF | not run | not run | M is not a readout | enlarge corpus (Dataset B), re-run accuracy with the power gate |

- **Results:** *(none — pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** The three are ordered so that each cheaper test can void the more expensive
  one. S is the decisive single experiment for the programme's central ambiguity ("no belief to
  edit" vs "cannot deliver"): an optimised vector that generalises means the second; one that does
  not means the first, at this site. T is the experiment we should have run on day one — it is
  standard causal tracing and it answers where to write before writing.
- **New questions / new hypotheses:** deferred to results.
- **Next Steps:** implement `trace_llr.py` with a CPU smoke test that (a) reproduces M(∅) on 3
  items twice and reports the SD, and (b) recovers M(L0 prompt) > M(L1b prompt) on the 6 flippable
  items — if the clean prompt does not prefer its own clean trace, M is broken and nothing downstream
  runs.
