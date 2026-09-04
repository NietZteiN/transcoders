### Target Date: 2026-09-03 (R2 — damage-controlled readout, rule frozen before scoring)

Amendment to [`2026-09-03_site-alive-prereg.md`](2026-09-03_site-alive-prereg.md), raised by
[`2026-09-03_site-alive-results.md`](2026-09-03_site-alive-results.md). Neither of those entries is
edited; this one supersedes the metric they used. **Written before any R2 number is computed.**

**Disclosure — this is not a blind pre-registration, and must not be reported as one.** The
per-arm point estimates of the clean half are *already visible*: the post-hoc decomposition table in
the results entry lists mean Δ log P(y_clean | x_l1b) for all 28 arms. What is still blind, and what
this rule is therefore worth: **the CIs, the reproducibility floor on the clean half, the scale G,
the behavioural veto join, and the entire specificity pass** (not on disk — it needs GPU). R2's
confirmatory weight lives in the vetoes and in the pre-stated predictions below, not in the ranking.

- **Hypotheses / what we're testing:**
  - **H-R2:** with the corrupt-trace term removed and damage vetoed, some **belief-shaped** write at
    the banked site moves the model toward its own correct reply. CONFIRM if a belief arm is a mover,
    separates from its matched-α random control with **non-overlapping CIs**, and clears both vetoes.
    REFUTE if movers exist but none is belief-shaped, or if no arm moves at all.
  - This is the last question the banked rows can answer. It does **not** re-open accuracy-level
    rescue, which stays gated by the 6/60 flippable denominator.

- **Setup (frozen definitions).** Per item *i*, per arm *a*, all per-token, all from
  `data/nla/p0/trace_llr/gemma12b/llr_rows.jsonl` (rows `sid|tag|clean`, `logp_sum / n_tok`):
  - `c_i(a)` = log P(y_clean_i | x_l1b_i, θ_a) / |y_clean_i| · `c_i(∅)` = the `noop#1` row.
  - **M2_i(a) = c_i(a) − c_i(∅)** — the clean half alone. No corrupt term appears anywhere in R2.
  - **G_i = c_i(l0prompt) − c_i(∅)** — what swapping the prompt for the un-obfuscated one achieves.
    This is the **unit**: the scale of the thing a rescue would have to do.
  - **F2 = 2 × SD_i[ c_i(noop#1) − c_i(noop#2) ]**, fallback 0.01 if degenerate. Computed before any
    arm is scored; noop rows carry no belief information, so estimating F2 from them is not peeking.
  - CIs: percentile bootstrap, 10,000 resamples over items, seed 20260724.
  - `belief` = the condition set already defined in `nla/src/trace_llr.py` (unchanged, so R and R2
    partition the arms identically).
  - **Sanity S2:** G_i > 0 on ≥ 90 % of items **and** mean G > 10 × F2. If either fails the unit is
    undefined → **R2-VOID**, and nothing else is read.
  - **Mover:** mean M2(a) ≥ max(0.10 × mean G, F2) with bootstrap CI95 excluding 0.
  - **Random separation:** belief arm *b* separates iff its CI95 does not overlap that of the random
    arm at the *same* α / positions / ML. This replaces R's floor-based test, which passed a 0.30-nat
    gap between two CIs overlapping across most of their range.
  - **Veto V-behav (no GPU, 25 of 28 arms):** from the banked greedy rows in
    `p0/{nla_steer,coverage,steerv2}/gemma12b/steer_results.jsonl` — arm must not drop L1b parse rate
    or accuracy by more than **0.05** absolute vs the unsteered baseline. An arm with no banked rows
    is marked `veto-unavailable` and **cannot be reported as supported**.
  - **Veto V-spec (GPU, blind, staged):** Δ log P(y_clean | x_l0, θ_a) / |y_clean| ≥ **−0.10 × mean G**
    — the write may not cost more on the clean prompt than the support threshold asks it to win on the
    corrupt one. These rows do not exist and require a forward pass per arm × item.
  - **Staging (deliberate):** V-spec runs **only for arms that clear Mover + separation + V-behav**.
    If no arm clears, the verdict is reached with **no GPU at all**. Smoke first (3 items, 2 arms):
    the `l0prompt` reference must reproduce its banked `c_i(l0prompt)` to < 0.01 nats/token, else the
    pass is misconfigured and is not read.

- **Decision table (frozen):**

  | verdict | condition | reading |
  |---|---|---|
  | **R2-BELIEF** | ≥ 1 belief arm: mover **∧** separates from matched random **∧** V-behav **∧** V-spec | the site carries a belief-shaped write; the banked nulls were about magnitude and denominator, not the site |
  | **R2-GENERIC** | movers exist, but none is a belief arm separating from matched random | movement at this site is generic perturbation — the readout hypothesis stands |
  | **R2-NULL** | no arm clears the mover threshold | nothing at this site moves the clean trace at a meaningful fraction of a prompt swap |
  | **R2-VOID** | sanity S2 fails | the unit is undefined; report the sanity failure and stop |

- **Predictions, stated now.** From the visible point estimates I expect **R2-NULL or R2-GENERIC**,
  and I expect **R2-NULL** to be the more likely of the two: the leaders on the clean half are
  `P_prompt` (+0.069) and `R_random@0.149|id_spans` (+0.056), **neither belief-shaped**, and if mean G
  is anywhere near the +0.905 nats/token scale of the R sanity gap then the 0.10 × G threshold lands
  around +0.09 and **no arm clears it**. The best belief arm at matched energy is `V1_gloss@0.149`
  (+0.015), roughly a quarter of the random control at the same α. **R2-BELIEF would surprise me**;
  I am recording that in advance so that if it happens it counts for something.
  Secondary prediction: `P_prompt` clears the mover threshold iff mean G < 0.69.

- **What this cannot settle.** R2 is a likelihood readout on 60 items from one host, one layer, one
  tier. A positive R2 would license one further accuracy experiment, not a claim — and only if the
  flippable denominator is grown first. A negative R2 bounds the *site*, not the hypothesis that some
  other site or depth carries the belief.

- **Next Steps:** implement `nla/src/r2_score.py` (CPU only, reads the banked rows, writes
  `r2_stats.json` with F2, G, the per-arm table, both vetoes and the verdict); run it; log the result
  in a new dated entry. Only then decide whether the V-spec GPU pass is needed.
