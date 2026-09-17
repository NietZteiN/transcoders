### Target Date: 2026-09-17 (H-R15 — the replication floor, and whether the one positive effect survives it)

- **Hypotheses / what we're testing:**

  H-R14 closed with a caveat that undercuts every interval this thread has published, including its own:
  between two runs, **on the same 50 snippets**, arms moved by ±0.02–0.04 — `ridge_map` +0.0461 → +0.0061,
  `codesteer` −0.0169 → +0.0177, `codesteer_auto` −0.0023 → +0.0154. The cluster bootstrap resamples
  **snippets**, not generations, so it cannot see this at all. This entry measures that floor directly.

  **The framing is corrected before the run, not after.** I previously called this "seed noise". Two
  facts say that name is wrong. (1) Every run in this family already used the *same* hard-coded
  `torch.manual_seed(20260724)`, so the H-R7→H-R14 drift was **not** a seed change — it mixes RNG-stream
  position (148 snippets reach a given snippet at a different RNG state than 50 do), hardware/kernel
  nondeterminism, and, for `ridge_map` only, a genuine change of vectors (single-split → cross-fit).
  (2) This thread has already measured the underlying phenomenon on a comparable pipeline:
  [`2026-08-29_determinism-floor-structural.md`](2026-08-29_determinism-floor-structural.md) found
  **per-item agreement 0.8333 on the same card with deterministic kernels pinned**. So the quantity of
  interest is the **run-to-run replication floor**, of which the seed is only one component and possibly
  a negligible one.

  - **H-R15a (the floor vs our error bars) — primary.** For each measured arm let `s = |Δ_repB − Δ_repA|`,
    where Δ is that arm's case-weighted contrast against `unsteered` **computed within its own replicate**
    (so the pairing that our published contrasts enjoy is preserved). Compare to that contrast's H-R14
    snippet-bootstrap half-width `h` (`ridge_map` h = 0.0414; `codesteer_auto` h = 0.0379).
    **`CI-OPTIMISTIC`** if any arm has `s ≥ h/2` · **`CI-HONEST`** if all arms have `s ≤ h/3` ·
    **`CI-MARGINAL`** otherwise. *Prediction: **`CI-OPTIMISTIC`***. If it fires, every interval in this
    family — H-R2, H-R6, H-R7, H-R14 — is a lower bound on its true uncertainty, and that gets stated
    in the thread README rather than buried here.
  - **H-R15b (is a replicate even a replicate?) — the control that decides how to read H-R15a.**
    Re-run `unsteered` at the **same** seed on the **same** 148 snippets.
    **`SEED-DETERMINISTIC`** if it reproduces H-R14's `unsteered` accuracy to ≤ 0.005 **and** per-item
    label agreement ≥ 0.99 · **`SEED-NOT-DETERMINISTIC`** otherwise.
    *Prediction: **`SEED-NOT-DETERMINISTIC`***, from the 0.8333 figure above. **This is the more
    important of the two**: if a same-seed re-run already drifts as much as a different-seed re-run, then
    fixing seeds buys nothing, the floor is intrinsic to sampled autoregressive decoding on this cluster,
    and the only remedies are more runs per case or greedy decoding.
  - **H-R15c (does the one effect that cleared zero survive?).** H-R14's sole arm whose CI excluded zero
    (uncorrected) was `codesteer_auto`, +0.0379 [+0.0009, +0.0766] — **theirs, not ours**. At a new seed:
    **`AUTO-REPLICATES`** if Δ ≥ +0.02 with the same sign · **`AUTO-DOES-NOT-REPLICATE`** if Δ ≤ +0.01 or
    negative · **`AUTO-PARTIAL`** in between. *Prediction: **`AUTO-REPLICATES`**, held with low
    confidence* — it was positive on both snippet halves (+0.0154 / +0.0454) and about half of it is a
    parse-compliance effect, which ought to be the stable half.
  - **Descriptive, no verdict:** per-item label agreement between replicates, for comparison with the
    0.8333 banked on the other pipeline; and the same split of Δ into parse-rate and
    accuracy-given-parsed that H-R14 used.

  **Pre-committed consequence, so it cannot be negotiated afterwards:** if H-R15a returns
  `CI-OPTIMISTIC`, then **no steering claim in this programme may be made from a single run**, and the
  H-R14 report and artifact get an added line saying so. That includes claims that favour us.

- **Setup:** the frozen 148-snippet set (`full/snippets_148.json`, sha `9056b6b6aba9b0b6…`), same packs,
  same runtime, same chat template, same sampler (T 0.7 / top_p 1.0 / top_k 7, 3 runs/case), same
  cross-fitted vectors for `ridge_map` (`full/vectors_codellama7b_L7_full.pt`, unchanged — so this is a
  pure decoding replicate, not a re-fit). CodeLlama-7B-Instruct.

  **The one code change:** `ase_steer_run.py` gains `--seed` (new sha `9fc1389f2075acf5…`, previous
  `cad05d864ac91eaa…`). It replaces the three uses of the hard-coded module constant — `torch.manual_seed`,
  the `foreign` arm's `default_rng`, and their `rand` prior's `rand_seed` — and **defaults to that same
  constant**, so every invocation banked before today is unchanged; the diff is 5 sites and touches
  nothing else. The seed is now stamped into each run's residual-meta provenance.

  **Replicates (5 runs, all on the same 148 snippets):**

  | id | arm | seed | purpose |
  |---|---|---|---|
  | A2 | `unsteered` | **20260724** (same as H-R14) | H-R15b — is a replicate a replicate? |
  | B1 | `unsteered` | 20260917 | baseline at a new seed |
  | C1 | `unsteered` | 20260918 | third point → a real between-run spread |
  | B2 | `ridge_map` | 20260917 | our arm, paired with B1 |
  | B3 | `codesteer_auto` | 20260917 | H-R15c, paired with B1 |

  Seeds **B = 20260917 and C = 20260918** are today's and tomorrow's date, fixed here before any run —
  not chosen from results. Contrasts are always computed **within** a seed (B2−B1, B3−B1) so no contrast
  ever straddles two decoding streams.

  ~6 GPU-h (`codesteer_auto` is 4 shards × ~55 min; the rest ~35–40 min each), three
  `--dependency=singleton` lanes, **≤ 3 concurrent GPU jobs**. Scoring reuses
  `ase_r14_stats.py` primitives; the replicate-specific analysis is a new `ase_r15_repro.py`.

- **Results:** pending — this entry is the pre-registration.

- **What worked / hypothesis verdict:** pending.

- **Observations (recorded before the data):**
  - The cheapest possible version of this experiment would have been "re-run one arm at a new seed".
    That would have been **uninterpretable**, because without A2 a drift could be blamed on the seed when
    the 2026-08-29 result says it probably cannot be. The control is the experiment.
  - H-R15c is a test of *their* arm, and the pre-registered thresholds are the same ones that would have
    applied to ours. If it replicates, the honest summary of this whole programme becomes "the only
    steering effect that survives replication is CodeSteer's, and roughly half of it is format
    compliance" — a conclusion that does not favour the instrument this sub-project exists to build.
  - What this cannot do: three replicates give a crude spread, not a precise variance. The verdict
    thresholds are set as ratios against a known half-width for exactly that reason.

- **New questions / new hypotheses:** deferred to the results entry.

- **Next Steps:** smoke `--seed` wiring on 2 snippets → submit the 5 replicates → score → results entry,
  and if `CI-OPTIMISTIC` fires, amend the H-R14 report and artifact with the floor.
