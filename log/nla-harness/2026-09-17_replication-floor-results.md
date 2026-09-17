### Target Date: 2026-09-17 (H-R15 RESULTS — the pipeline is exactly reproducible, and nothing in the bake-off survives a different draw)

- **Hypotheses / what we're testing:** as pre-registered in
  [`2026-09-17_replication-floor-prereg.md`](2026-09-17_replication-floor-prereg.md). Predictions on
  record: **a** `CI-OPTIMISTIC` · **b** `SEED-NOT-DETERMINISTIC` · **c** `AUTO-REPLICATES` (low confidence).

- **Setup:** the frozen 148-snippet set, same packs, same runtime, same chat template, **same cross-fitted
  vectors** (`ridge_map` is a pure decoding replicate, not a re-fit). `ase_steer_run.py` gained `--seed`
  (sha `9fc1389f2075acf5…`, was `cad05d864ac91eaa…`; 5-site diff, defaults to the frozen constant so every
  earlier run is unchanged). Analysis `ase_r15_repro.py` sha `37345d62a8a70cf2…`. Five replicates, jobs
  409831–409838 (+ smoke 409820), all rc=0, nodes g-07-13 / g-04-02 / g-06-01, ≤ 3 concurrent GPU jobs.
  **A2 was pinned to the h200 partition to match A1's architecture**, so a non-reproduction could not be
  blamed on hardware. Contrasts computed **within** a replicate (B2−B1, B3−B1), never across seeds.

  | id | arm | seed | acc | parse | acc\|parsed |
  |---|---|---|---|---|---|
  | A1 (H-R14) | `unsteered` | 20260724 | 0.6801 | 0.9024 | 0.7536 |
  | **A2** | `unsteered` | **20260724 (same)** | **0.6801** | **0.9024** | **0.7536** |
  | B1 | `unsteered` | 20260917 | 0.6967 | 0.9093 | 0.7662 |
  | C1 | `unsteered` | 20260918 | 0.6818 | 0.9030 | 0.7550 |
  | A1 (H-R14) | `ridge_map` | 20260724 | 0.6770 | 0.8993 | 0.7528 |
  | B2 | `ridge_map` | 20260917 | 0.6745 | 0.9170 | 0.7356 |
  | A1 (H-R14) | `codesteer_auto` | 20260724 | 0.7179 | 0.9317 | 0.7706 |
  | B3 | `codesteer_auto` | 20260917 | 0.6936 | 0.9074 | 0.7644 |

- **Results:**

  **H-R15b `SEED-DETERMINISTIC` — my prediction REFUTED.** A1 vs A2: per-item label agreement
  **1.0000** over every case-run, **|Δacc| = 0.000000**, parse and acc|parsed identical to 4 dp — across
  *different nodes* (g-08-04 vs g-07-13, both h200). Given the seed, the snippet set and the order, this
  pipeline reproduces **exactly**. The 0.8333 floor banked on 2026-08-29 does not describe it.

  **Per-item agreement across seeds: 0.6107 – 0.6302** (six pairings). Changing only the draw flips
  roughly **38 %** of individual case-run labels — far below the 0.8333 of the other pipeline, and not far
  above the 0.50 that two independent draws on a coin-flip case would give.

  **H-R15a `CI-OPTIMISTIC` — as predicted.** Contrasts, each computed within its own replicate:

  | arm | Δ at seed A | Δ at seed B | drift | vs published half-width |
  |---|---|---|---|---|
  | `ridge_map` | −0.0031 | −0.0222 | 0.0191 | **0.46 ×** (h = 0.0414) |
  | `codesteer_auto` | **+0.0379** | **−0.0031** | **0.0409** | **1.08 ×** (h = 0.0379) |

  From the three independent `unsteered` draws (0.6801 / 0.6967 / 0.6818) the per-run decoding SD is
  **σ = 0.0092**, so a single-run contrast carries ≈ √2 σ = 0.0129 and a difference of two contrasts
  ≈ 2σ = 0.0183. Observed drift is 1.0 × that for `ridge_map` and **2.2 ×** for `codesteer_auto`.
  Folding decoding noise into the published intervals in quadrature widens them **≈ 1.17–1.20 ×**:
  `ridge_map` ±0.0414 → **±0.0486**; `codesteer_auto` ±0.0379 → **±0.0456**.

  **H-R15c `AUTO-DOES-NOT-REPLICATE` — my prediction REFUTED.** `codesteer_auto − unsteered` goes from
  **+0.0379 to −0.0031**. The single effect whose interval excluded zero in H-R14 is gone at a different
  draw.

  **The parse-compliance mechanism did not replicate either**, which pre-empts H-R16:

  | | Δ accuracy | Δ parse | Δ acc·given·parsed |
  |---|---|---|---|
  | seed A (H-R14) | +0.0379 | **+0.0293** | +0.0170 |
  | seed B (H-R15) | −0.0031 | **−0.0019** | −0.0018 |

- **What worked / hypothesis verdict:**
  - **H-R15a SUPPORTED** (`CI-OPTIMISTIC`). Every interval this thread has published on a single sampled
    run is too narrow; the correction is ≈ 1.2 ×.
  - **H-R15b REFUTED my prediction** (`SEED-DETERMINISTIC`). Good news mechanically — replication is
    exact and free to verify — and it **relocates the cause** of the H-R7 → H-R14 drift. That drift was
    never hardware and never a seed change (both used the same hard-coded seed): it was **RNG-stream
    position**. Running 148 snippets instead of 50 means every snippet after the first is generated from
    a different point in the sampling stream. The determinism is real but brittle in exactly the way that
    matters — *any* change to the corpus reshuffles every downstream draw.
  - **H-R15c REFUTED my prediction** (`AUTO-DOES-NOT-REPLICATE`). I predicted it would replicate, with
    low confidence, on the reasoning that its positive sign on both snippet halves and its
    parse-compliance component were evidence of something real. Both arguments were wrong.
  - **The programme-level consequence, pre-committed in the prereg and now owed:** *nothing in this
    bake-off survives replication — not ours, not theirs.* H-R14's `codesteer_auto` was the last arm
    standing and it does not survive. The pre-registered rule fires: **no steering claim here may rest on
    a single run**, and the H-R14 report and artifact are amended to say so.

- **Observations:**
  - **Aggregate stability hides item-level chaos.** Overall accuracy is stable to σ = 0.0092 while ~38 %
    of individual predictions flip. The flips are near-symmetric, so means survive what individual
    answers do not. **Any per-item analysis in this programme built on a single run is largely measuring
    the draw** — that includes per-item correlations with HCI, the flippable-item subsets, and any
    error-case inspection. This is a wider warning than H-R15 was scoped to give.
  - Two ways to buy the lost precision, both cheap relative to what has already been spent: **more runs
    per case** (σ falls as 1/√runs, so 12 runs instead of 3 halves it) or **greedy decoding** (σ → 0 by
    H-R15b, at the cost of departing from the paper's sampler). Greedy is the stronger option *because*
    the pipeline is exactly reproducible; it converts every future contrast into a deterministic
    measurement whose only uncertainty is over snippets.
  - **The oracle bound from H-R14 is the one result that is not threatened**, because it is an upper
    bound rather than a contrast: `swap_oracle` +0.0163 with a decoding-inclusive interval still cannot
    reach the ~+0.05 that beating CodeSteer would need. The conclusion "no better L7 vector can win"
    survives H-R15.
  - What H-R15 does **not** establish: σ = 0.0092 rests on three draws and each drift figure is a single
    difference, not a variance. The direction is unambiguous; the exact 1.2 × factor is not precise.

- **New questions / new hypotheses:**
  - **H-R18 (supersedes H-R16, which is void — there is no parse gain left to explain):** re-run the
    decisive arms under **greedy decoding** on the 148 snippets. By H-R15b this removes decoding noise
    entirely and makes every contrast a deterministic function of the corpus, at ~3 GPU-h since one run
    per case suffices. It is the cheapest way to make any future steering claim in this programme
    checkable, and it should run **before** CruxEval or a stronger renamer.
  - **H-R19:** re-read the per-item claims of Papers 2–3 against a 38 % label-flip rate. If per-item
    correctness is that unstable under sampling, per-item measures taken from single runs need a
    stability estimate attached before they enter the GLMM stack.
  - **H-R17 (unchanged)** — the damage gap to their Qwen2.5-7B still needs the stronger renamer.

- **Next Steps:** amend the H-R14 report + artifact with the floor (pre-committed); H-R18 greedy pass
  before any further steering work.
