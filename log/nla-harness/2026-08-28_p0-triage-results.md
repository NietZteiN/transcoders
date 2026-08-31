### Target Date: 2026-08-28 (Phase-0 triage — results and verdicts)

- **Hypotheses / what we're testing:** The frozen pre-registration
  [`2026-08-27_p0-triage-prereg.md`](2026-08-27_p0-triage-prereg.md). B4 is refuted and B5 is null,
  and both are currently read as *"there is no item-level belief to edit."* That reading is not
  licensed, because a second explanation predicts identical tables: **the injection channel cannot
  deliver anything, whatever it carries.** Phase 0 decides between them.
  - **H-P0.1** — the task direction `h_clean − h_obf` at layer 20 points the same way at layers
    21–27. SOFT if cos(Δ₂₀, Δ_ℓ) ≥ 0.50 for every ℓ ∈ [21,27]; HARD if it falls below anywhere.
  - **H-P0.2 (channel)** — widening the write from one token position to every reply position
    recovers the oracle. CHANNEL-LIMITED if V4 improves by **≥ +0.10** balanced Δaccuracy with a CI
    excluding zero and `R_random` does not match the gain; NOT CHANNEL-LIMITED otherwise.
  - **H-P0.3 (site)** — layer 20 is a live site for *some* intervention. SITE LIVE if the [20,20]
    attention-steering cell differs from the unsteered baseline by more than the baseline's own
    seed-to-seed spread, in the same direction at both seeds. SITE DEAD if [20,20] is within seed
    noise while [20,27] is not. UNINFORMATIVE if [20,27] is also within seed noise — explicitly
    **not** to be read as SITE DEAD.

- **Setup:** All three runs executed on the **origin box** `csr-94608.utdallas.edu` (4×A6000, no
  scheduler) on 2026-08-28 between 09:29 and 12:27 local, orchestrated by
  `nla/scripts/p0_autopilot.sh` under `setsid`. Scoring and adjudication were performed **after the
  migration**, on `juno-l-01` (SLURM, `/work/jvl210002/migration/transcoders`), CPU only.

  | | |
  |---|---|
  | seed | 20260724 (P0.1, P0.2) · 1000 + 2000 (P0.3) |
  | P0.1 | `nla/src/layer_rotation.py`, 60 L0/L1b pairs, all 28 layers, final prompt token, no NLA |
  | P0.2 | `nla/src/steer_run.py --positions all_reply --alphas 1.0 --only-conditions V4_oracle,V3_taskvec,R_random,V1_gloss --out-dir data/nla/p0/p02_allreply --max-hours 6` · `Qwen/Qwen2.5-7B-Instruct` L20 · 60 pairs, 30 wrong at L1b baseline · 300 rows, 0 errors · 2.199 h · finished 2026-08-28T16:42:26Z |
  | P0.2 comparison | the **banked** `last_prompt` sweep `data/nla/n12/steer_stats_sweep.json`, resumed not recomputed, at the same α = 1.0 |
  | P0.2 scoring | `nla/src/steer_stats.py --primary-alpha 1.0` with `--results/--baseline/--out` all passed explicitly (its defaults point at the banked B4 run) |
  | P0.3 | `nla/scripts/p03_site.sh` · `Qwen/Qwen2.5-Coder-7B-Instruct` · `adversarial_rename` · 164 Java programs · bands [20,20] [20,21] [20,23] [20,27] + unsteered baseline × seeds 1000/2000 · belief channel off (α = 0) |
  | instrument | `kitft/nla-qwen2.5-7b-L20-{av,ar}`, frozen released checkpoints — nothing trained |
  | adjudication | `nla/src/p0_verdict.py` (new this entry) → `data/nla/p0/P0_VERDICT.json`; tables via `nla/src/p0_summary.py` → `data/nla/p0/P0_SUMMARY.md` |

  Script sha256:
  ```
  9fb2a74952d66c61725777cf457d87a46be1ff4e63f721026538e198b55e3124  nla/src/steer_run.py
  b004359017cd27febee595fde160941b0dd336e8b66bf3a90999ac183d03d575  nla/src/steer_stats.py
  558e128cbf568cb945424371b889afac889fbb99ec82dd848eb2ac067ff7d3a3  nla/src/p03_score.py
  de495539e108c2c3ba015248cf16d0cfd04a5f7aaccfd330a680f1ca63987e5c  nla/src/p0_summary.py
  89862ebf591360931cdbe1c714d16cc6076ddb2759a53f7a057f6387d47d44e4  nla/src/p0_verdict.py
  d983c7bfaa1a84f19e3e7bf263f95923f12b13c2b31cb9bc67e5d825984d52ae  data/nla/p0/p02_allreply/steer_stats.json
  c8e46ad93ffe9fce3ff63cfd358fd2ae276ff912c4eb07de69efe67b2d06fa6c  nla/continuation/artifacts/p03_partial_cells.json
  ```

  **Provenance caveat on the P0.3 numbers, stated up front.** `p03_score.py` derives cells by
  walking the per-run `score.json` tree under `allocation_replication/artifact/artifacts/
  obfuscation/result/`. That tree is ~400 MB, is not in git, and **did not travel to juno**. The
  cell table used here is the copy the pre-migration session saved as
  `nla/continuation/artifacts/p03_partial_cells.json`, restored to
  `data/nla/p0/p03/cells_scored.json` with `cells_scored.PROVENANCE.md` beside it recording exactly
  that. Until the result tree is re-copied or the cells re-run, **no P0.3 cell can be re-scored on
  a different denominator here.**

- **Results:**

  **P0.1 — layer rotation.** Layer-index validation gate passed at **0.0e+00 relative error**
  (`hidden_states[K+1]` bit-identical to the hooked extractor), so the layer axis is correctly
  labelled and the curves are believable. min cos(Δ₂₀, Δ_ℓ) over ℓ ∈ [21,27] = **0.277**, against
  the frozen threshold of 0.50. Decay is smooth: L21 0.818 · L22 0.672 · L23 0.583 · L24 0.500 ·
  L27 0.277. Below L18 the direction is essentially unrelated to L20's (cos 0.016–0.374 over
  L0–L17). Cross-item leave-one-out **coherence peaks at L13 (0.543)**, not at the instrument's
  L20 (0.459); **relative magnitude peaks at L20**.

  **P0.2 — channel.** 300 rows, 0 errors, n = 60 per condition, primary α = 1.0. Baseline L0 acc
  0.600, L1b acc 0.500.

  | condition | Δacc `last_prompt` (banked) | Δacc `all_reply` (new) | contrast | `all_reply` CI95 | parse rate `last_prompt` → `all_reply` |
  |---|---|---|---|---|---|
  | **V4_oracle** | +0.0833 | **−0.3333** | **−0.4166** | [−0.4833, −0.2000] | 0.917 → **0.383** |
  | V3_taskvec | +0.0500 | −0.3667 | −0.4167 | [−0.5000, −0.2333] | 0.900 → **0.283** |
  | V1_gloss | +0.0500 | −0.4500 | −0.5000 | [−0.6000, −0.3000] | 0.917 → **0.100** |
  | R_random | +0.0333 | −0.1500 | −0.1833 | [−0.3167, +0.0167] | 0.917 → 0.850 |
  | P_prompt (α=0) | +0.1000 | +0.1000 | — | [−0.0167, +0.2333] | 0.900 → 0.900 |

  Reported unnetted, per the standing constraint — at `all_reply` α=1, V4 **recovered 3** /
  **damaged 23**; V1 recovered 2 / damaged 29; V3 recovered 1 / damaged 23; R_random recovered 8 /
  damaged 17.

  **P0.3 — site.** Baseline seed-to-seed spread = |64.974 − 64.180| = **0.794** points.

  | cell | seed 1000 | seed 2000 | Δ vs same-seed baseline | cases | complete |
  |---|---|---|---|---|---|
  | baseline | 64.974 | 64.180 | — | 5,790 × 2 | ✅ |
  | **[20,20]** | 63.541 | 62.522 | **−1.434 / −1.658** | 5,790 × 2 | ✅ |
  | [20,21] | 61.883 | 62.971 | −3.092 / −1.209 | 5,790 × 2 | ✅ |
  | [20,23] | 62.798 | 63.972 | −2.176 / −0.207 | 5,790 × 2 | ✅ |
  | [20,27] | 61.821 | 61.777 | −3.153 / −2.403 | 1,472 / 1,452 | ❌ **~25%, non-random subset** |

  Every complete cell has `runs_unscorable = 0` at 492 runs / 164 snippets.

- **What worked / hypothesis verdict:**

  - **H-P0.1 → HARD.** 0.277 < 0.50. Re-using the layer-20 vector across the 20–27 band is **not
    licensed**; the single-layer limit is now a *measured* constraint rather than an assumption.
    A narrow 20–22 band (0.818, 0.672) is arguably defensible but is exploratory, not
    pre-registered.
  - **H-P0.2 → NOT CHANNEL-LIMITED.** V4's primary contrast is **−0.4166**, not ≥ +0.10. It is not
    a near miss and not a power problem — the oracle moves the wrong way by four times the
    threshold's magnitude. `R_random` is not the disqualifier here; it degrades least of the four.
  - **H-P0.3 → SITE LIVE.** [20,20] moves accuracy **−1.434 / −1.658** points against a baseline
    seed spread of 0.794 — same direction, both seeds, both magnitudes above the noise floor. The
    rule says *differs*, not *improves*, and this is a consistent, reproducible degradation.
  - **Licensing table → row 4: "the site works but belief-shaped writes do not."** This is recorded
    in the pre-registration as **the strongest support for the "no item-level belief" reading**, and
    routes to **Phase 1b — the readout paper with a bounded, mechanistic causal negative.**

- **Observations:**

  - **The P0.2 result is stronger than "no delivery", and the mechanism is visible in the parse
    rates.** The alternative explanation Phase 0 existed to kill was *the channel cannot deliver
    anything*. Writing at every reply position delivers enormously — it moves accuracy by 0.33–0.45
    and drives V1's parse rate to **0.100**. The channel is not under-powered. It is that a fixed
    vector written at every reply position destroys fluent generation before it can carry content.
    B4's negative therefore stands as a claim about representations, not about plumbing.
  - **The content-bearing directions do more damage than random**, which is the informative part of
    the row. R_random degrades −0.183 while V1/V3/V4 all degrade −0.42 to −0.50 — and V4, which has
    seen the clean program, is indistinguishable from V3 and no better than V1. Perfect information
    buys nothing through this channel at this width.
  - **Accuracy at `all_reply` is confounded with parseability and should not be quoted as a clean
    accuracy.** With parse rates of 0.10–0.38, most of the accuracy drop is unparseable output
    scored as wrong. The verdict does not depend on separating the two — every route from these
    numbers fails the ≥ +0.10 improvement test — but the *magnitude* is not interpretable as
    comprehension damage. This is a silent-failure-class hazard: a reader taking −0.45 as "steering
    made the model wrong" would be over-reading it.
  - **On applying the P0.3 rule to an incomplete table.** `nla/continuation/00_STATE.md` says "do
    not apply the decision rule to this table" because [20,27] is partial. That is correct about the
    *design* and over-cautious about the *rule*: SITE LIVE is defined purely on [20,20] versus
    baseline, and those four cells are complete at full scale. [20,27] is required only to separate
    SITE DEAD from UNINFORMATIVE — branches this table never enters. `p0_verdict.py` encodes exactly
    that asymmetry and returns `NEEDS_DATA` rather than guessing on the branches that do need it.
  - **[20,23] at seed 2000 is the one cell inside the noise floor** (−0.207 vs 0.794). The scope
    curve is not monotone in band width, so it should be reported as four points, not as a trend.
  - **P0.1's dissociation is still the most re-usable finding here.** Coherence peaks at L13 while
    magnitude peaks at L20 — the same shape N13 found for `act_norm` vs `rt_cos`, in an unrelated
    measurement. V3 is a cross-item mean, so it was built at the depth where the task direction is
    *least shared* across items.
  - **Cluster migration.** All three runs predate the move; only scoring and adjudication ran on
    juno. Nothing in this entry required a GPU. The environment needed to *finish* [20,27] does not
    yet exist on the new box — see Next Steps.

- **New questions / new hypotheses:**
  - **HT-P0a** — is the `all_reply` collapse a *generation* failure rather than a comprehension
    failure? Predicts: restricting the write to a small window (e.g. the first k reply tokens, or
    reply positions only above layer 20) restores parse rate ≥ 0.85 while leaving Δaccuracy within
    noise of `last_prompt`. Confirmed if parse rate recovers and Δaccuracy does not; refuted if
    accuracy tracks parse rate one-for-one across the window sweep.
  - **HT-P0b (P0.4, still needs its own frozen rule before it runs)** — the instrument is at the
    wrong depth. V3/V4 are pure activation differences defined at every layer, so V3@13 vs V3@20 is
    testable with no autoencoder. Confirmed if V3@13 beats V3@20 by more than the seed spread.
  - **HT-P0c** — attention steering at [20,20] *hurts* by 1.4–1.7 points at both seeds. Is that a
    generic cost of perturbing any single layer, or specific to 20? A layer-matched control band
    (e.g. [13,13], [26,26]) at the same two seeds would say. Without it, "SITE LIVE" means "layer 20
    is not inert", which is what the rule asked, but not yet "layer 20 is special".
  - Does the licensing outcome change the B4 write-up? It should: B4's negative can now be stated
    as bounded by a *measured* live site rather than by an untested assumption about the channel.

- **Next Steps:**
  1. ~~Score P0.2~~ — already done on the origin box at 11:42; the handoff's "unscored" note was
     stale. Corrected in `CLAUDE_SCRATCHPAD.md`.
  2. Rebuild the environment on juno: `module load miniconda/24.11.1` → `nla-mi` + `codesteer`
     (they are **not** interchangeable — transformers 5.12 vs 4.57), clone
     `allocation_replication` (remote HEAD `47b6cfe`; confirm it carries `--steer-layers`), fetch a
     **JDK 21** (system java is 17; the module tree has only `java/11`), re-verify the G0
     chat-template patch.
  3. Port `p03_site.sh` from "pick idle GPUs with `nvidia-smi`" to `sbatch` on the `h200` partition
     and finish the two `[20,27]` cells — **no longer load-bearing for the verdict**, but it closes
     the pre-registered design.
  4. Re-copy `allocation_replication/artifact/artifacts/obfuscation/result/` (~400 MB) from the
     origin box, or accept that the eight finished cells cannot be re-scored.
  5. Clear the rest of `nla/continuation/03_LEDGER_DEBTS.md`: the missing B5 entry (2026-08-26), the
     superseded 08-17 programme report, and the four wording corrections.
