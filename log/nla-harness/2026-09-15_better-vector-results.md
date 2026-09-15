### Target Date: 2026-09-15 (H-R6 (ASE) RESULTS — `role_proto` is the best non-oracle arm, but the prompting baseline collapsed on format and the honest reading is `PROMPT-SUFFICES`)

- **Hypotheses / what we're testing:** the frozen rules of [2026-09-14_better-vector-prereg.md](2026-09-14_better-vector-prereg.md)
  (paired per snippet, case-weighted Pass@1, cluster bootstrap N_BOOT 10 000 seed 20260724, n = 50, same runtime/packs/parser as H-R2):
  - **H-R6a `MAP-BEATS-MEAN`:** `ridge_map − erasure` ≥ 0.05 with CI > 0; else `MAP-NOT-BETTER`. Gate passed 2026-09-15 (cos 0.378 vs 0.174).
  - **H-R6b `CATEGORY-MEANING-HELPS`:** `role_proto − foreign` ≥ 0.05 with CI > 0; else `CATEGORY-MEANING-INERT`.
  - **H-R6c `LATENT-BEATS-PROMPT`:** best of {`ridge_map`, `role_proto`} − `prompt_types` with CI > 0; else `PROMPT-SUFFICES`.
  - Predictions: a positive only in activation space, b small, c `PROMPT-SUFFICES`.
- **Setup:** vectors `/scratch/juno/jvl210002/ase2026/vectors_codellama7b_L7.pt` (fit [2026-09-15_pool-fit-gate.md](2026-09-15_pool-fit-gate.md):
  FIT POOL 98 snippets / 3 506 spans, λ = 100, rank 256; prototypes at `type|role` 1 214 / `type` 64 / `kind` 15 of 1 293 test spans);
  `nla/scripts/nla_r2_arm_sbatch.sh` → `nla/src/ase_steer_run.py --vectors …` at K = 7, β = 1 (`PositionReplacer`): `role_proto`
  **401535** · `prompt_types` **401536** · `ridge_map` **401537** (h200/h100, 1 GPU each, chained ≤ 3 of ours). Scored by
  `nla/src/ase_bakeoff_stats.py` (sha `69f8e29f…`) with the H-R2 files; kept in
  [`../../nla/results/2026-09-15_ase_bakeoff/`](../../nla/results/2026-09-15_ase_bakeoff/) (stats + fit report + dated config).
- **Results (case-weighted Pass@1; `unsteered` 0.5323):**
  - `role_proto` **0.6498** (+0.1175 [−0.035, +0.271] vs unsteered; parse rate **0.998**) · `ridge_map` **0.6014** (+0.069
    [−0.091, +0.223]; parse 0.939) · `prompt_types` **0.4263** (−0.106 [−0.246, +0.036]; parse **0.633**).
  - **H-R6a:** `ridge_map − erasure` = **+0.0253** [−0.109, +0.163] → **`MAP-NOT-BETTER`**.
  - **H-R6b:** `role_proto − foreign` = **+0.0760** [−0.049, +0.196] → **`CATEGORY-MEANING-INERT`** (point estimate clears 0.05, CI does not).
  - **H-R6c (registered):** `role_proto − prompt_types` = **+0.2235** [+0.078, +0.364] → **`LATENT-BEATS-PROMPT`**.
  - **H-R6c (sensitivity, not registered — against the strongest prompting arm):** `role_proto − prompt` = **+0.0161** [−0.118, +0.145];
    `ridge_map − prompt` = −0.032 [−0.166, +0.107].
  - Other descriptive contrasts: `role_proto − codesteer` +0.042 [−0.099, +0.177] · `ridge_map − codesteer` −0.007 · `ridge_map − role_proto`
    −0.048 [−0.200, +0.101]. Crude conditional accuracy `Pass@1 / parse`: `role_proto` 0.65 · `ridge_map` 0.64 · `prompt` 0.76 · `codesteer` 0.73 · `unsteered` 0.59.
- **What worked / hypothesis verdict:**
  - **H-R6a REFUTED (`MAP-NOT-BETTER`).** A map that predicts the clean delta at cos 0.38 on held-out pool spans buys +0.025 over the
    mean-delta on accuracy — inside the noise floor. Held-out cosine did not transfer to accuracy, as the gate entry feared.
  - **H-R6b INCONCLUSIVE-leaning-null (`CATEGORY-MEANING-INERT`).** +0.076 is the largest non-oracle paired gain in the whole bake-off
    and the point estimate clears the bar, but the CI spans −0.05 … +0.20 — n = 50 cannot settle a 0.05 effect.
  - **H-R6c: the registered verdict is `LATENT-BEATS-PROMPT`, and I do not believe it.** `prompt_types` collapsed the parse rate to 0.63
    (the 7B Instruct model stops emitting the bare JSON once a fact list precedes the instruction — the injected lines were verified correct,
    e.g. Java_001 `` `peers`: declared `int`, role: accumulator ``), so the registered baseline is degenerate. Against the intact prompting
    arm the latent write is +0.016 with a CI of ±0.13 — **`PROMPT-SUFFICES`**, which is what the prereg predicted. Both readings are
    reported; the conservative one is the one carried into the ledger.
- **Observations:**
  - `role_proto` reaches parse 0.998 — higher than any arm including the oracle (0.88). Its Pass@1 lead over `prompt` is therefore
    mostly *answering more often*, not answering better (conditional 0.65 vs 0.76). This is the same split the 2026-09-13 parse
    diagnostics found on Gemma: residual writes at a shallow layer change whether the model answers; prompts change how well.
  - The ridge map's CV optimum sat at the grid corner (λ 100, rank 256) and still lost to a category prototype — a 4 096-d linear map
    from 3 506 samples is data-starved; more pool (H-R7's larger stimulus set would double as a larger pool) is the cheap lever.
  - Prereg wording discrepancy carried: "auths → iterated" vs the frozen rule's `parameter`; the code is the rule.
  - Everything here is bounded by H-R2's finding: renaming costs +0.025 on these 50 in their runtime; there is nothing to restore.
- **New questions / new hypotheses:**
  - **H-R7 (ASE): damage-bearing stimuli.** Pre-register a damage screen on held-out snippets (not these 50), then re-run the four arms
    that matter (`unsteered`, `codesteer`, `prompt`, `role_proto`, `swap_oracle`) — with `--save-replies`.
  - **H-R8 (ASE): is `role_proto`'s effect a compliance effect?** Parsed-only paired contrast from saved replies; if conditional
    accuracy is flat, the category prototype is a *format* steer and `CATEGORY-MEANING-INERT` is confirmed.
  - `prompt_types` should be re-issued in a format the model tolerates (facts *after* the instruction, or as code comments) before
    any latent-vs-prompt claim is made; that is a new registered arm, not a retune.
  - `swap_guess` and `ar_role` stay **deferred**: H-R6a–c give no evidence that latent writes have headroom over prompting on this
    host and stimulus set, so an NLA AR port to CodeLlama (~1 day + ~20 GPU-h) is not justified yet.
- **Next Steps:** write H-R7's prereg (stimulus screen + `--save-replies`) before any further arm; H-R5 heads-vs-CodeSteer can
  proceed on the existing files.
