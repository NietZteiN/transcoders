### Target Date: 2026-09-15 (H-R2 (ASE) RESULTS — the steering bake-off on CodeLlama-7B × 50: `MATCH-CODESTEER`, `SLICE-IRRELEVANT`, nothing but the oracle clears the noise floor)

- **Hypotheses / what we're testing:** the frozen rules of [2026-09-14_steering-bakeoff-prereg.md](2026-09-14_steering-bakeoff-prereg.md)
  as amended by [2026-09-14_bakeoff-runtime-amendment.md](2026-09-14_bakeoff-runtime-amendment.md) (H-R2c denominator =
  the original packs through *their* runtime), [2026-09-14_bakeoff-beta-fault.md](2026-09-14_bakeoff-beta-fault.md)
  (comparator = better of `codesteer`/`codesteer_auto`; β = 0 runs are the noise floor) and
  [2026-09-14_residual-arms-amendment.md](2026-09-14_residual-arms-amendment.md) (`swap_oracle`/`foreign`/`erasure`/`combined`).
  - **H-R2a** best-of-ours ∈ {`erasure`, `prompt`} (`knockout` never built) − better paper arm: `SURPASS` / `MATCH`
    (CI ∋ 0 and |Δ| < 0.05) / `BELOW`, read on the Bonferroni α/3 interval. Predicted `MATCH`.
  - **H-R2b** `codesteer − rand_prior`: `SLICE-MATTERS` if CI excludes 0, else `SLICE-IRRELEVANT`. Predicted `SLICE-IRRELEVANT`.
  - **H-R2c** restoration ratio `(steered − unsteered)/(original − unsteered)` per arm, descriptive; their figure 104.99 %.
  - **H-R4** (descriptive, no rule): raw-runtime damage `original_unsteered − unsteered` predicted **larger** than the chat-templated +0.113.
- **Setup:** their runtime `/scratch/juno/jvl210002/ase2026/LLM-Attention-Fixation_submission` (`SteeredCausalLM`, llama backend,
  raw prompt, greedy, 3 runs × cases per snippet), CodeLlama-7B-Instruct, the 50 `packs_orig_subset` snippets renamed; `codesteer` =
  README-exact β_post 0.8 over layers 24..31 with the `Level2Effect` gate live (mean L1 attention shift 0.221, 1 236 effective
  level-2 calls/run); `codesteer_auto` = Eq. 10 head calibration top-4/layer (shift 0.275); residual arms via `PositionReplacer`
  at K = 7, β = 1. Jobs: `unsteered` 399522 · `swap_oracle` 399685 · `prompt` 399686 · `foreign` 399687 · `erasure` 399688 ·
  `original_unsteered` 399863 · `codesteer` 400250 · `codesteer_auto` 400251 · `rand_prior` 400252 · `uniform_prior` 400253 ·
  `combined` 400254 · β = 0 identity runs 399523/399524 (noise floor). Scored with `nla/src/ase_bakeoff_stats.py`
  (sha `69f8e29f…`): `python nla/src/ase_bakeoff_stats.py --dir /scratch/juno/jvl210002/ase2026/bakeoff_codellama7b --original
  …/original_unsteered.jsonl --out …/bakeoff_stats.json` — paired per snippet, case-weighted Pass@1, cluster bootstrap
  N_BOOT 10 000 seed 20260724, n = 50. Kept copy: [`../../nla/results/2026-09-15_ase_bakeoff/`](../../nla/results/2026-09-15_ase_bakeoff/).
- **Results (case-weighted Pass@1; Δ vs `unsteered` 0.5323, 95 % CI):**

  | arm | Pass@1 | Δ vs unsteered | parse rate |
  |---|---|---|---|
  | `original_unsteered` (the clean code, same runtime) | **0.5576** | +0.0253 (damage) | 0.76 |
  | `codesteer_beta0` / `rand_prior_beta0` (identity, noise floor) | 0.5991 / 0.5991 | +0.0668 [−0.082, +0.212] | — |
  | `codesteer` (paper, all heads) | **0.6083** | +0.0760 [−0.081, +0.233] | 0.84 |
  | `codesteer_auto` (paper, calibrated heads) | 0.5760 | +0.0438 [−0.087, +0.175] | — |
  | `rand_prior` | 0.5853 | +0.0530 [−0.086, +0.192] | — |
  | `uniform_prior` (attention shift 0.0002) | 0.5968 | +0.0645 [−0.089, +0.211] | — |
  | `prompt` (ours, warning text) | **0.6336** | +0.1014 [−0.038, +0.233] | 0.84 |
  | `erasure` (ours, LOO mean-difference at K = 7) | 0.5760 | +0.0438 [−0.124, +0.201] | 0.83 |
  | `foreign` (other snippet's state) | 0.5737 | +0.0415 [−0.110, +0.196] | 0.96 |
  | `combined` (erasure + codesteer) | 0.5438 | +0.0115 [−0.123, +0.147] | — |
  | `swap_oracle` (item's own clean state, ceiling) | **0.7350** | **+0.2028 [+0.065, +0.346]** | 0.88 |

  - **H-R2a:** `prompt − codesteer` = **+0.0253** 95 % [−0.108, +0.161], α/3 [−0.138, +0.194] → **`MATCH-CODESTEER`**.
    `codesteer_auto − codesteer` = −0.032 [−0.192, +0.128] (the calibrated subset is not better).
  - **H-R2b:** `codesteer − rand_prior` = **+0.0230** [−0.115, +0.164] → **`SLICE-IRRELEVANT`**.
  - **H-R2c:** the denominator is +0.0253 (0.5576 − 0.5323), so every ratio is unstable: `codesteer` 300 % [−1000, +1214],
    `prompt` 400 %, `swap_oracle` 800 %, `role_proto` 464 %, `combined` 45 %. Reported, not interpreted.
  - **H-R4:** raw-runtime damage **+0.025** vs chat-templated **+0.113** — the prediction was the opposite sign of the
    difference. Parse rate is 0.76 on the *original* code vs 0.90 renamed.
  - Cross-reference (registered under H-R6, scored there): `role_proto` 0.6498, `role_proto − codesteer` = +0.042 [−0.099, +0.177].
- **What worked / hypothesis verdict:**
  - **H-R2a `MATCH-CODESTEER` — SUPPORTED as predicted**, but in the weak sense: the paper's steering (+0.076) and our best arm
    (+0.101, the *prompt*) are both inside the ±0.14 noise floor that the two identity runs set (+0.067 with β = 0 — i.e. the
    run-to-run spread of *their own unsteered decoding* is the size of every steering effect here). Only `swap_oracle`
    clears it (+0.203 [+0.065, +0.346]).
  - **H-R2b `SLICE-IRRELEVANT` — SUPPORTED as predicted**: the AST slice prior is worth +0.023 over a random prior, and the
    uniform prior — which moves attention by 0.0002 — scores the same as everything else (+0.065).
  - **H-R2c**: not readable — their 104.99 % figure is computed against a −36-point damage; ours is +2.5 points.
  - **H-R4 REFUTED**: the raw prompt makes the model *worse on clean code* (0.5576 vs H-R1's chat-templated 0.7995) far more
    than it makes renaming more damaging. The paper's damage magnitude is not recoverable on CodeLlama-7B under either format.
- **Observations:**
  - The bake-off's real finding is the **denominator**: on these 50 snippets renaming costs 2.5 points in their runtime, so
    "restoration" has no meaning and every arm — theirs and ours — is a noise-floor comparison. This is H-R1's `DAMAGE-PARTIAL`
    verdict again, now with the paper's own steering on top of it.
  - Parse rate confounds every Pass@1 (unparsed = wrong). Crude conditional accuracy `Pass@1 / parse rate`: `unsteered` 0.59 ·
    `codesteer` 0.73 · `prompt` 0.76 · `erasure` 0.70 · `foreign` 0.60 · `swap_oracle` 0.83 · `role_proto` **0.65** at parse 0.998.
    So `codesteer` and `prompt` *lower* the parse rate and raise conditional accuracy, while the residual writes mostly change
    *whether the model answers* — the 2026-09-13 parse-diagnostics split (obfuscation → not answering; full replacement →
    answering) reappears on a different host, runtime and stimulus set.
  - `combined` (erasure + codesteer) is the worst steered arm (0.5438): the two interventions do not add and probably interfere.
  - `codesteer_auto`'s head calibration does not beat all-heads on this host (−0.032); the head question (H-R5) stays open but
    now has a floor to beat.
- **New questions / new hypotheses:**
  - **H-R7 (ASE): a damage-bearing stimulus set.** Nothing can be ranked on a +0.025 denominator. Candidate: select snippets by
    a *pre-registered* damage screen on a held-out run (not on these 50), or use a host where H-R1 found damage (none at ≥ 0.15).
  - Is the parse-rate route the whole story for `role_proto`? Needs the raw replies (`--save-replies`) and a parsed-only paired
    contrast. Deferred to the H-R6 entry.
- **Next Steps:** H-R6 results entry when `ridge_map` (401537) lands (b/c already computable); H-R5 heads-vs-CodeSteer now
  has its comparator; decide H-R7 before any further arm on these 50.
