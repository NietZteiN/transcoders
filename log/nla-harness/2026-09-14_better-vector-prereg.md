### Target Date: 2026-09-14 (H-R6 (ASE) — a better deployable vector: pre-registration, before any pool capture)

- **Why.** The bake-off's residual arms so far read `foreign` **0.5737** ≈ `erasure` **0.5760** ≪ `swap_oracle`
  **0.7350** (noise floor ±0.14, nothing scored yet). The training-free mean-difference vector is doing no more
  than an unrelated snippet's state, while the item's own clean state recovers most of the damage. This is the
  Gemma tier ladder's `ERASURE-FLOOR` again ([`2026-09-13_tier-ladder-results.md`](2026-09-13_tier-ladder-results.md):
  `L1 − L1b` −0.035, `L0 − L1` +0.083) and W16's split (meaning ≈ 51 %, decoy removal ≈ 49 %): a vector that
  only removes the decoy is bounded at ≈ 0 on accuracy; only a **meaning-installing** write has headroom.
  The user asked (2026-09-14) for a better vector found *across the dataset* using NLA. This entry freezes the
  ladder of oracle-free meaning sources and their rules before any new activation is captured.
- **Hypotheses (H-R6, all paired per snippet vs `unsteered`, cluster bootstrap N_BOOT 10 000 seed 20260724, same
  50 test snippets / packs / decoding / parser as H-R2):**
  - **H-R6a `MAP-BEATS-MEAN`:** a reduced-rank ridge map from the decoy state to the clean-minus-decoy delta,
    fit on the *other* snippets, beats the mean-delta (`erasure`) on Pass@1 by ≥ 0.05 with the paired CI
    excluding 0. **Pre-GPU gate** (frozen): the map must beat the mean-delta model on held-out pool cosine to
    the true delta by ≥ 0.05; if it does not, `ridge_map` is not run and H-R6a reads `MAP-LEARNS-NOTHING`.
  - **H-R6b `CATEGORY-MEANING-HELPS`:** a (type, role)-matched prototype of other snippets' clean states beats
    `foreign` (the unmatched version of the same write) by ≥ 0.05 with CI excluding 0. `foreign < role_proto <
    swap_oracle` is the predicted ladder; `role_proto ≈ foreign` reads `CATEGORY-MEANING-INERT`.
  - **H-R6c `LATENT-BEATS-PROMPT`:** the best of {`ridge_map`, `role_proto`} beats `prompt_types` (the same
    type + role information given as text, the CLAUDE.md §4 mandatory prompting baseline) with CI excluding 0;
    else `PROMPT-SUFFICES`. This is the reading that decides whether any latent write is worth its cost.
  - Deferred, own prereg when built: `swap_guess` (states of a prompt de-obfuscated by the host's own
    per-identifier guesses; partner = the de-obfuscated prompt as text) and `ar_role` (an NLA AR trained on
    CodeLlama-7B at L7 writing `AR(type + usage description)`; needs the Gemma-specific pipeline ported to Llama,
    ~1 day + ~20 GPU-h — held until H-R6a–c say latent writes have headroom over prompting).
- **Frozen (config [`../../nla/configs/ase_vectors.yaml`](../../nla/configs/ase_vectors.yaml), sha256 `ac90157771f60444…`):**
  - **Split:** TEST = the 50 bake-off snippets (`gate/packs_orig_subset.jsonl`); FIT POOL = every other aligned
    snippet of the 156 renamed packs (abort if < 60 align). No LOO on the test set any more: nothing from a
    test snippet's original prompt, clean state or true names enters a vector written to it.
  - **Pool capture:** `prepare_residual` as in the bake-off (their `SteeredCausalLM` prompt builder, K = 7,
    `align()` + measured BOS offset), per span `h0_mean`, `h1b_mean`, per-token states, plus tags from
    `ase_roles.py`: declared **type** (javalang declaration lookup on the renamed source), **role** ∈ {method,
    parameter, loop_index, iterated, accumulator, returned, local} by first matching rule in that order,
    **kind** ∈ {method, variable}.
  - **`ridge_map`:** `delta_hat = (h1b_mean − μ)·A + c`, `v = h1b_mean + delta_hat` at every decoy token;
    λ ∈ {1e2…1e6}, rank ∈ {4, 16, 64, 256} by grouped 5-fold CV over the pool (groups = snippets, criterion =
    held-out cosine to the true delta). `A = 0` is exactly the `erasure` construction, so the arm nests it.
  - **`role_proto`:** `v` = mean `h0_mean` over pool spans with the same `type_role` tag; fallback `type`, then
    `kind`, when a bucket has < 5 spans; fallback share reported.
  - **`prompt_types`:** the `prompt` arm's WARN line plus one line per renamed identifier
    ``- `{decoy}`: declared `{type}`, role: {role}`` — the same information the vectors use, as text.
  - **Identity gate, effect gate, arm_guard, T = 0.7 × 3 runs, Pass@k:** unchanged from the bake-off.
- **Predictions, stated now:** H-R6a **`MAP-BEATS-MEAN`** in activation space (the map sees the declaration
  through context) but **below 0.05** on Pass@1 at n = 50; H-R6b **`CATEGORY-MEANING-HELPS`** small (+0.03–0.08);
  H-R6c **`PROMPT-SUFFICES`** — the tier ladder says token-space reconstruction is worth more than any latent
  write we have, and `prompt` already reads 0.634.
- **Verification before GPU:** `nla/tests/test_ase_vectors.py` — role tagger on `Java_000` (auths → `List<Double>`,
  iterated; startupargs → `double`, parameter; fileName → `double`, local; socketconnection → method); ridge fit
  recovers a planted low-rank map on synthetic data and returns `A = 0`-equivalent vectors when `rank = 0`;
  `role_proto` fallback hierarchy; the test-set exclusion (a test snippet's own `h0` never in its vector).
- **Cost:** pool capture ≈ 156 × 2 forwards, one job ≈ 10 min; fit on CPU; three arms × 0.7 GPU-h.
- **Next Steps:** write `ase_roles.py`, `ase_pool_capture.py`, `ase_vectors.py`, the three arms; tests; capture
  job after the H-R2 arms have scored (≤ 3 of ours on GPU at once).
