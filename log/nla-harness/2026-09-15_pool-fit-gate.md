### Target Date: 2026-09-15 (H-R6 (ASE) — pool captured, fit gate PASSED, the three arms submitted; one YAML fault)

- **Hypotheses / what we're testing:** setup + gate day for H-R6 ([2026-09-14_better-vector-prereg.md](2026-09-14_better-vector-prereg.md)).
  The only pre-registered test decided here is the **pre-GPU ridge gate**: held-out (grouped 5-fold by snippet)
  cosine of the reduced-rank ridge prediction to the true clean-minus-decoy delta must beat the mean-delta model
  by ≥ 0.05, else `ridge_map` is not run (`MAP-LEARNS-NOTHING`). No accuracy is read today.
- **Setup:**
  - Job **401200** `nla_r6_pool` (h200/h100, 1 GPU, `nla/scripts/nla_r6_pool_sbatch.sh`): tests →
    `nla/src/ase_pool_capture.py` (CodeLlama-7B-Instruct, K = 7, all 156 renamed packs, `SteeredCausalLM`
    steering OFF, `prepare_residual` alignment as the bake-off) → `nla/src/ase_vectors.py`. Pool saved to
    `/scratch/juno/jvl210002/ase2026/pool_codellama7b_L7.pt` (296 MB).
  - **Fault:** the fit stage crashed (`ufunc 'multiply' … dtype('<U5')`) because PyYAML reads `1.0e2` as a
    *string* (its float regex wants a signed exponent). Fix: config ladder rewritten `1.0e+2 … 1.0e+6` (same
    values; config sha `ac901577…` → **`fdca8630…`**) and `cv_select` / `main` now cast λ/rank to
    float/int (`ase_vectors.py` sha **`09b3defe…`**). Tests 4/4. The frozen λ/rank ladder is unchanged.
  - Fit re-run on CPU from the saved pool (login node, `OMP_NUM_THREADS=8`, 3.6 min):
    `python nla/src/ase_vectors.py` → `log/slurm/local_nla_r6_fit_20260915T051504Z.out`; outputs
    `/scratch/juno/jvl210002/ase2026/vectors_codellama7b_L7.pt` + `.fit.json`. Seed 20260724.
  - Arms submitted with `nla/scripts/nla_r2_arm_sbatch.sh`, chained so at most 3 of our GPU jobs run at once:
    `role_proto` **401535** (afterany 400251), `prompt_types` **401536** (afterany 400253), `ridge_map`
    **401537** (afterany 400254). Monitor `bnsj5dvev`.
- **Results:**
  - Pool: **148 aligned / 8 excluded** (Java_018/093/111/115/116/122/158 occurrence mismatch, Java_106 method not
    renamed) → 4 799 spans; tags missing 0; roles `method` 3 384 · `parameter` 499 · `local` 462 · `accumulator`
    225 · `returned` 152 · `iterated` 77. Split: **FIT POOL 98 snippets / 3 506 spans**, **TEST 50 / 1 293**
    (disjoint asserted). `min_pool_snippets` 60 cleared.
  - Ridge CV (held-out cosine to the true delta, mean over 5 folds): mean-delta model **0.1742**; grid
    λ=100: r4 0.230 · r16 0.272 · r64 0.321 · **r256 0.378**; λ=1e3: 0.230/0.275/0.324/0.364; λ=1e4:
    0.220/0.252/0.279/0.291; λ=1e5: ≤ 0.205; λ=1e6: ≤ 0.178. Best **λ = 100, rank 256, cos 0.3780**, margin
    **+0.2039** → **gate PASS**, `ridge_map` runs.
  - Prototypes: 1 293 test spans resolved at `type_role` 1 214 · `type` 64 · `kind` 15 · unresolved 0.
    Largest test buckets `int|method` 203, `boolean|method` 195, `List<Integer>|method` 169, `String|method` 133.
- **What worked / hypothesis verdict:** pre-GPU gate **PASSED** by 4× its bar — the decoy state carries
  recoverable information about its own clean delta beyond the pool mean (cos 0.38 vs 0.17). H-R6a–c stay open
  until the three arms land.
- **Observations:**
  - The CV optimum sits at the **corner of the frozen grid** (smallest λ, largest rank); a wider ladder
    (λ < 100, rank > 256) would probably score higher on CV. Not extended — the grid was frozen in the prereg
    and the point of the arm is a fair test of the ladder as registered; a wider grid is a follow-up, not a
    retune.
  - Roles are dominated by `method` (70 % of spans) — the renamed method name recurs on every call site, so the
    span count is not the identifier count. `role_proto` for the method spans is essentially a return-type
    prototype (`int|method`, `boolean|method`…).
  - Prereg wording discrepancy carried from yesterday: the verification bullet says `auths → iterated` but the
    frozen rule order (`ROLE_ORDER` parameter before iterated) and the test assert `parameter`. The code is
    the frozen rule; noted here rather than editing the entry.
- **New questions / new hypotheses:** does CV cosine (0.38) translate to accuracy at all? `swap_oracle` is
  cos 1.0 by construction and gives +0.20; a 0.38-cosine write may land nearer `erasure`. That is exactly H-R6a.
- **Next Steps:** wait for 400251/400253/400254 (H-R2) and 401535–401537 (H-R6); score H-R2 when its 12 files
  exist, then H-R6a–c from the same stats script; results entries for both.
