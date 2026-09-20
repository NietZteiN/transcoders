# instrument3 — the charter's own experiments (E1–E8)

*Last updated: 2026-09-20*
**Status:** active — Tier 0 complete; E1 unblocked.

The `nla-harness` thread carried the ASE/CodeSteer replication (closed 2026-09-18 on both obfuscation
routes). This thread carries **Instrument 3 itself**: the SAE/transcoder experiments in `CLAUDE.md` §3,
which had never run because the environment was unusable on this cluster.

## Hypotheses — open
- **T0.4:** why is L0 ≈ 22–27 against a trained `top_k` of 50, uniformly across layers and checkpoints? Check each layer's own `hyperparams.json` for a layer-specific `dataset_average_activation_norm`. Free.
- **T0.5:** does the 32× width variant reconstruct materially better at L20? Decides which dictionary E1 uses.
- **E1:** semantic-capture feature diff (L0 vs L1b) at L16/L20 — the charter's first committed experiment.

## Hypotheses — resolved
- **T0.1/T0.2 ✓ (2026-09-19)** — env restored. The blocker was `environment.lock.txt`, not the machine: a `@ file://` local build path on line 93, plus a pinned `torch==2.13.0+cu130` that needs a CUDA 13 driver this cluster does not have.
- **T0.3 ✓ `DICT-MARGINAL` (2026-09-20)** — [`2026-09-20_t03-dictionary-gate.md`](2026-09-20_t03-dictionary-gate.md). Base→instruct transfer is **not** the problem (≤0.049) and the reconstruction ceiling is uniform, so L0-vs-L1b contrasts stay interpretable. Largest correction: **exclude BOS** (FVE −2100.8 → +0.444).

## What worked
- Delegating SAE loading to SAELens's official `llama_scope` loader, so a bad number means a bad dictionary rather than a bad reimplementation.
- Testing three competing explanations for the low FVE instead of accepting the first verdict.

## What didn't
- Four gates in a row fired for the wrong reason (a connectivity check run without the variable that breaks it; a single-package pip test that skips the failing code path; a provenance check comparing raw against transformed weights; a smoke that included BOS). Each cost more wall-clock than the bug it was hiding.

## Entries
- [2026-09-20_t03-dictionary-gate.md](2026-09-20_t03-dictionary-gate.md) — T0.3 `DICT-MARGINAL`; provenance confirmed to 1e-14; BOS correction; three refuted hypotheses.
