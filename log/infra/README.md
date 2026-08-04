# infra — pipeline code, environment, compute plumbing

*Last updated: 2026-08-04*
**Status:** active

Engineering thread: the extraction/steering/probing harnesses, conda env, configs, and the
discipline machinery (seeds, GPU pinning, provenance). Experiments consume what this thread builds.

## Hypotheses — open
- (none — engineering thread; correctness is gated by smoke + review, not hypotheses)

## Hypotheses — resolved
- (none)

## What worked
- Scaffold + smoke-first discipline: two transformers deprecations and a flagship-corrupting
  token-alignment bug caught before any GPU hour was spent.
- Adversarial review workflow (34 agents, 3 lenses + refuters): 28 confirmed findings, all fixed;
  smoke now exercises the same identifier-span path the real E1 run uses (mean resolution 1.000).
- Verify-or-nothing dictionary hunt (6 agents): all 14 registry entries + 12 model ids pinned with
  HfApi-verified shas, zero guesses; env built + frozen; stimuli/behavioral tables located + linked.

## What didn't
- First smoke design validated only `token_positions='last'` — the risky branch (identifiers)
  was never exercised, hiding the static-token-index bug. Fixed; lesson: smoke the real code path.
- circuit-tracer's pins silently downgraded transformers/transformer-lens/nnsight during install —
  caught by `pip freeze` diff; env pins now document the install-order dependency.

## Open ideas
- **Q-norm (blocking E1 analysis):** FVU 0.506 with cosine 0.773 on Instruct activations — test
  whether Llama Scope's dataset-wise `norm_scaling_factor` / TransformerLens-hook convention
  closes the gap; else switch E1 primary to the Goodfire l19 Instruct-trained SAE.
- Full Dataset-A E1 sweep (100 rows, layers 12/16/20) + auto-interp decoy/true feature sets.
- NLA replication check (`nla/src/replicate_example.py`) — gear verified complete 2026-08-04.
- Verify the `facebook/crv-8b-instruct-transcoders` lead (would de-confound E3-on-8B).
- Gated licenses still needed only for gemma-2-2b / Llama-3.2-1B (Llama-3.1-8B-Instruct turned
  out to be already cached); E3 PoC re-planned onto Qwen3-0.6B + mwhanna PLTs meanwhile.

## Entries
- `2026-08-04_e1-path-live.md` — stimuli converter (cross-tier spans, decoy↔true pairing), SAE loaders, real-model E1 smoke on Llama-3.1-8B (GPU); first transfer numbers (FVU 0.506 / cos 0.773 / L0 28) → Q-norm open.
- `2026-08-04_env-dictionaries-data.md` — env finalized + frozen; 14 dictionaries + 12 models pinned; stimuli/behavioral located + symlinked.
- `2026-08-03_pipeline-scaffold-smoke.md` — scaffold built; smoke + CWD-independence pass; 28 review findings fixed.

## Doc / results links
- [`../../src/README.md`](../../src/README.md) — module map + post-review behavior.
- [`../../docs/CHECKLIST.md`](../../docs/CHECKLIST.md) — Phase-0 boxes this thread is burning down.
