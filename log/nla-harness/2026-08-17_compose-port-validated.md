# 2026-08-17 — the NLA write hook ports to the codesteer environment (B5 unblocked)

**Why this mattered.** B5, E6 and E4's missing V5 all need ONE process driving both levers:
CodeSteer's attention reweighting and the NLA residual write. Those live in different conda
environments — `codesteer` (transformers 4.57.1) and `nla-mi` (transformers **5.12.1**) — and the
attention backend imports transformers-4.x decoder internals (`Qwen2Attention`,
`apply_rotary_pos_emb`, `repeat_kv`), so it cannot simply move to `nla-mi`.

The plan therefore went the other way: leave attention steering where it works and bring the NLA
hook to it. That rested on an **asserted** claim — "`steer.py` has no transformers coupling" —
which had never been tested in the target environment.

## Result: the claim holds

`nla/src/steer.py` imports only `torch` and stdlib, and resolves decoder layers by attribute path
(`model.layers` / `model.model.layers` / `transformer.h`), so nothing binds it to a transformers
version. Run under `/data/jvl210002/conda_envs/codesteer` (transformers 4.57.1, torch 2.9.1),
CPU, Qwen2.5-0.5B-Instruct (same Qwen2 decoder family as the 7B subject):

| property | result |
|---|---|
| **alpha = 0 byte-identical to unsteered** | **True** |
| alpha = 6 changes the output | True |
| positions changed by an edit at `[5]` | **`[5]`** |
| `n_positions_written` | **1** |
| neighbours (4, 6) bit-identical | True |

The first row is the load-bearing one: if the hook were not a perfect no-op at zero strength,
every downstream "steering changed the answer" number would be contaminated.

## A false alarm, and why it is worth recording

The first attempt reported positions `[5, 17]` — an apparent extra edit. **The defect was in the
test, not the hook.** `ActivationSteerer.reset()` rewinds the position cursor but does *not*
uninstall the spec, so the "clean" reference pass still carried the previous step's
`alpha=6 / last_prompt` spec and was itself steered at position 17. Position 5 then differed
because the edit was added and 17 because it was removed.

Uninstalling properly with `set_spec(None)` gives exactly `[5]`.

This is the project's own standing lesson — *when a test contradicts a fix, suspect the test* —
and it argues for an API change: `reset()` and "stop steering" being different operations is a
sharp edge that will cut someone again. Either rename it or have it clear the spec.

## Consequence

The composed runner is viable as designed: precompute V1/V2/V3/V4 directions in `nla-mi`, save as
`.npy`, and load them in a `codesteer`-env runner that installs
`artifact.steering.backends.install_steering_hooks` and the copied `ActivationSteerer` on the same
model. The two environments never have to coexist in one interpreter.

**Still to build:** the direction-export script, `compose_run.py` itself (eager model load,
case-pack loop, `java_counterfactual` grading), and the hard validation gate — with `alpha=0` and
attention steering ON, it must reproduce the banked `obf_steer:adversarial_rename` accuracy to
within ±0.02, or B5 is dropped rather than debugged into existence.
