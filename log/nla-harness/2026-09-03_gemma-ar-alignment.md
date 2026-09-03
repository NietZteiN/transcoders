# 2026-09-03 — The Gemma AR is valid, and it points somewhere the activation contrast does not

**Thread:** nla-harness · **Job:** 374534 (gemma12b) · **Status:** gate PASSED, one finding worth its own entry

---

## Why this ran

The released Gemma NLA pair (`kitft/nla-gemma3-12b-L32-{ar,av}`) had been downloaded for days but
was never wired in: `steer_run.py` resolved a **hardcoded path** to `nla/data/checkpoints/ar`,
which is the **Qwen** reconstructor (`Qwen2ForCausalLM`, `hidden_size` 3584). So the situation was
worse than "NLA steering has not run on a permitted host" — any attempt would have loaded a
3584-wide reconstructor against a 3840-wide residual stream. That raises here. **On a same-width
host it would have run silently.**

AR checkpoints are now resolved per host, and the runner cross-checks the checkpoint's own
`nla_meta.yaml` against `--layer`, because an AR reconstructs into exactly one layer's basis.

Constants worth recording, since they are not interchangeable and inheriting the wrong one puts
vectors out of distribution:

| | Qwen2.5-7B | Gemma-3-12B |
|---|---|---|
| layer | 20 / 28 | **32 / 48** |
| d_model | 3584 | **3840** |
| mse_scale | 59.867 | **61.968** |
| AV injection_scale | 150 | **80000** |

The ~500× injection-scale gap is the activation-norm gap between the hosts, and it is exactly the
constant the vendored code warns produces CJK gibberish when wrong.

## The four checks

| check | result | verdict |
|---|---|---|
| dimension | AR output 3840 = sidecar d_model 3840 | PASS |
| scale | AR norm median **56,806** vs real L32 activation norm **74,226** → ratio **0.765** | PASS |
| separation | 0/20 items with `AR(gloss_true) == AR(gloss_decoy)`; V1 norm median 9,169 | PASS |
| alignment | cos(V1, V3) mean **0.0446**, median 0.0535, range [−0.0836, +0.0913] | reported, not gating |

**Scale is the reassuring one.** The AR emits vectors at 0.77× the real activation norm at its own
layer — the same order of magnitude, in the same space. Combined with the dimension check, the
checkpoint is being used correctly, which is not something the Gemma port had established before
today. **Separation matters more than it looks:** had the AR mapped both glosses to the same
vector, V1 would be identically zero and every V1 null in the programme's history would be an
artefact of the instrument rather than a result.

## The finding: detectably non-random, practically orthogonal

cos(V1, V3) = **0.0446**. In 3840 dimensions two independent random vectors have cos with sd
1/√3840 = **0.0161**, so over n = 20 the standard error of the mean is 0.0036 and the observed mean
sits **z = 12.4** above zero. It is not noise.

It is also **0.20% shared variance**. Per-item values span −5.2 to +5.7 chance-sd. In every sense
that matters for steering, **the NLA-derived direction and the model's own clean-minus-decoy
contrast point independent ways.**

## Why this changes how the next result must be read

This was pre-registered as *reported, never gating*, and the reason now has evidence behind it.

- If **V1 moves the task where V3 does not**, the NLA is addressing something real that the plain
  contrastive difference misses — which would be the most important result in the programme.
- If **V1 does nothing**, the reading *"the AR is not addressing this space"* is supported by an
  **independent measurement** rather than inferred from the null itself. That is a different claim
  from *"there is no item-level belief to edit"*, and **distinguishing those two is precisely what
  B4 could never do** — B4 could only observe that V1 failed to beat V3, not whether the two were
  even aimed at the same thing.

A low alignment does **not** invalidate V1. V1 could be a better direction than V3. What it rules
out is the assumption — never stated, but implicit in reading V1 and V3 as comparable arms — that
the two are approximately measuring the same axis.

## Bounds

- n = 20 items, one host, one layer, one gloss construction. V3 here is the leave-one-out mean
  contrastive direction at L32; V1 orthogonal to *that* is not V1 orthogonal to everything.
- The AV was not exercised. `reconstruct()` is AR-only, so V1/V2 need no server — but the full
  AV→AR round trip (**G0 stage B**) remains unrun on Gemma, and this entry is not a substitute
  for it.
- Alignment was measured on `gloss_true − gloss_decoy`, the V1 construction. A different gloss
  would give a different vector.

## Provenance

`nla/src/ar_gate.py`, job 374534, seed 20260724. Artifact
`data/nla/p0/steerv2/gemma12b/ar_gate.json`. AR snapshot
`models--kitft--nla-gemma3-12b-L32-ar/snapshots/3d6901d8…`.

## New questions

- **Is cos(V1, V3) ≈ 0 a Gemma fact or a general one?** The Qwen measurement would say, and cannot
  be made under the model constraint. If a Qwen-era artifact holds both vectors, it could be
  computed retrospectively from banked data — worth checking before assuming it is lost.
- **Does V1 align better with anything else?** The obvious candidates are the attention-based
  measures from Instrument 1 and the answer-line read direction, both of which exist.
