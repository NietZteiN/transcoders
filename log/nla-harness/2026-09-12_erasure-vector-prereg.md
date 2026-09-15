# 2026-09-12 · PRE-REGISTRATION — the cross-item decoy-erasure vector, no autoencoder (H-E1 / H-E2 / H-E3)

**Thread:** nla-harness · **Experiment:** `E_erasure_vector`, Gemma-3-4B-it · **Status:** pre-registered,
**held** until job 391968 (accuracy) reports; nothing run.

## The question

The NLA `edit` at L7 recovers **+32.06** nats of a **+73.15** `swap` ceiling (banked
[`gate_rows.jsonl`](../../data/nla/ml/gemma4b/gate/gate_rows.jsonl)), and its content-specific part is only
`edit − foreign` = **+8.18**. The activation-space tier decomposition says the clean state's effect is
**≈ 51 % meaning / 49 % decoy removal / 1 % structure**
([`2026-09-07_repaired-rerun-results.md`](2026-09-07_repaired-rerun-results.md)). Both point the same way: most
of what the NLA edit delivers may be *generic* decoy erasure, which should not need an autoencoder at all.

So: build the erasure direction **without the NLA** — the leave-one-item-out mean of the clean-minus-decoy
displacement over the other 59 items' spans — write it at the same span positions, and ask whether it does what
the NLA edit does. This is the task-vector control (V3 in the B4 ladder) rebuilt at the **span positions and
early layers where the specific effect actually lives**, instead of one token at L20 where V3 was built before.
It is inference-only: forward passes on banked vectors, the cost class of attention reallocation.

Banked coherence check (no forwards): at L7 the per-span displacements `Δ_s = h0_s − h1b_s` have
**cos(Δ_s, mean Δ) = 0.258 on average, positive on 96.8 % of 471 spans**; ‖mean Δ‖ = 315 against a mean ‖Δ_s‖
of 1 300 and ‖h1b‖ of 5 427. A shared direction exists; the question is what it is worth.

## Arms (per span `s` of item `i`, layer `K`)

`E_K^{(−i)}` = mean over all spans **not in item i** of `Δ_s`; `Ê` its unit vector. The target written is

    erase_α :  v_s = h1b_s + α · ‖Δ_s‖ · Ê_K^{(−i)}        α ∈ {0.5, 1.0, 2.0}; α = 1.0 is PRIMARY

i.e. move the decoy state as far as *this* span's own displacement, but along the **shared** direction. Controls:
`erase_rand` (same formula with a seeded random unit direction, norm-matched — a perturbation of equal size that
carries no shared content) and `erase_own` (`v_s = h1b_s + ‖Δ_s‖·unit(Δ_s) = h0_s` exactly — reproduces `swap`
and is an identity gate). Re-run for pairing: `edit`, `foreign`, `swap`, `self`. Two layer sets, as in every
recent run: **L7 single β = 1.0** (primary — the banked comparison layer) and **band L2–13 β = 0.35** (secondary).
Readout `G_sum` (teacher-forced logp of the banked L0 reply under the L1b prompt), 60 items, 471 spans,
cluster bootstrap N_BOOT 10 000, seed 20260724, paired per item.

## Identity gates (exit 3, nothing reportable)

`erase_own` at L7 β=1 must equal the banked `dG_S_swap_L7` to ≤ 0.05 nats on every item; `edit`/`foreign`/`swap`
at L7 β=1 must reproduce the banked rows to ≤ 0.05; `self` = 0.000 (≤ 1.0). Position counts asserted per forward.

## Frozen rules (all at L7 β = 1, paired per item)

- **H-E3 (gate on the others) — is the shared direction content or just a push?**
  `erase_1.0 − erase_rand` ≥ **+3.0** with CI excluding 0 → **`ERASURE-DIRECTIONAL`**; else **`ERASURE-NOISE`**,
  and H-E1/H-E2 are void (any match to `edit` would be perturbation size, not content).
- **H-E1 — does generic erasure do what the NLA edit does?** `erase_1.0 − edit`:
  CI lower > 0 → **`ERASURE-BEATS-EDIT`**; CI upper < 0 → **`EDIT-CARRIES-CONTENT`** (the autoencoder delivers
  something beyond the shared direction); CI spans 0 → **`ERASURE-MATCHES-EDIT`**, reported **with the CI width**
  (an interval spanning 0 at n = 60 is "not distinguished", not "equal").
- **H-E2 — is erasure capped at the removal share?** The banked removal share is 0.493
  (`T_L1_sub / T_L0_all` = 20.47 / 41.52); bar **0.55** (share + slack). Paired `erase_1.0 − 0.55·swap`:
  CI upper < 0 → **`ERASURE-WITHIN-SHARE`**; CI lower > 0 → **`ERASURE-EXCEEDS-SHARE`** (the shared direction
  carries meaning too); else **`SHARE-UNRESOLVED`**.

Descriptive, no rule: the α dose curve; the band-set replicas of every arm; per-item Spearman between the
`erase_1.0` gain and `cos(Δ_item, E^{(−i)})`; `erase_α − foreign` (erasure vs. a single other item's true term).

**Predictions, stated now:** `ERASURE-DIRECTIONAL`; `ERASURE-MATCHES-EDIT` (edit is 0.44 of swap, the removal
share is 0.49 — they should land together); `ERASURE-WITHIN-SHARE`. If instead `EDIT-CARRIES-CONTENT`, the NLA
is buying real item content at L7 and the "mostly erasure" reading of the edit is wrong.

## Setup (frozen)

Code `nla/src/nla_erasure.py`; vectors `data/nla/ml/gemma4b/gate/vectors/L{K}.npz` for K ∈ {2…13}; traces
`data/nla/p0/trace_llr/gemma4b/traces.jsonl`; `PositionReplacer` with β as banked. Output
`data/nla/ml/gemma4b/gate/erasure/{erasure_rows.jsonl,erasure_stats.json}`. `--deterministic` OFF. Cost: ~13
arms × 2 sets × 60 items ≈ 1 600 forwards at ~37 ms ≈ **1 min compute**; the job is model load. One GPU.

## Why held

Step 1 of the post-accuracy plan agreed 2026-09-12. If 391968 shows `c3_band` moving accuracy, this run adds the
accuracy readout for `erase_1.0` before submission (a new dated entry, not an edit of this one).
