# 2026-09-12 · Steering results — steer ONE layer (L7); the effect is INTERMEDIATE, led by L23H3

**Thread:** nla-harness · **Experiment:** E-S (H-S1…H-S3; H-S4 pending) · **Status:** H-S1/H-S2/H-S3 complete
**Pre-registered:** [`2026-09-11_steering-sweep-prereg.md`](2026-09-11_steering-sweep-prereg.md) — thresholds frozen
before any data. Builds on the 33 live 4B pairs from
[`2026-09-10_multilayer-gate-results.md`](2026-09-10_multilayer-gate-results.md).

## Provenance

| | |
|---|---|
| H-S1 | job **390856**, `nla/src/nla_layer_sweep.py` sha `4f6740575bc6e5ab…`, 6 840 forwards @ 37 ms |
| H-S2/H-S3 | job **390925**, `nla/src/nla_head_sweep.py` + `head_patch.py`, 30 060 forwards @ 30 ms |
| stimuli | the same 60 items / 471 repaired spans as every run in this family; banked gate vectors |
| seed / boot | 20260724 · N_BOOT 10 000 · cluster bootstrap over items |

## H-S1 — **MORE-IS-WORSE**: steering one layer beats steering many

| k | `dG_c3` top-k | depth-spaced control | `swap` ceiling | random null |
|---|---|---|---|---|
| **1** | **+65.75** | +50.26 | 73.15 | −15 |
| 2 | +65.58 | +48.61 | 74.06 | −105 |
| 4 | +62.41 | +54.95 | 68.66 | −243 |
| 8 | +58.63 | +54.52 | 59.00 | −843 |
| 16 | +58.01 | +55.77 | 60.26 | −978 |
| 33 | +54.06 | +54.06 | 58.65 | −2655 |

`k=33 − k=1 = −11.70` nats, CI **[−17.22, −6.51]** (excludes 0) → **MORE-IS-WORSE**, best_k = 1, 95 % of max at k=1.
**Layer CHOICE dominates layer COUNT:** at k=1 the ranked layer (L7) beats the depth-matched control by **+15.5 nats**,
against a −11.7 penalty for using all 33. Split-half rankings agree — `[7,6,8,2,9,5]` and `[7,6,5,9,8,4]` — so the
ranking is stable, not noise-fitted.
**Mechanism: host damage, not channel failure.** The `swap` ceiling (writing the GENUINE clean state) also falls,
73.15 → 58.65, while the fidelity ratio c3/swap stays flat (0.90 → 0.92), and the random null degrades catastrophically
(−15 → −2655). Multi-layer writing degrades the host's capacity to benefit from *any* intervention. This converges with
Phase B's independent `M-NO-GAIN` on the edit arm — two different arms, same conclusion.
**Note for the record:** the 6-item smoke returned **MORE-IS-BETTER, best_k=16** — the opposite verdict. Same code, same
ladder; a 6-item verdict is noise. It was not reported as a result.

## H-S2 — **INTERMEDIATE** localisation; one component carries a third of the effect

Write at L7 (the H-S1 winner); 26 downstream layers × 8 heads + 26 MLPs = **234 components**. `dG_S = +65.75`
CI [+58.73, +72.90]. **Identity checks exact: `max|SELF| = 0.0000`, `mean|ALL gap| = 0.0000`** on all 60 items.

| k | recovered | layer-matched random | advantage (CI) |
|---|---|---|---|
| 1 | **33.1 %** | 5.1 % | +18.37 [+15.53, +21.27] |
| 2 | 48.2 % | 27.7 % | +13.52 [+11.23, +15.85] |
| **4** | **67.3 %** | 32.2 % | +23.10 [+19.97, +26.35] |
| 8 | 84.2 % | 40.6 % | +28.64 [+24.55, +32.85] |
| 16 | 97.7 % | 73.4 % | +15.99 [+13.13, +18.95] |
| 32 | 98.4 % | 85.3 % | +8.58 [+6.29, +10.89] |

Frozen rule: top-16 recovers ≥50 % but its advantage over the layer-matched random null (+15.99) is **below** the
0.30·dG_S = 19.7 required for CONCENTRATED, and 50 % *is* reached (at k=4), so not DISTRIBUTED → **INTERMEDIATE**.
Every k's advantage over the null has a CI excluding zero, so this is selection rather than counting.
Top-k ranking is stable across halves: `[L23H3, L22M, L33H2, L13H6, L30H2, …]` in both.

## H-S3 — sufficiency and necessity are DIFFERENT heads (descriptive, no verdict word)

| by sufficiency (carries it alone) | | by necessity (lost when blinded) | |
|---|---|---|---|
| **L23H3** | +21.75 | **L33M** | +4.673 |
| L22M | +14.75 | L11M | +3.596 |
| L33H2 | +13.97 | L11H2 | +3.481 |
| L13H6 | +12.16 | L14M | +3.412 |
| L30H2 | +10.84 | L12M | +3.137 |

**ρ(sufficiency, necessity) = +0.117** — essentially uncorrelated. `L23H3` dominates sufficiency yet ranks 10th in
necessity (+2.47); the necessity leaders are mostly **MLPs**. The system is redundant: blinding any one component loses
little, while patching the right one alone achieves a third of the effect. "Which component can carry the signal" and
"which the signal cannot do without" are different questions with different answers.

**Global-attention layers carry 2.3× the necessity of local ones: +0.759 (n=36) vs +0.332 (n=198).** This is the
direction the structure predicts — the write lands on **prompt** positions, `G_sum` scores **reply** tokens, so every nat
must cross positions through attention above the write, and sliding-window layers cannot span the distance. Layers by
summed necessity: **L11 (12.88)**, L9 (8.08), L14 (7.08), L23 (6.32), L12 (6.19), L33 (5.85) — and **L11 is a global
layer** (globals are 5/11/17/23/29). Reported without a verdict word, as pre-registered.

## Practical answer to the request

- **How many layers to steer: ONE.** More is measurably worse, and the gain is in picking the layer (L7), not stacking.
- **Which heads matter: `L23H3`** for carrying the effect (33 % alone), the **L11/L33 MLPs and L11H2** for being
  indispensable, with **global-attention layers systematically more load-bearing**.
- **Accuracy (H-S4) still to run**, and its ceiling was declared in advance: the host loses 2 of 60 items to
  obfuscation (0.567 → 0.533) and the flippable census found 6/60 flippable, so the maximum detectable rescue is
  ≈ +0.03 overall / ≈ +0.10 on the flippable subset. **No accuracy claim will be drawn either way.**

## Two harness bugs, both mine, both caught by the smoke

1. `cU = patcher.stop_recording()` — `record()` returns the cache the next forward fills; `stop_recording()` returns
   `None`. Caught in <1 min on a 3-item smoke.
2. `patcher.reset()` only zeroes the write counter; **`set_patches(None)` is what clears the patches**. Leaving the
   previous item's `ALL` patch installed made the next `record()` refuse — `head_patch.py` protecting against recording
   patched activations as clean. Also added the `assert_written()` counter check the prereg promised and I had skipped.
