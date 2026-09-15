# 2026-09-10 · Phase B results — multi-layer NLA gate on Gemma-3-4B-it: **M-NO-GAIN** in every set

**Thread:** nla-harness · **Experiment:** Phase B (E-ML) · **Status:** complete, reportable (`identity_passes: true`)
**Pre-registration:** [`2026-09-09_multilayer-train-prereg.md`](2026-09-09_multilayer-train-prereg.md) ·
harness frozen in [`2026-09-09_multilayer-gate-harness.md`](2026-09-09_multilayer-gate-harness.md) ·
capture fix in [`2026-09-09_gate-smoke-self-exact.md`](2026-09-09_gate-smoke-self-exact.md) ·
quota incident in [`2026-09-09_quota-exhaustion-l18.md`](2026-09-09_quota-exhaustion-l18.md)

## Provenance

| | |
|---|---|
| host | `google/gemma-3-4b-it`, text-only ckpt `data/nla/ml/gemma4b/host_text` (34 layers, d 2560) |
| gate script | `nla/src/nla_ml_gate.py` sha256 **1f7a9d3191a9ce328d5789abd3e451090c43759f3b72b99cb673d2113079c419** |
| gate config | `nla/configs/nla_ml_gate.yaml` sha256 `c7c1d9e3ddc1ddde7ffee4dedf3772641637fdcef650a4895fbf4ea3e0e4baad` |
| trainer | `nla_train.py` b924ff21… · `nla_ml.yaml` b31f5e27… (inherited via `nla_train.provenance`) |
| jobs | train `384655` + `388490` (L18–L20 retry) · vectors `388771` (34/34 COMPLETED) · score `388772` on g-08-08 |
| stimuli | `data/nla/p0/trace_llr/gemma4b/traces.jsonl`, 60 items, **471 spans** (identical to the 12B W-family corpus) |
| seed | 20260724 · N_BOOT 10 000 · 15 240 forwards @ 36.1 ms · score wall 9.2 min |

## Liveness (frozen rules (a)–(d)) — reported FIRST, as pre-registered

**33/34 live. L0 is the only PAIR-DEAD, on rule (b) alone** (`holdout_fve` 0.035 < 0.20). Rules (a), (c), (d) pass at
**every** layer including L0: 0/96 no-tag and 0.0 CJK everywhere. **Both pre-registered sets are ALL LIVE** — no
quota-forced scope change was needed.

Depth profiles: AV gap +0.082 → **+0.186 peak L10** → +0.11 floor (L27–L33). AR fve 0.035 → **0.415 peak L6** →
**trough 0.304 (L13–L15)** → ~0.37 (L24–L28) → 0.266 (L33). Rule-(d) margin mirrors fve: +0.097 (L1) → **+0.0010
(L13–L15)** → +0.013 (L27). Per-read paired *t* for rule (d) is **flat with depth** (10.9–18.8, 86–94 of 96 reads
positive), so the shrinking margin is scale compression, not a weakening pair.
Global-attention layers (5/11/17/23/29) mean fve 0.363 vs 0.351 local — **+0.012 vs sd 0.040, no signal** (descriptive).

## Results

### Gate 0 — is the NLA channel real at 4B? **YES, in all three sets.**

| set | `M_c3 / G_prompt_swap` | CI95 | passes ≥0.25 |
|---|---|---|---|
| primary {8,22,24} | **0.349** | [0.299, 0.404] | ✓ |
| depth {8,17,26} | **0.358** | [0.308, 0.415] | ✓ |
| all (L1–L33) | **0.330** | [0.274, 0.393] | ✓ |

The smoke's `gate0 0.197 FAIL` was an artifact of its dead 40-doc AR, as flagged at the time.

### H-M1 / H-M2 / specificity — **M-NO-GAIN in every set**

| set | H-M1 `M_edit − max_ℓ S_edit_ℓ` | H-M2 `M_edit − M_random` (thr 0.30·M_c3) | spec `M_edit − M_foreign` | SELF | verdict |
|---|---|---|---|---|---|
| primary | **−7.05** [−11.63, −2.42] ✗ | +171.3 (thr +17.1) ✓ | +4.90 [3.10, 6.80] ✓ | 0.000 | **M-NO-GAIN** |
| depth | **−4.66** [−8.91, −0.22] ✗ | +894.6 (thr +17.6) ✓ | +6.13 [3.91, 8.47] ✓ | 0.000 | **M-NO-GAIN** |
| all | **−11.24** [−15.66, −6.92] ✗ | +2681.3 (thr +16.2) ✓ | +5.86 [2.83, 9.02] ✓ | 0.000 | **M-NO-GAIN** |

`best_single_edit_layer = L2` in all three sets. **H-M1 fails with the CI entirely below zero, and the deficit grows
with the number of layers written** (−4.66 at 3 layers → −11.24 at 33). Writing the edit at many layers is *worse* than
writing it at one well-chosen early layer.

**Caveat on H-M2 — the rule gets EASIER with more layers.** `M_random` is catastrophic at scale (per-layer
`random` means: −15 at L7, −472 at L11, −714 at L15), so `M_edit − M_random` inflates from +171 (3 layers) to +2681
(33 layers) without the edit improving. H-M2 as frozen is therefore not a strong non-genericity test at high layer
counts; **specificity (edit vs foreign, +4.9 to +6.1 nats, CI excluding 0) is the honest one** and passes on its own terms.

### H-M3 — verdict of record **INDETERMINATE**; a harness nan made REFUTE unreachable

Frozen output: `frac_layers_ci_lower_ge_min` **0.364** (12/33; needs ≥0.80 → no CONFIRM), `median_ratio` **nan** →
INDETERMINATE. **Cause of the nan: L33 alone**, where `c3` and `swap` are both **exactly 0.00** → 0/0. One nan poisons
`statistics.median`, and `nan < 0.80` is False, so the REFUTE branch could never fire.
**Post-hoc sensitivity analysis (labelled as such; does NOT replace the verdict):** over the 32 layers where the ratio
is *defined*, median = **0.790**, which is < 0.80 and would yield **REFUTE** under the same frozen threshold. Excluding
an undefined 0/0 is arguably the only coherent treatment, but the decision to exclude was made after seeing the data,
so it does not overwrite a pre-registered outcome. Layers clearing CI-lower ≥ 0.80: {2,3,6,7,8,9,11,12,13,14,15,16}.

### The fidelity profile is the substantive finding: the channel is most faithful EARLY

| K | 2 | 3 | 7 | 10 | 14 | 17 | 20 | 22 | 24 | 26 | 29 | 30–32 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `S_c3/S_swap` | **0.93** | 0.91 | 0.90 | 0.85 | 0.91 | 0.82 | 0.68 | 0.68 | 0.54 | 0.51 | 0.35 | ~0/undef | undef |

`S_swap` peaks at **L7** (+73.15) and `S_c3` at L7 (+65.75). Three independent measures converge on the early layers:
(i) fidelity ratio max at L2–L7; (ii) `best_single_edit_layer = L2`; (iii) the geometric clean-vs-decoy separation
`cos(h0,h1b)` is largest early — 0.680 (L0) · 0.710 (L1) · 0.799 (L2/L3) · 0.958 (L5) · 0.996 (L10–L15), recovering to
0.951 (L26) · 0.867 (L33). **The pre-registered sets {8,22,24} and {8,17,26} sit where separation is ≤1 %**; those depths
were chosen from the 12B host (L32/48) before any of this geometry was visible on a 4B host.

### **Four of the 33 live layers are causally inert for this readout** (structural, not a pair defect)

`S_swap` collapses at the top: +23.96 (L29), −3.09 (L30), −2.91 (L31), +0.11 (L32), **+0.000 (L33)**. The write lands on
**prompt** positions and `G_sum` scores **reply** tokens, so every nat must cross positions through at least one
attention layer **above** the write. At L33 there is none — the effect is necessarily and exactly zero, which it is, to
all printed digits. This is a **positive structural validation of the harness** (the write goes where we think, the
readout is genuinely downstream) and simultaneously a **gap in the liveness rules**: (a)–(d) test AV/AR quality and say
nothing about causal reach, so they certified four layers that cannot move the readout. Same class of problem as the
`editable` collapse: `n_editable` is 26 (L33) and 30 (L32) against a median of 97, and since `v_edit = v_rt` when a span
is not editable, **the edit arm is ~94 % identical to the rt arm at L32/L33**.

### Cross-model comparison — our 4B pairs are much weaker than the released 12B pair

The released `kitft/nla-gemma3-12b-L32` pair gives **98.3 %** C3pure fidelity at L32/48 (relative depth 0.67, W-family).
Our 4B pair at the equivalent relative depth (L22/34 = 0.65) gives **0.68**. Budget is the obvious explanation and must
be stated as a limitation: 86 k AV rows / 1 epoch / no RL here, versus the vendored recipe's 100 k docs × 10 positions
with RL. It is consistent with the read-content spot-check (below).

### Read-content spot-check (descriptive; computed after seeing data, labelled as such)

96 held-out reads at L0/L2/L8/L11 are fluent and **structurally** accurate (they nail the final-token fact and the
discourse form) but **confabulate the topic**: a TVTropes page on *From Russia with Love* is read as "the inner teacher"
(L0) / "One Punch Man" (L8) / "Star Trek fan fiction" (L11). Content-word overlap against own source vs 10 others:
lift +0.016 (L0) · +0.037 (L2) · +0.047 (L8) · +0.057 (L11) vs **gold explanation +0.153** — i.e. ~37 % of the gold's
specificity at L11. **`cos_cycle` ≈ 0.999 must not be read as "the read is right"**; it is dominated by the shared mean
direction (`cos_other` also 0.99). This is the CLAUDE.md §4 auto-interp interpretability illusion in plain sight.

## Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| **Gate 0** (channel real) | **PASS** in all three sets (0.330–0.358, CIs exclude 0 and clear 0.25) |
| **H-M1** (multi-layer edit > best single layer) | **✗ REFUTED** — negative in all sets, CI entirely below 0, deficit grows with layer count |
| **H-M2** (edit beats random by ≥0.30·M_c3) | **✓ PASSED** in all sets — but the comparator inflates with layer count; treat specificity as the real test |
| **specificity** (edit > foreign) | **✓ PASSED**, +4.9 to +6.1 nats, CIs exclude 0 |
| **SELF identity** | **✓ 0.000 exactly** in every set and every `S_self_L{K}` — the same-shape capture fix holds |
| **H-M3** (fidelity ≥0.80 at ≥80 % of live layers) | **INDETERMINATE** (verdict of record). Defined-ratio median 0.790 → REFUTE as a labelled sensitivity analysis |
| **H-M4** (decoy-removal vs truth-transport depth profile) | descriptive, reported above; `edit > rt` at **every** layer (e.g. L2 +37.5 vs +30.8; L7 +32.1 vs +23.3) |

**Answer to the question Phase B was built to settle:** the one-layer edit bottleneck found in the W family is **not a
one-layer artifact**. Given 33 live layers to write into, the edit still does not beat the best single layer — it does
strictly worse, monotonically worse the more layers are used. The NLA channel itself is real (Gate 0) and content-specific
(specificity), but *stacking write sites does not buy edit fidelity*.

## What I got wrong today (recorded so it is not silently dropped)

1. **Debug pass used `df` (101 T free) instead of `mfsgetquota`** and declared disk fine ~2 h before the 1.1 TB quota
   killed L18/19/20 mid-save. Rule adopted: check `mfsgetquota <project root>` before any run writing tens of GB.
2. **Forecast that AR fve would cross the 0.20 floor at L22** from a linear fit on L6→L13 (−0.0138/layer), with a
   predicted *steepening* past L17 on spectral grounds. Measured: L16 0.322, L17 0.336, **L22 0.359** — rising.
   An 8-point fit extended 9 layers was never evidence and should have been labelled a guess.
3. **Claimed FVE is "not comparable across layers" because the 0.20 floor meets targets of very different difficulty.**
   Measured: fve is flat (~0.37) across a >3× change in participation ratio (1.3 → 8.8), and **Spearman ρ = +0.030**
   (n=10 interior layers; −0.315 including the two endpoint blocks, which alone drive the sign). No monotone relation;
   the L13–L15 co-trough of fve/margin/PR is a coincidence of depth. The PR table stands as a measurement; every
   inference I drew from it does not.
4. Two premature alarms from reading truncated command output (a "7× slowdown" that was AV-vs-AR confusion, and
   "L20/L27–L33 vanished" that was a `head`-truncated `squeue`).

## New questions / next steps

1. **Re-run the gate with an early-layer set** (e.g. {2,3,7}), pre-registered before running. Three independent measures
   now point there, and every pre-registered set sits in the ≤1 %-separation zone. This is a *new* hypothesis needing its
   own prereg — not a re-scoring of this one.
2. **Add a causal-reach liveness rule** (e.g. `S_swap_ℓ` CI excluding 0) so structurally inert layers are excluded before
   scoring rather than diluting joint writes. Pre-register before applying.
3. **Fix the H-M3 nan** (skip undefined 0/0 ratios; require a minimum `S_swap_ℓ`) and state the fix in a dated entry
   before any re-run, so the threshold is never seen to move after data.
4. **The H-M2 comparator needs re-specification** for multi-layer sets — a random-vector null that destroys the model is
   not a meaningful floor. Candidate: per-layer-count-matched random null.
5. **Budget question:** would a full-recipe AV/AR (100 k × 10 positions, RL) close the 0.68 → 0.98 fidelity gap against
   the released 12B pair at equal relative depth? That is the cleanest test of whether "early is better" is a property of
   the network or of our training budget.
6. Still owed from before Phase B: H-W39 joint stage re-run, H-W35d, folding H-W39 into RESULTS.md.
