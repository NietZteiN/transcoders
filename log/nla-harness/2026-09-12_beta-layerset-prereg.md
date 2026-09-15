# 2026-09-12 · PRE-REGISTRATION — can multi-layer NLA steering be made to work? (H-S5…H-S9)

**Thread:** nla-harness · **Experiment:** E-S continuation, Gemma-3-4B-it, 33 live pairs
**Status:** pre-registered, **nothing run**. Every threshold below is frozen; a change after the first
submission is a new dated entry, not an edit.
**Builds on:** [`2026-09-12_steering-sweep-results.md`](2026-09-12_steering-sweep-results.md) (H-S1
`MORE-IS-WORSE`, H-S2 `INTERMEDIATE`), [`2026-09-10_multilayer-gate-results.md`](2026-09-10_multilayer-gate-results.md)
(`M-NO-GAIN`, the per-layer fidelity profile), and the flippable census
[`2026-09-07_flippable-census-results.md`](2026-09-07_flippable-census-results.md) (H-W30).
**User direction (2026-09-12):** *"I want to plan on the 4b running steering on all layers to try to get better
results."*

## Why this is not a re-run of H-S1

H-S1 returned `MORE-IS-WORSE` and the Phase-B gate returned `M-NO-GAIN`, which together read as "multi-layer
steering is a dead end". Re-reading the **banked per-item rows** — `data/nla/ml/gemma4b/gate/gate_rows.jsonl`
and `data/nla/ml/gemma4b/gate/layer_sweep/sweep_rows.jsonl`, **no GPU spent, no new measurement** — shows those
verdicts were narrower than they sound. Four facts, all recomputable from those two files:

1. **The verdict was about the ceiling, not the channel.** From k=1 to k=33 the transport arm `c3` loses
   −11.70 [−17.19, −6.41] but the ceiling `swap` (the raw clean state) loses **more**: −14.50 [−20.31, −8.67].
   The ratio `c3/swap` therefore *rises*, 0.899 → 0.922. Multi-layer replacement damages the host's ability to
   use the written positions at all; it does not corrupt the NLA channel.
2. **For the arm we want to improve, "more is worse" was never established.** `edit` at k=16 minus k=1 is
   **+0.91 [−3.31, +5.31]** — flat. Only k=33 clears (−5.76 [−10.11, −1.26]).
3. **The sweep optimised the wrong objective.** It ranked layers by `dG_c3` (peak L7). Raw `edit` peaks at
   L1/L2 instead — but at L1/L2 the nulls nearly match the edit (`rt` +31.00, `foreign` +31.45 against `edit`
   +35.31), so **~+4 of the +35 is specific to the actual edited content**. Raw nats reward non-specific decoy
   destruction: at L2/L3 even a norm-matched *random* vector scores +16.8/+18.9. The specific component
   `edit − foreign` is **worst at L1 (+1.98)** despite L1 having the second-highest raw `edit`, and peaks at
   **L3 (+9.02 [+5.74, +12.63])**. Its CI excludes 0 at every layer L1–L29.
4. **Multi-layer specificity has never been measured.** `nla_layer_sweep.py:42` is
   `ARMS = ("c3","swap","edit","random")` — **no `foreign`, no `rt`**. Raw `edit` being flat is exactly what you
   expect if the non-specific part saturates (the decoy can only be removed once) while the specific part still
   adds. Nothing in the banked data distinguishes those.

Also dissolved, so it is not chased: the best `edit` cell in the whole sweep, `spaced2 = {L1,L33}` at +36.50, is
**L1 alone** — L33 contributes exactly 0.000 (`swap` = 0.000, causally inert). There is no two-layer synergy there.

## The mechanism this experiment attacks, and the knob that does not exist

`PositionReplacer._hook` (`nla/src/steer.py:250-272`) writes `h[p] ← ‖h[p]‖·unit(v_p)` — a **full replacement** —
and its docstring (`steer.py:227-231`) records that the absence of a coefficient is deliberate: *"there is no
coefficient to sweep, which is the point."* That is right for a single layer, because the AR emits a whole state
rather than a delta. At k layers it means the span position's residual trajectory is **overwritten k times**, so
the position stops computing between writes. That is the mechanical account of fact 1 above, and of the `random`
arm degrading −15 → −2655 as k grows.

This experiment adds the omitted coefficient:

> **h[p] ← ‖h[p]‖ · unit( (1−β)·unit(h[p]) + β·unit(v_p) )**

β = 1 reproduces today's replacement exactly; β = 0 is a value no-op; β ∈ (0,1) lets every layer nudge the
position while the position keeps computing. **β = 1.0 is the default at all 10 existing call sites** (each passes
`(model, layer)` positionally), so no banked result changes.

## Why the specific effect might add — with a number attached

`v_edit = v_rt` whenever a span is not editable, so only editable spans carry a real edit, and **which** spans
are editable differs by layer (pairwise Jaccard 0.63–0.81 among early layers). The union of editable spans over
the SPEC-ranked top-k, from the banked `L{K}_spans.jsonl` sidecars:

| k | 1 | 2 | 4 | 8 | 16 | 29 |
|---|---|---|---|---|---|---|
| union of editable spans (of 471) | 113 | 124 | 143 | 159 | 176 | **185** |

**Pure per-span union-additivity therefore predicts SPEC(k=29)/SPEC(k=1) ≈ 185/113 = 1.64.** That is the
primary prediction — a number, not a direction.

*Clarified before submission (nothing has run):* the 113 above is the editable count of the layer that the
**pooled** SPEC ranking puts first (L3). Under the split-half discipline the two halves need not agree on the
top layer — half 0 ranks L3 first (113 editable), half 1 ranks L7 first (101) — so the run-time union ratio can
land near 1.64 or nearer 1.83. **The decision bar stays fixed at 1.64**; the harness additionally reports the
union ratio of the sets it actually selected, as the mechanistic reference the bar came from. The bar is not
recomputed from the run.

## Setup (frozen)

Host: Gemma-3-4B-it text ckpt, 33 live NLA pairs (L0 PAIR-DEAD, excluded). Stimuli: the same **60 items / 471
repaired spans** as every run in this family, selected by `nla_ml_gate.select_pairs` reused verbatim. Vectors
**already banked** at `data/nla/ml/gemma4b/gate/vectors/L{K}.npz` (34 layers, npz keys `h0, h1b, c3, edit,
foreign, rt, self_all`) — **no AV/AR is loaded and no re-extraction happens**; host only, one GPU.
Readout: summed `G_sum` (teacher-forced logp of the L0 reply under the L1b prompt). Seed **20260724**,
N_BOOT **10 000**, cluster bootstrap over items, paired per item. `--deterministic` stays OFF.
Config `nla/configs/nla_beta_sweep.yaml`; code `nla/src/nla_beta_sweep.py` + the β parameter in
`nla/src/steer.py`. Arms: `edit, foreign, rt, c3, swap, random, self` (all seven, at every cell — this is the
gap H-S1 left).

**Primary contrast.** `SPEC(S, β) = dG_edit(S, β) − dG_foreign(S, β)`, paired per item. `foreign` (the decoy
replaced by a *different item's* true term) is the null that holds the write's shape, position count and
norm fixed and varies only the content, so it is the one that isolates content-specificity.

**Selection discipline.** Split-half by `crc32(snippet_id) & 1`; any rank-selected layer set is ranked on the
**other** half and evaluated on this one. Layer rankings (by SPEC and by `dG_c3`) are computed from the **banked**
β=1 single-layer gate rows rather than re-measured, and the harness re-verifies 3 layers per half against a fresh
forward to ≤ 0.05 nats (identity gate 1).

**Named layer sets** (fixed before the run; none selected on today's outcome):
`FAITHFUL` = {2,3,6,7,8,9,11,12,13,14,15,16} (the fidelity CI-lower ≥ 0.80 set frozen 2026-09-10) ·
`BAND` = {2…13} (contiguous) · `SPACED` = 12 depth-spaced live layers · `SPEC-TOP12` (split-half) ·
`COVERAGE` = greedy maximisation of the editable-span union, computed from span **metadata only**, never from
any dG · `LIVE-NO-TAIL` = 1–29 (drops L30–L33, where `swap` ≤ 0.11 and the ratio is ≤ 0.47 or undefined) ·
`ALL-LIVE` = 1–33, so the user's literal "all layers" is answered · plus the SPEC-ranked top-k ladder
k ∈ {1,2,4,8,16,29}. β ladder: **{0.1, 0.2, 0.35, 0.5, 0.75, 1.0}**.

## Hypotheses and decision rules (frozen, declared before any data)

- **H-S5 (primary) — does the specific effect add across layers?** SPEC-ranked top-k, k ∈ {1,2,4,8,16,29}, each
  at its best β (β chosen on the other half, so β is selected and evaluated on disjoint items).
  **`SPEC-ADDITIVE`** if max_k SPEC / SPEC(k=1) **≥ 1.64** with the paired CI of (best k − k=1) excluding 0 ·
  **`SPEC-SATURATING`** if that ratio is **≥ 1.25 but < 1.64** with the paired CI excluding 0 ·
  **`SPEC-NON-ADDITIVE`** if no k > 1 beats k=1 with a CI excluding 0.
  *Prediction stated now:* **SATURATING** — the W family found span-level accumulation with saturation
  (H-W15 `P_1/P_all` 0.225; H-W39 `W39-SMOOTH`).
- **H-S6 — does β repair the multi-layer ceiling?** At `LIVE-NO-TAIL` (k=29) across the β ladder, on the `swap`
  arm. **`CEILING-REPAIRED`** if `dG_swap` at the best β reaches **≥ 95 % of the k=1/β=1 ceiling (+73.15)** with
  its CI excluding the k=29/β=1 value · **`NOT-REPAIRED`** otherwise. Secondary, descriptive: does the argmax β
  per k follow a dose law β ≈ β₁/k? The full β × k surface for SPEC is reported as a table.
- **H-S7 — selection objective.** SPEC-ranked top-1 vs `c3`-ranked top-1 (both split-half), scored on SPEC.
  **`SELECTION-MATTERS`** if the paired difference is **≥ +3.0 nats with CI excluding 0**; otherwise
  **`SELECTION-IMMATERIAL`**. (The banked estimate of the analogous contrast on *raw* `edit` is
  +3.25 [−2.03, +8.47] — suggestive and underpowered, which is why it is being tested on the specific contrast.)
- **H-S8 — layer-set shape** at matched k = 12: `FAITHFUL` vs `BAND` vs `SPACED` vs `SPEC-TOP12` vs `COVERAGE`.
  **Descriptive, no verdict word**; the best set is reported with its CI and carried forward to H-S9's gate.
  Pre-registered expectation: `COVERAGE` ≥ `SPEC-TOP12` if H-S5's union mechanism is what drives additivity.
- **H-S9 (gated) — accuracy.** Runs **only** if the grid's best configuration beats the best single layer's SPEC
  by **≥ +3.0 nats with CI excluding 0**. Then: generate answers with **n = 8 samples/item** (per-item pass-rate
  rather than binary greedy — the only cheap way to add power here), arms {edit, foreign, rt, unsteered} plus the
  mandatory **prompting baseline** ("the identifiers may be misleading; reason about the code's behaviour") per
  CLAUDE.md §4. Grading: `steer_run.graded` / `ANSWER_RE`, `MAX_NEW_GEN = 1100`, unchanged so numbers stay
  comparable with the banked corpus. Wilson CIs on all 60 items and on the 7 flippable.
  **Pre-declared bound, so a null cannot be read either way:** this host scores `acc_l0` 0.567 → `acc_l1b` 0.533,
  i.e. **2/60 net headroom and 7 flippable**, so the maximum detectable rescue is ≈ **+0.03 overall / ≈ +0.10 on
  the flippable subset**. **No accuracy conclusion will be drawn in either direction.** The stage exists to check
  steering does not *harm* accuracy and to size a purpose-built corpus
  (`docs/nla_flippable_corpus_scoping.md`, written this round). H-W30 already established a better trap host is
  not available: Llama-3.1-8B cleared the letter of the power gate at 9 flippable and the pass was judged not real.

**Identity gates — any failure ⇒ exit 3, verdict `S-HARNESS-FAULT`, nothing reportable:**
1. **β = 1.0 nested identity.** Single-layer β=1 cells reproduce the banked `gate_rows.jsonl` values to
   **≤ 0.05 nats** (checked on 3 layers per half × the `edit`/`foreign` arms), and the `ALL-LIVE`/β=1 `c3` cell
   reproduces the banked **+54.06**.
2. **β = 0.0 is a *value* no-op, not a *position* no-op.** |dG| ≤ 0.05 on every arm **while reporting the same
   `n_positions_written` as β = 1**. This distinction is load-bearing: `arm_guard.paired` refuses a contrast whose
   two arms wrote different position counts, and three call sites raise on a count mismatch
   (`nla_ml_gate.py:376`, `nla_layer_sweep.py:115`, `nla_heads.py:293`). A β that changed *which* positions are
   written would break all of them silently.
3. **`self` arm = 0.000** at every cell, as in every run of this family (tol 1.0 as banked, but it has been
   exactly 0.000 since the capture-shape fix of 2026-09-09).

## Known limitations, stated up front

1. **SPEC is diluted.** Because `v_edit = v_rt` on non-editable spans, only 81–116 of the 471 spans per layer
   carry a real edit. The **pooled** figure is primary (comparable with every banked run); an
   **editable-restricted** figure is reported as a sharper secondary. Neither is re-defined after seeing the data.
2. **`foreign` is a content null, not a behaviour null.** It holds shape/position/norm fixed and varies content,
   which is what H-S5 needs; it does not control for the *fluency* of the substituted term.
3. **β is a new degree of freedom**, so H-S5's per-k best β is selected on the opposite half. The β × k surface
   is reported in full precisely so the selection is auditable rather than summarised.
4. **Accuracy cannot be settled here** (limitation, not a result) — see H-S9's pre-declared bound.
5. **This is one host at one corpus.** Nothing here tests whether the β story transfers to the 12B pair; the 12B
   has one trained layer (L32) and H-C2 is unadjudicated.

## Cost

Grid ≈ **23 k forwards** at the measured **37 ms** ⇒ **~14 min compute**, dominated by model load; one GPU
(`--partition=h200,h100 --gres=gpu:1`, host only ~9 GB bf16), ≤ 1 h wall including queue. Gated accuracy stage
≈ **2 h** (60 items × 4 arms × 8 samples × 1100 tokens). Corpus scoping doc: no GPU. **Total ≤ 3 GPU-h** —
against 44.6 GPU-h for the 4B pairs themselves, this is a rounding error, which is the argument for running it.
