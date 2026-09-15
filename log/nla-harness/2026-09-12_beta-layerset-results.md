# 2026-09-12 · RESULTS — β fixes multi-layer steering; the SPECIFIC effect still does not stack (H-S5…H-S9)

**Thread:** nla-harness · **Experiment:** E-S, `S_beta_layerset` on Gemma-3-4B-it, 33 live pairs
**Status:** H-S5/H-S6/H-S7 resolved · H-S8 descriptive · **H-S9 gate did NOT fire, accuracy stage not run**
Pre-registration: [`2026-09-12_beta-layerset-prereg.md`](2026-09-12_beta-layerset-prereg.md) — rules applied unchanged.

### Target Date: 2026-09-12 (β × layer-set grid; the all-layer question settled)

- **Hypotheses / what we're testing:** frozen in the prereg. **H-S5** does the content-specific effect
  `edit − foreign` ADD across layers (`SPEC-ADDITIVE` ≥ 1.64 = the editable-union ratio · `SPEC-SATURATING`
  ≥ 1.25 · else `SPEC-NON-ADDITIVE`, all requiring the paired CI to exclude 0). **H-S6** does β repair the
  ceiling multi-layer replacement breaks (`CEILING-REPAIRED` if `swap` at the best β ≥ 0.95 × 73.15 = +69.49
  with CI excluding the β=1 value). **H-S7** was H-S1's selection objective wrong (`SELECTION-MATTERS` if
  ≥ +3.0 nats, CI excluding 0). **H-S8** layer-set shape, descriptive. **H-S9** accuracy, gated at +3.0 nats.

- **Setup:** job **391301** (`nla_bsweep`, h200, node `g-08-13`, 1 GPU, 2026-09-12T10:14:39 → 10:31:54,
  **elapsed 00:17:15 = 0.29 GPU-h**, COMPLETED rc=0). `nla/scripts/nla_beta_sweep.sh` sha `f24a300dc3965eff…`
  → `nla/src/nla_beta_sweep.py` sha **`04644656b1f33d69…`**, `nla/src/steer.py` sha **`fdf3151c80f5a500…`**
  (the β parameter), config `nla/configs/nla_beta_sweep.yaml` sha **`b05e47819d7d2a36…`**, tests
  `nla/tests/test_beta_sweep.py` sha `412995537655d4c2…`. git HEAD `3c52f26`. Seed **20260724**, N_BOOT
  **10 000**, cluster bootstrap over items, paired per item. Host only (no AV/AR); banked vectors
  `data/nla/ml/gemma4b/gate/vectors/L{K}.npz`; same **60 items / 471 repaired spans** via
  `nla_ml_gate.select_pairs`; traces `data/nla/p0/trace_llr/gemma4b/traces.jsonl`.
  **26 040 forwards at 33.8 ms.** Rankings taken from the banked `gate/gate_rows.jsonl` and re-verified by
  fresh forwards (identity gate 1). Outputs `data/nla/ml/gemma4b/gate/beta_sweep/{beta_rows.jsonl,
  beta_sweep_stats.json}`; log `log/slurm/391301_nla_bsweep.out`.
  β ladder {0.1, 0.2, 0.35, 0.5, 0.75, 1.0} × k ladder {1, 2, 4, 8, 16, 29} + 7 named sets, 7 arms.

- **Results:**

  **Identity gates — all EXACT.** `beta1_vs_banked` max_abs_diff **0.0** on all 12 checks (L2/L3/L4/L7/L9/L13
  × edit/foreign): β=1.0 reproduces the banked gate rows *bit-exactly*, so nothing previously banked moved.
  `beta0` max |dG| **0.0** on all five core arms while still writing every position. `self` **0.0** at all four
  cells. `identity_passes: true`, `reportable: true`.

  **The ceiling surface (`swap`, raw clean state) — the headline.** Rows = k (SPEC-ranked top-k, split-half),
  columns = β:

  | k \ β | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
  |---|---|---|---|---|---|---|
  | 1 | 4.13 | 11.15 | 31.21 | 59.17 | **70.04** | 69.11 |
  | 2 | 8.99 | 29.21 | 60.98 | 71.74 | **71.99** | 70.64 |
  | 4 | 24.43 | 58.88 | 73.26 | **74.17** | 67.22 | 64.31 |
  | 8 | 56.65 | 71.90 | **76.34** | 72.62 | 67.09 | 63.02 |
  | 16 | 70.26 | **77.04** | 70.92 | 67.15 | 62.31 | 60.34 |
  | 29 | 72.89 | **75.35** | 66.14 | 63.26 | 60.64 | 58.72 |

  A clean ridge: argmax β falls monotonically with k (0.75 · 0.75 · 0.5 · 0.35 · 0.2 · 0.2) and the peak
  ceiling *rises* to **+77.04 at k=16/β=0.2** — above the best single-layer cell (+70.04) and above the frozen
  banked k=1/β=1 reference (+73.15).

  **The SPEC surface (`edit − foreign`):** max **+9.12 at k=16/β=0.35**; best single-layer cell **+7.20**
  (k=1/β=0.75); k=1/β=1.0 gives +6.05.

  | k \ β | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
  |---|---|---|---|---|---|---|
  | 1 | 0.21 | 0.88 | 1.49 | 3.19 | **7.20** | 6.05 |
  | 8 | 3.26 | 6.66 | **8.30** | 8.11 | 6.87 | 7.29 |
  | 16 | 6.82 | 8.23 | **9.12** | 8.21 | 6.64 | 5.31 |
  | 29 | 6.68 | 7.67 | **8.01** | 7.93 | 6.95 | 5.68 |

  **What all 33 layers now delivers** (the user's literal request), β=0.35 vs the banked β=1:

  | arm | `all_live` β=0.35 | `all_live` β=1.0 (banked form) | banked best single layer, β=1 |
  |---|---|---|---|
  | `edit` | **+35.35** | +26.30 | +32.06 |
  | `c3` | **+66.44** | +54.06 | +65.75 |
  | `swap` | +66.10 | +58.65 | +73.15 |
  | `SPEC` | +8.12 | +5.86 | +7.20 (β=0.75) / +6.05 (β=1) |

  **H-S8 — layer sets at matched k=12, each at its own best β** (all at β=0.35 except `spaced`, β=0.5):

  | set | layers | editable union | SPEC |
  |---|---|---|---|
  | **`band`** | 2–13 contiguous | 171 | **+9.85** [+7.15, +12.77] |
  | `coverage` | 2,3,4,6,7,8,10,11,15,16,18,22 | **183** | +8.92 [+6.22, +11.82] |
  | `spec_top12` | split-half | 172 | +8.82 [+5.65, +12.17] |
  | `faithful` | frozen 2026-09-10 | 167 | +8.67 [+5.63, +11.94] |
  | `all_live` (33) | 1–33 | 185 | +8.12 [+5.32, +11.18] |
  | `live_no_tail` (29) | 1–29 | 185 | +8.01 [+5.24, +11.06] |
  | `spaced` | 1,4,…,33 | 166 | +7.63 [+5.35, +10.16] |

- **What worked / hypothesis verdict:**
  - **H-S6 → `CEILING-REPAIRED` ✓.** At k=29, `swap` runs **+58.72 (β=1) → +75.35 (β=0.2)**, clearing the
    required +69.49, with CI [+67.99, +82.71] excluding the β=1 value. **The mechanism proposed in the prereg
    is confirmed:** H-S1's `MORE-IS-WORSE` was an artifact of the *write form*, not of layer count. A partial
    write at 16 layers delivers **more** of the clean state (+77.04) than any single full replacement does.
  - **H-S5 → `SPEC-NON-ADDITIVE`** by the frozen rule. Best k=16, ratio **1.266** against the 1.64 bar, paired
    **+1.92 [−0.567, +4.438]** — the CI includes 0. Reported honestly: the *point estimate* sits inside the
    `SPEC-SATURATING` band (≥1.25), and it is the **CI that fails**, so this is "not established at n=60",
    not "shown absent". The run-time union ratio for the selected sets was 1.566.
  - **H-S7 → `SELECTION-IMMATERIAL`, and my pre-run expectation was refuted in SIGN.** Selecting by SPEC minus
    selecting by `c3` = **−2.13 [−4.382, +0.026]** — SPEC-ranked selection is *worse*, not better. The banked
    analogue on raw `edit` was +3.25 [−2.03, +8.47], and I carried that into the prereg as the motivation.
    Cause: the `c3` ranking is **more stable across halves** (both halves put L7 first) while the SPEC ranking
    is not (half 0 → L3, half 1 → L7), so it transfers worse *even though it is the objective we care about*.
  - **H-S9 gate did NOT fire:** best config `band` +9.85 vs best single layer +8.18, gap **+1.67 < +3.00**.
    The accuracy stage therefore did not run, as pre-registered — saving ~2 GPU-h on a measurement that
    [`../../docs/nla_flippable_corpus_scoping.md`](../../docs/nla_flippable_corpus_scoping.md) shows could not
    have been interpreted anyway.

- **Observations:**
  1. **The user's question has a two-part answer.** *Can we steer all layers without it hurting?* **Yes, now.**
     All 33 layers at β=0.35 beats all 33 at β=1 on every arm (`edit` +35.35 vs +26.30, `c3` +66.44 vs +54.06)
     and beats the banked best single layer on raw `edit` (+35.35 vs +32.06) and `c3` (+66.44 vs +65.75). *Does
     it get materially better results?* **Not on the quantity that matters.** The specific component tops out
     at +8–9 nats whichever way the layers are arranged, and the multi-layer gain over one layer (+1.92) has a
     CI spanning zero. The extra raw effect is mostly the **non-specific** decoy-removal component that the
     W family already attributed ~49 % of the effect to.
  2. **A dose law exists but the dose is not conserved.** argmax β falls with k, but β·k *grows*
     (0.75 → 1.50 → 2.00 → 2.80 → 3.20 → 5.80). A strict β ∝ 1/k would predict 0.75 → 0.026; observed β at
     k=29 is 0.2, ~8× that. So more layers genuinely absorb more total write before damaging the host — the
     constraint is not a fixed energy budget.
  3. **`c3/swap` fidelity is highest at moderate β, not at β=1.** At k=16: 0.818 (β=0.1) → 0.888 → 0.962
     (β=0.35) → **0.972 (β=0.5)** → 0.949 → 0.932 (β=1). So the *channel* is most faithfully delivered by a
     partial write, which is a new statement about the NLA channel and not just about host damage.
  4. **Contiguity beats coverage.** `band` (L2–L13) wins at +9.85 despite the **smallest-but-one** editable
     union (171 vs `coverage`'s 183). The prereg's expectation `COVERAGE ≥ SPEC-TOP12` technically holds
     (+8.92 vs +8.82) by 0.10 nats, i.e. by nothing. So the editable-union mechanism that motivated H-S5's
     1.64 prediction is **not** what governs which layer set wins — consistent with H-S5 failing to clear.
  5. **Even one layer does better without full replacement** (descriptive, post-hoc cell comparison): k=1 SPEC
     +7.20 [+4.45, +10.13] at β=0.75 vs +6.05 [+3.56, +8.76] at β=1, with raw `edit` essentially identical
     (+29.29 vs +29.31) — the gain comes entirely from `foreign` falling (+22.08 vs +23.26). CIs overlap
     heavily; this is a lead, not a result. **Every banked single-layer number in this thread used β=1.**
  6. `spaced` is worst (+7.63): it is the only k=12 set containing the causally inert tail (L30, L33).
  7. Silent-failure checks all clean: identity exact to 0.0 across 21 assertions; `self` 0.000; position counts
     verified on every forward (`n_positions_written` guard); the CPU test suite caught a real scorer bug
     (`arm()` called with one argument) **before** submission.

- **New questions / new hypotheses:**
  - **H-S10:** is the k=1 β≈0.75 gain real? A focused single-layer β sweep at L3/L7 with the full β ladder and
    a pre-registered rule would settle it for ~0.1 GPU-h, and it would apply retroactively to every
    single-layer result in this thread.
  - **H-S11:** the SPEC ceiling of ~+9 nats against a `swap` ceiling of ~+77 means the **edit step**, not the
    write, is the binding constraint (SPEC is 12 % of the ceiling). That points back at H-C4 (budget dose) as
    the higher-value follow-up than any further steering geometry.
  - **H-S12:** why does contiguity beat coverage? A band may let the *same* position accumulate a consistent
    rotation over adjacent layers, where a scattered set fights the intervening computation. Testable by
    sweeping band *position* (L2–L13 vs L8–L19 vs L14–L25) at fixed width.
  - **H-S5 is not closed, it is underpowered.** Ratio 1.266 with CI [−0.57, +4.44] at n=60; the corpus doc's
    arithmetic applies here too. Any re-test should come with more items, not more cells.

- **Next Steps:** index this entry; update the thread README ledger and `CLAUDE_SCRATCHPAD.md`. Do **not**
  re-run the grid — H-S5 needs items, not cells. If steering is to continue, H-S10 is the cheap one and H-S12
  the interesting one; H-S11 argues for returning to training budget (H-C4) instead.
