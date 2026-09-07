### Target Date: 2026-09-06 (H-W23 — H-W17 re-run on repaired pairings, and the first test of the meaning effect in JavaScript. Frozen before running.)

Enabled by [`2026-09-06_unpaired-sentinel-defect.md`](2026-09-06_unpaired-sentinel-defect.md) and the
recovery method committed at `2edf5ca`. **Committed before the run.**

- **What changed.** `repair_pairs.recover()` rebuilds the original↔renamed correspondence the
  stimulus pipeline lost, **validated at precision 1.000 / recall 1.000 on L1 and L1b in both
  languages** against the pairings the pipeline did record. Coverage on the W corpus's 476
  annotated spans:

  | anchor | before | after |
  |---|---|---|
  | L0 | 375 | **471 (98.9 %)** |
  | L1 | 131 | **471** |
  | — of which **JavaScript** | **7** | **240** |

  **The recorded `rename_map` still takes precedence wherever it has a real key; recovery is used
  only where the pipeline wrote a `?unpaired` sentinel.** Validation says the two agree perfectly,
  so this changes nothing statistically — it keeps ground truth primary, which matters if the
  method is ever found wrong on a case the validation set did not cover.

- **Hypotheses.**
  - **H-W23a (replication):** `T_L0 − T_L1` ≥ **+12.11** nats, CI excluding 0, on the repaired
    471-span set. H-W17a measured **+21.82** [+15.52, +28.26] on 131 spans / 20 Python items.
    CONFIRM → the meaning effect survives a 3.6× expansion of the span set. REFUTE → H-W17a was a
    property of the 20-item subset.
  - **H-W23b (the new question — JavaScript, for the first time):** the same contrast computed
    **separately by language**, both reported with CIs. **If Python clears +12.11 and JavaScript
    does not, the finding is labelled language-specific** and every prior W claim about identifier
    meaning is re-scoped to Python in this ledger. If both clear, the Python-only bound recorded on
    2026-09-06 is **lifted**.
  - **H-W23c (H-W16a on more spans):** `T_L0 − T_L2` ≥ +12.11. Measured **+0.66** on 375 spans;
    expected to remain ≈ 0. A large value here would mean the expanded span set is not comparable
    to the old one, and would call H-W23a into question rather than being a finding of its own.

- **Setup (frozen).** `nla/src/nla_tiers.py --repair`. With repair on, the L1-anchored subset and
  the full span set coincide, so the existing `T_L0_sub`/`T_L1_sub` arms *are* the H-W23a contrast
  at 471 positions and `T_L0_all`/`T_L2_all` carry H-W23c. `PositionReplacer`, direction-only, no α.
  `arm_guard` is wired in and will **refuse** rather than average any arm that writes nothing —
  the fix for bugs #8/#9. **SELF re-asserted** (`|ΔG| ≤ 1.0`, **exit 3** on failure).
  `G_sum`; paired bootstrap 10,000; seed 20260724; `--deterministic` OFF; host `gemma12b` L32/48.
  Language subgroup sizes are fixed by the corpus: **30 JavaScript, 19 Python** items.

- **Stated in advance.** I have now been wrong twice on this hypothesis in the direction of my own
  prior — first predicting removal would suffice (a confound agreed with me), then framing the
  20-item subset as "thorough renaming" when it was broken metadata. **I therefore record no
  expectation for H-W23b.** The honest position is that the meaning effect has never been tested
  outside 19 Python items and that either outcome is informative.
- **Bounds.** The repaired pairings are a *derived* artifact. Their accuracy is 1.000 on the
  validation set, but that set is exactly the entries the pipeline could pair, which may be the
  easier ones. **A systematic error on hard cases would be invisible to this validation**, so a
  large H-W23a/H-W17a discrepancy must be read as a warning about the repair, not as a result.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
