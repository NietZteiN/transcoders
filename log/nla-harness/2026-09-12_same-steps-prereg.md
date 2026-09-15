# 2026-09-12 · PRE-REGISTRATION — is the dose effect DATA or OPTIMISER STEPS? (H-C10)

**Thread:** nla-harness · **Experiment:** H-C10, same-steps control at 4B **L7** · **Status:** pre-registered,
nothing run. Thresholds frozen.
**Builds on:** [`2026-09-12_dose-response-results.md`](2026-09-12_dose-response-results.md) (H-C4 `DATA-LIMITED`)
and [`2026-09-12_dose-edit-effect.md`](2026-09-12_dose-edit-effect.md) (H-C8: the edit effect rises too).

## The confound this closes

`--train-frac` at a fixed 1 epoch varies **unique rows** and **optimiser steps** together — limitation 2 of
[`2026-09-12_dose-response-prereg.md`](2026-09-12_dose-response-prereg.md), now the thing standing between H-C4
and a ~35 GPU-h purchase. Measured step counts at global batch 128:

| arm | rows | epochs | steps/epoch | **total steps** |
|---|---|---|---|---|
| `f100` (H-C4's 100 %) | 86 822 | 1 | 678 | **678** |
| **`f050e2` (this run)** | 43 411 | **2** | 339 | **678** |
| `f050` (H-C4's 50 %) | 43 411 | 1 | 339 | 339 |

`f050e2` matches `f100` on steps **exactly** and halves the unique rows, so the contrast isolates data from
optimisation. If half the data seen twice reproduces the full-data result, H-C4's effect was optimisation and
**buying a corpus would be a mistake.**

## Setup (frozen)

Identical to H-C4 in every other respect: host Gemma-3-4B-it, L7, `nla/configs/nla_ml.yaml`, seed 20260724,
`--deterministic` OFF, same shared acts (pre-flight already reproduced the banked L7 norms to fp32), own root
`/scratch/juno/jvl210002/nla_ml_dose/f050e2` so no banked or dose pair is overwritten. `--epochs` (added today to
`nla/src/nla_train.py`, default = config's 1) overrides epochs and is recorded in both SFT manifests.
Scored by `nla_ml_gate.py` with `nla/configs/nla_ml_gate_dose.yaml` and **`--ignore-liveness`**, per the standing
deviation ([`2026-09-12_dose-liveness-deviation.md`](2026-09-12_dose-liveness-deviation.md)) so all dose arms stay
apples-to-apples. Readouts: paired per-item **`S_c3`** (primary) and **`SPEC = S_edit − S_foreign`** (co-primary,
because H-C8 showed that is the quantity the programme wants). Step bar reused unchanged: **+1.46 nats**
(= 0.02 × the +73.15 ceiling). N_BOOT 10 000, cluster bootstrap over the same 60 items.

## Hypotheses and decision rules (frozen)

Let `A = f050e2` (half data, 2 epochs), `B = f100` (full data, 1 epoch, equal steps), `C = f050` (half data,
1 epoch, half the steps). All contrasts paired per item on `S_c3`; `SPEC` reported alongside under the same rule.

- **`STEPS-NOT-DATA`** if `B − A` has its CI **containing 0** *and* `A − C` ≥ **+1.46** with CI excluding 0.
  ⇒ the second pass over half the data recovered the full-data result; H-C4's effect is **optimisation**, and
  **H-C7's ~30 GPU-h corpus regeneration should NOT be bought.**
- **`DATA-NOT-STEPS`** if `B − A` ≥ **+1.46** with CI excluding 0. ⇒ unique rows beat repeated rows at equal
  steps; H-C4 is a genuine data effect and **H-C7 is justified.**
- **`MIXED`** otherwise (e.g. `A − C` clears *and* `B − A` clears, or neither clears) — reported as such, with
  both contrasts, and H-C7 left as a judgement call rather than argued either way.

**Prediction recorded: `DATA-NOT-STEPS`.** Unique data normally beats repetition in SFT at matched steps. Stated
with low confidence and flagged deliberately: my last pre-registered prediction (`DATA-SATURATED`, H-C4) was
**refuted**, and the lesson I drew there — that insensitivity on one axis is not evidence about another axis —
applies to this prediction too, since it is an extrapolation from generic SFT behaviour rather than from anything
measured on this pipeline.

## Cost

AV 678 steps ~45 min · AR ~671 steps ~20 min · check ~4 min · gate vectors ~35 min + score ~5 min ⇒
**~1.8 GPU-h**, one GPU, acts and corpus already on scratch. (My earlier "~1 GPU-h" in the H-C8 entry
under-counted the 35-minute vector stage; corrected here.)
