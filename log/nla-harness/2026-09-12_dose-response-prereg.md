# 2026-09-12 · PRE-REGISTRATION — is the NLA fidelity gap closable with DATA? (H-C4, re-scoped)

**Thread:** nla-harness · **Experiment:** Phase C follow-up, dose-response at Gemma-3-4B-it **L7**
**Status:** pre-registered, **nothing run**. Thresholds frozen; a change after the first submission is a new
dated entry, not an edit.
**Builds on:** [`2026-09-12_12b-budget-results.md`](2026-09-12_12b-budget-results.md) (H-C1 `BUDGET-EXPLAINS`) ·
[`2026-09-12_beta-layerset-results.md`](2026-09-12_beta-layerset-results.md) and
[`2026-09-12_beta-paired-followup.md`](2026-09-12_beta-paired-followup.md) (steering geometry is exhausted; the
**edit step** is the binding constraint at ~14 % of the ceiling).

## Why this, and a cost correction I owe

H-C1 established that our pipeline at 12B L32 reaches causal fidelity **0.739** where the released pair reaches
**0.98**, so the gap follows our **training recipe**, not host size. The recipe differs from ours in two ways at
once — **data volume** (86 k SFT rows from 20 k docs vs 100 k docs × 10 positions) and **method** (we have no RL).
H-C4 separates them. If fidelity is still climbing in data at our current volume, volume is the lever; if the
curve has already flattened, the residual gap is method and no amount of data will close it.

**Cost correction.** [`2026-09-12_12b-budget-results.md`](2026-09-12_12b-budget-results.md) proposed H-C4 as
"~5 GPU-h" for a 2×/4× data increase. **That estimate was wrong: it counted only training.** More SFT rows means
more documents, which means regenerating the explanation corpus, and the banked explain stage cost **10 shards ×
~3 h ≈ 30 GPU-h** (`log/slurm/384635_*_nla_ml_explain.out`, 20 000 rows/shard at ~170 min ETA). A 2× run is
therefore **~35 GPU-h**, not ~5. This entry runs the **downward** dose-response instead, at a fixed corpus, which
costs ~5 GPU-h and *decides whether the 35 GPU-h is worth spending.*

## Setup (frozen)

Host Gemma-3-4B-it text ckpt; **layer L7** — the H-S1 transport optimum, live on all four rules
(AV gap +0.168, AR fve 0.398), banked causal fidelity `S_c3/S_swap` = **0.899** and the cheapest live layer
(~71 min/pair). Config `nla/configs/nla_ml.yaml` unchanged (lr 1.41e-5, global batch 128, micro 8, **1 epoch**,
no RL, seed 20260724): **the only thing that varies is the number of TRAIN rows.**

**Dose points:** `--train-frac` ∈ {**0.25**, **0.50**, **1.00**} of the 86 822 AV / 85 895 AR train rows, plus the
**banked 100 %** pair as a reference. `--train-frac` (added today, `nla/src/nla_train.py`) takes a *seeded random*
fraction of the train rows and **leaves the holdout completely intact** — deliberately unlike `--limit`, which
also truncates the holdout to 32/64 rows and would make `holdout_fve` incomparable with the banked 858/848-row
numbers. Random rather than a prefix because rows are grouped by document, so a prefix would subsample
*documents* and confound volume with topic coverage. Tested in `nla/tests/test_beta_sweep.py`.

**Re-extraction, and why it needs a guard.** The per-layer activation files `acts/L{K}.npy` were deleted in the
2026-09-09 quota incident ([`2026-09-09_quota-exhaustion-l18.md`](2026-09-09_quota-exhaustion-l18.md)); only
`acts/norms.json` survives. They must be re-extracted (~70 GB to scratch, all 34 layers in one pass, ~30 min).
The banked 100 % pair was trained on the *original* activations, so the comparison is only clean if re-extraction
reproduces them. **Pre-flight gate (pre-registered):** the re-extracted `acts/norms.json` must reproduce L7's
banked values — `injection_scale` **exactly 5100** and `mean_av_train` **5015.6255 ± 0.5** (n_rows 200 000). If it
does not, the run stops and reports `DOSE-EXTRACT-MISMATCH`; no dose numbers are reported. This is the same check
that validated re-extraction at L2 during the FSDP equivalence work.

**The 100 % point is re-trained as a CONTROL**, not reused, so that any re-extraction or run-to-run drift is
*measured* rather than assumed away (`--deterministic` stays OFF, so training is not bit-reproducible).

**Readout.** Per-item `S_c3_L7` from `nla_ml_gate.py` (sha `1f7a9d31…`) on the same **60 items / 471 repaired
spans**, config `nla/configs/nla_ml_gate_dose.yaml` (`sets: primary: [7]`). The **paired per-item `S_c3`** is the
primary quantity, not the ratio: `S_swap` is the raw clean state and does **not depend on the trained pair at
all**, so it is a constant baseline across dose points and the ratio moves only through `S_c3`. Pairing over the
same 60 items is far more powerful than comparing two ratio-of-means with overlapping CIs. Cluster bootstrap over
items, N_BOOT 10 000, seed 20260724. Each dose point trains into its own root so **no banked pair is overwritten**.

## Hypotheses and decision rules (frozen, declared before any data)

Scale: banked L7 `S_swap` = **+73.15**, so a **0.02 change in the fidelity ratio = +1.46 nats of `S_c3`**. That is
the step size the rules are written on.

- **H-C4 (primary) — is fidelity still rising in data at our current volume?**
  - **`DATA-LIMITED`** if `S_c3(100 %) − S_c3(50 %)` ≥ **+1.46 nats** with the paired CI excluding 0.
    ⇒ the curve has not flattened; the ~35 GPU-h 2× run is justified and is the next experiment.
  - **`DATA-SATURATED`** if that step's CI **contains 0** while `S_c3(50 %) − S_c3(25 %)` ≥ +1.46 with its paired
    CI excluding 0. ⇒ data stopped paying before our current volume; the residual 0.899 → 0.98 gap is **method**
    (RL / explanation quality), and **no further data should be bought.**
  - **`DOSE-INSENSITIVE`** if neither step clears. ⇒ fidelity does not move over a 4× data range at all, which
    would say the SFT objective saturates very early and would itself redirect the programme.
  *Prediction stated now:* **`DATA-SATURATED`**. Reasoning: 1 epoch over 86 k rows already drives AV holdout loss
  to 1.317 with a permuted gap of +0.168 and AR fve to 0.398, and H-C1 found the 12B (same recipe, 2.9× the
  parameters) landed at fve 0.344 — i.e. fve barely moves across a large capacity change, which is the signature
  of an objective that is not data-starved. Recorded so it can be scored against the result.
- **H-C5 (control, must pass for H-C4 to be reportable) — does the 100 % re-train reproduce the banked pair?**
  Report `S_c3(100 % re-trained) − S_c3(banked)` paired. **Descriptive, no verdict word**, but a |difference| >
  **+4.0 nats** (≈ 0.055 of the ratio, over 2× the largest step H-C4 tests) is flagged `DOSE-CONTROL-DRIFT` and
  H-C4's verdict is reported as **provisional**, because the dose steps could not then be distinguished from
  training noise.
- **H-C6 (descriptive) — do the cheap proxies track causal fidelity?** AR `holdout_fve` and AV
  `holdout_gap_permuted_minus_real` at each dose point against `S_c3`. H-C1 already found fve is a *poor* proxy
  for fidelity across hosts (12B 0.3438 vs 4B 0.3381 at near-identical fve, very different fidelity); this asks
  whether it is a better proxy *within* a host at fixed layer. No verdict word.

## Known limitations, stated up front

1. **Downward is not upward.** A flat 50 %→100 % step is evidence that the curve has flattened *at our volume*;
   it cannot strictly exclude a second rise further out. It is the cheap discriminator, not a proof.
2. **Volume and epochs are confounded by construction** — `--train-frac` at 1 epoch means fewer optimiser steps
   as well as fewer rows. A same-steps arm (fewer rows, more epochs) would separate them and is **not** run here;
   noted as the obvious follow-up if the verdict is `DATA-LIMITED`.
3. **One layer, one host.** L7 on the 4B. H-C1's 12B L32 is a different depth and a different host.
4. **`--deterministic` is OFF**, so H-C5 measures training noise and dose together; that is exactly why H-C5
   exists and why its flag makes H-C4 provisional rather than void.
5. The explanation corpus is fixed, so this varies *volume of SFT rows*, never explanation quality — which
   H-C1 named as part of "our budget" and which this design cannot touch.

## Cost

Re-extract acts (~70 GB scratch, all 34 layers) ~0.5 GPU-h · train 25 % ~0.3 · 50 % ~0.6 · 100 % ~1.1 ·
checks ~0.3 · gate vectors+score × 3 ~2.1 ⇒ **~5 GPU-h on one GPU**, one node, `h200,h100`. Disk: 70 GB on
`/scratch` (29 TB free); `/work` untouched at 68.8 % of its 1.1 TB hard quota.
