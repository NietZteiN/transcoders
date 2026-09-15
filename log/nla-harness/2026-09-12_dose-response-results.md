# 2026-09-12 · RESULTS — H-C4 `DATA-LIMITED`: the fidelity gap is VOLUME, not method. My prediction is refuted.

**Thread:** nla-harness · **Experiment:** `C4_dose_response`, Gemma-3-4B-it **L7** · **Status:** H-C4 resolved,
H-C5 control **passed**, H-C6 reported. Pre-registration
[`2026-09-12_dose-response-prereg.md`](2026-09-12_dose-response-prereg.md) — rules applied unchanged; the one
deviation was declared in advance in
[`2026-09-12_dose-liveness-deviation.md`](2026-09-12_dose-liveness-deviation.md).

### Target Date: 2026-09-12 (dose-response at fixed corpus; is data the lever?)

- **Hypotheses / what we're testing:** **H-C4** — is causal fidelity still rising in training volume at our current
  data? `DATA-LIMITED` if the paired `S_c3(100 %) − S_c3(50 %)` step ≥ **+1.46 nats** (= 0.02 of the +73.15 ceiling)
  with CI excluding 0 · `DATA-SATURATED` if that step's CI contains 0 while 50 %−25 % clears · `DOSE-INSENSITIVE`
  if neither. **H-C5** — does a 100 % re-train reproduce the banked pair (|Δ| > 4.0 nats ⇒ provisional)?
  **H-C6** — do `fve` / AV gap track causal fidelity *within* a host? **Prediction recorded in the prereg:
  `DATA-SATURATED`.**

- **Setup:** jobs **391398** (FAILED at 01:20:30 on `g-08-05`, see the deviation entry) and **391522**
  (`nla_dose`, h200, node `g-07-04`, 1 GPU, 2026-09-12T13:27:26 → 16:43:55, **elapsed 03:16:29 = 3.27 GPU-h**,
  COMPLETED rc=0). Script `nla/scripts/nla_dose_sbatch.sh` sha `ef8cdab41f1d4746…` → trainer
  `nla/src/nla_train.py` sha `8747fc53cfc11790…` (with the new `--train-frac`), gate `nla/src/nla_ml_gate.py`
  sha `1f7a9d31…` with `nla/configs/nla_ml_gate_dose.yaml` sha `bd34400976a263b9…` (differs from the banked gate
  config in `sets` **only**), scorer `nla/src/dose_score.py` sha `4c1b9336804ecbb0…`. Seed **20260724**,
  N_BOOT 10 000, cluster bootstrap over the same **60 items / 471 repaired spans**, `--deterministic` OFF.
  Dose points trained into separate roots `/scratch/juno/jvl210002/nla_ml_dose/f{025,050,100}` with
  corpus/explain/acts symlinked, so **no banked pair was touched** and the ~70 GB extraction was shared.
  Outputs `data/nla/ml/gemma4b/gate/dose/dose_stats.json`; log `log/slurm/391522_nla_dose.out`.

  **Pre-flight passed exactly.** The activations deleted in the 2026-09-09 quota incident were re-extracted and
  reproduced the banked L7 norms to fp32: `injection_scale` **5100 = 5100**, `mean_av_train`
  **5015.62548828125** vs the banked **5015.6255**, n_rows 200 000.

- **Results:**

  | dose | train rows (AV) | `S_c3` | fidelity ratio `S_c3/S_swap` | AR `fve` | AV gap | live? |
  |---|---|---|---|---|---|---|
  | 25 % | ~21.7 k | **+38.23** [+32.66, +43.87] | **0.523** | 0.1407 | +0.1312 | ✗ (rule b) |
  | 50 % | ~43.4 k | **+57.58** [+51.21, +64.33] | **0.787** | 0.3264 | +0.1454 | ✓ |
  | 100 % (re-trained) | 86 822 | **+66.54** [+59.64, +73.59] | **0.910** | 0.3984 | +0.1681 | ✓ |
  | 100 % (banked) | 86 822 | +65.75 [+58.70, +72.94] | 0.899 | 0.3984 | +0.1680 | ✓ |

  `S_swap` = **+73.15**, pair-independent (the raw clean state does not involve the trained AV/AR), so it is a
  constant baseline and the ratio moves only through `S_c3`.

  **Paired steps (the test statistic):**
  - `S_c3(100 %) − S_c3(50 %)` = **+8.96 [+5.08, +13.24]**, n=60 — **6.1× the +1.46 bar**, CI far from 0.
  - `S_c3(50 %) − S_c3(25 %)` = **+19.35 [+14.97, +24.04]**, n=60.

- **What worked / hypothesis verdict:**
  - **H-C4 → `DATA-LIMITED`.** The curve is still climbing steeply at 100 % of our data. The top step alone is
    worth **+8.96 nats = 0.123 of the ceiling**, six times the pre-registered bar.
  - **My pre-registered prediction `DATA-SATURATED` is REFUTED.** I argued it from `fve` barely moving across a
    2.9× *capacity* change (12B 0.344 vs 4B 0.338), and concluded the objective was "not data-starved". That
    inference does not transfer from capacity to **volume**: over a 4× volume range `fve` runs
    0.141 → 0.326 → 0.398 and `S_c3` runs +38.2 → +57.6 → +66.5. The error was treating one axis's insensitivity
    as evidence about a different axis. (Flagged under pressure in the deviation entry once the 25 % `fve` came
    in, before the decisive 50 %→100 % step was observed.)
  - **H-C5 control → PASSED, so H-C4 is not provisional.** The 100 % re-train minus the banked pair is
    **+0.78 [−1.76, +3.21]** — inside the 4.0-nat flag and a CI containing 0. Re-extraction plus a fresh
    non-deterministic training run reproduces the banked pair, so the dose steps (+8.96, +19.35) are far outside
    training noise and are not an artifact of the deleted activations.
  - **H-C6 → the cheap proxies DO track fidelity within a host.** `fve` 0.141 / 0.326 / 0.398 and AV gap
    +0.131 / +0.145 / +0.168 are both monotone with `S_c3`. This sharpens H-C1 rather than contradicting it: `fve`
    is a poor proxy **across hosts** (12B 0.3438 vs 4B 0.3381 at near-identical `fve` but very different fidelity)
    and a usable one **within** a host at fixed layer. So `fve` may be used as a cheap dose monitor, never as a
    cross-host fidelity claim.

- **Observations:**
  1. **This resolves the ambiguity H-C1 left.** H-C1 showed the 0.98-vs-ours gap follows our *recipe* rather than
     host size, but the recipe differed in **volume** and **method (no RL)** at once. H-C4 now separates them:
     inside our recipe, **volume is the binding constraint**, and RL is not needed to explain the gap.
  2. **Forecast, registered as falsifiable rather than asserted.** Increments per doubling are **+0.264** then
     **+0.122** — a ratio of **0.463**, i.e. roughly halving. A linear-in-log₂ fit is **inadmissible** (it predicts
     ratio 1.13 at 2×, and fidelity is bounded by 1), so only the saturating form is usable. Extrapolating the
     halving: **2× → 0.966, 4× → 0.992.** The released pair's **0.98 therefore sits between 2× and 4× our data.**
     This is three points and two increments, extrapolated post-hoc — recorded as a **prediction to be scored**,
     not a result. If a 2× run lands below ~0.95 the saturating model is wrong too.
  3. **The 35 GPU-h 2× run is now justified** — which is precisely the decision H-C4 was built to make, for
     3.27 GPU-h instead of 35. The cost is dominated by regenerating the explanation corpus
     (~30 GPU-h, 10 shards × ~3 h), not by training (~1.2 GPU-h/pair at L7).
  4. **A 25 %-data pair is not a usable instrument** (`fve` 0.1407 < the 0.20 floor) yet still carries
     `S_c3` +38.23 — over half the ceiling. Liveness and causal usefulness are not the same axis, which is worth
     remembering wherever the liveness rule is used as a filter.
  5. **The edit step remains the real bottleneck, and this does not fix it.** H-C4 moves *transport* (`S_c3`).
     The specific edit effect measured in [`2026-09-12_beta-layerset-results.md`](2026-09-12_beta-layerset-results.md)
     was ~+9 nats against a ~+77 ceiling (~12–14 %). Better transport raises the ceiling the edit works against; it
     is not yet evidence that the edit itself improves with data. That is the next thing to test, not to assume.

- **New questions / new hypotheses:**
  - **H-C7 — buy the data.** 2× volume (40 k docs) at 4B L7: ~30 GPU-h explanations + ~1.2 GPU-h training +
    ~0.7 GPU-h gate. Pre-registered target from observation 2: **ratio 0.966 ± ?**; `DOSE-MODEL-HOLDS` if within
    a band to be frozen before the run, refuted otherwise.
  - **H-C8 — does the EDIT arm improve with volume, or only transport?** Free from the same three dose points:
    `S_edit − S_foreign` per dose is already in the banked `gate_rows.jsonl` files. **No GPU.** This should be run
    before H-C7, because if the specific edit effect is flat in volume then buying data raises the ceiling without
    buying the thing the programme actually wants.
  - **H-C9 — is the 12B's 0.739 also volume-limited?** Same design at 12B L32 would cost ~56.5 GPU-h/layer plus
    the corpus; deferred, and H-C7 at 4B is the cheaper test of the same mechanism.
  - Volume and optimiser steps remain confounded (prereg limitation 2): `--train-frac` at 1 epoch varies both. A
    same-steps arm (fewer rows, more epochs) separates them and is now worth running given `DATA-LIMITED`.

- **Next Steps:** run **H-C8 first (no GPU)** — it decides whether H-C7's 35 GPU-h buys the edit or only the
  ceiling. Index this entry; update the thread README ledger and `CLAUDE_SCRATCHPAD.md`.
