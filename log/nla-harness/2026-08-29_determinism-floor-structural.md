### Target Date: 2026-08-29 (the reproducibility floor survives deterministic kernels)

Follow-up to [`2026-08-29_reliability-floor-and-length.md`](2026-08-29_reliability-floor-and-length.md),
which measured a per-item floor of 0.85–0.90 on a fixed card and proposed this test. Rule frozen
in `nla/scripts/p04_det_sbatch.sh` before the run.

- **Hypotheses / what we tested:**

  **H-det.** The floor is per-process **autotuning** — cuBLAS algorithm selection, cuDNN
  benchmarking, and flash/mem-efficient attention reducing in a nondeterministic order. If so,
  pinning all three makes two runs of the identical command on one card agree exactly, the floor
  becomes a **choice**, and every future paired run should set `--deterministic`.

  Frozen rule: **D1↔D2 = 1.000 → FLOOR IS A CHOICE.** Anything less → **FLOOR IS STRUCTURAL**, and
  it must be quoted in every paired per-item claim. Secondary: **D1 vs A1** — does deterministic
  mode *change* answers rather than only stabilise them? If it does, deterministic runs cannot be
  pooled with the banked corpus.

- **Setup:**
  ```
  job      358220, h200 g-07-11, 01:24:43 · D1 and D2 in one job => same card, UUID 7db9acf2
  flags    CUBLAS_WORKSPACE_CONFIG=:4096:8 (exported BEFORE python; the runner refuses without it)
           torch.use_deterministic_algorithms(True, warn_only=True)
           cudnn.deterministic=True, cudnn.benchmark=False
           SDPA: flash OFF, mem_efficient OFF, math ON
  replicate  identical to A1/A2: baseline (60 L0 + 60 L1b) + V4_oracle alpha=1.0, layer 20,
             last_prompt, seed 20260724
  new      steer_run.py --deterministic (OFF by default — the math SDPA path is slower and
           silently enabling it would make new runs incomparable with the bank)
  ```

- **Results:**

  **FLOOR IS STRUCTURAL.** Deterministic kernels do not reproduce.

  | pairing | baseline L1b | baseline L0 | steered V4 α=1 |
  |---|---|---|---|
  | A1↔A2 — same card, **default** | 0.9000 | 0.9167 | 0.8500 |
  | **D1↔D2 — same card, deterministic** | **0.8333** | 0.9000 | **0.9000** |
  | D1↔A1 — deterministic vs default | 0.8000 | 0.9500 | 0.8333 |
  | A1↔B1 — different cards, default | 0.9500 | 0.9167 | 0.8667 |

  Pinning the kernels moved nothing: 0.8333/0.9000 against 0.9000/0.8500, and every pairing in the
  table — same card, different cards, deterministic, default — sits in the same 0.80–0.95 band.

  **The pinning genuinely applied.** `warn_only=True` was used so that an op lacking a
  deterministic implementation would proceed and warn rather than abort. **The run emitted zero
  warnings** (`grep -ic warn` → 0): every operation had a deterministic kernel and used it. This
  is not a case of the flags silently doing nothing.

  **Deterministic mode also changes answers, it does not merely stabilise them.** D1 vs A1 agrees
  on only 0.8000 of L1b items — the *lowest* number in the table — while marginal accuracy is
  identical (0.5667 both). Forcing the math SDPA path produces a different arithmetic trajectory,
  so **deterministic runs are a separate population from the banked corpus and cannot be pooled
  with it.**

  **A partial mechanism, and it is not the kernels.** Comparing the 10 disagreeing items against
  the 50 agreeing ones in D1↔D2:

  | | disagreeing (n=10) | agreeing (n=50) |
  |---|---|---|
  | mean reply length (chars) | **2073.6** | 1655.4 |
  | parse rate, D1 / D2 | **1.00 / 0.70** | — |

  Items that disagree generate **~25% longer replies**, and on 3 of the 10 one run failed to emit
  an `Output:` line at all while the other succeeded. **A meaningful share of the "disagreement" is
  not a different answer — it is one run running out of generation budget.** That points at
  `MAX_NEW_GEN = 1100` and at long-reply censoring, not at float reduction order.

- **What worked / hypothesis verdict:**
  - **H-det ✗ REFUTED — the floor is structural, not autotuning.** It cannot be removed with the
    standard determinism toolkit. **The 0.85–0.90 same-card figure stands and must be carried by
    every paired per-item claim in this programme.**
  - **Secondary ✗ — `--deterministic` is not a free upgrade.** It changes outcomes (0.80 agreement
    with default mode), so it forks the corpus rather than improving it. The flag stays off by
    default and should not be turned on for anything meant to be compared with banked runs.

- **Observations:**
  - **The residual cause is not identified, and saying so is the honest position.** Kernels are
    pinned, no op warned, decoding draws no RNG, and the card is fixed — yet two runs diverge on
    ~17% of items. The length/censoring signal above explains part of it. An untested candidate is
    bf16 with a KV cache, where buffer allocation and alignment can vary between processes and
    change kernel tiling; `use_deterministic_algorithms` guarantees same-op-same-input
    determinism, which is not the same as identical inputs. **This is a hypothesis, not a finding.**
  - **The actionable mitigation is not a determinism flag — it is the truncation budget.** If a
    chunk of the churn is replies that run past the cap, raising `MAX_NEW_GEN` or reporting
    parse-censoring separately would shrink the floor more than anything tried here. That is
    cheap to test and was not on the list before this run.
  - **The L1b/L0 asymmetry is consistent across every pairing.** Clean code reproduces better
    (0.90–0.95) than adversarially renamed code (0.80–0.95, and the lowest numbers are all L1b).
    The obfuscated condition is where the model is closest to indifferent between continuations,
    so it is exactly where a 1-ULP difference flips a token and cascades. The instability is
    *largest in the condition the whole programme studies.*

- **New questions / new hypotheses:**
  - **Test the truncation hypothesis directly** — re-run one replicate pair with `MAX_NEW_GEN`
    raised (2048) and see whether same-card agreement rises. If most of the floor is censoring,
    this is the fix, and it is ~40 min.
  - **Score on parsed replies only, as a sensitivity analysis.** Every accuracy in this programme
    counts an unparsed reply as wrong, which conflates "answered incorrectly" with "did not
    finish". The floor measurement makes that conflation quantitatively visible for the first time.
  - **Does the floor apply to the read side?** Everything here is about *generation*. Activations
    are single forward passes with no cache and no sampling; they may be perfectly reproducible,
    in which case read-side measures (`rt_cos`, `act_norm`, the P0.1 curves) are unaffected and
    only behavioural claims carry the floor. Cheap to check and it would materially narrow the
    scope of this caveat.

- **Next Steps:** the truncation test above, then the per-layer dense probe (depth for a *read*).
  P0.3-ext still blocked on `adversarial_rename`.
