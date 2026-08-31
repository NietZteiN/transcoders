### Target Date: 2026-08-30 (reads are bit-exact; truncation is not the mechanism)

Two follow-ups from [`2026-08-29_determinism-floor-structural.md`](2026-08-29_determinism-floor-structural.md),
run back to back in **one job on one card** so both comparisons are same-card by construction.
Rules frozen in `nla/scripts/p04_followup_sbatch.sh` before the run.

- **Hypotheses / what we tested:**
  - **H-read.** Everything measured so far concerns generated text. Generation is autoregressive
    over ~1,100 cached decode steps, so one flipped token cascades; an activation is a **single
    forward pass, no cache, no sampling, no cascade**. If reads reproduce exactly, the floor is a
    property of the causal arm and read-side measures carry no caveat at all.
  - **H-trunc.** Of the 10 items that flipped between two same-card runs, **3 had one run emit no
    `Output:` line**, and disagreeing items ran ~25% longer than agreeing ones. If the floor is
    partly replies truncated before they finish, raising `MAX_NEW_GEN` from 1100 to 2048 should
    raise agreement.

- **Setup:**
  ```
  job     359039, h200 g-07-08, 01:06:45 · all four steps one allocation, GPU-65cd8be7
  reads   nla/src/p04_read_repro.py — layer 20, final prompt token, the exact activations
          steer_run.last_tok_act feeds to V3/V4. 60 pairs x 2 tiers = 120 vectors.
          R1, R2 = two processes; each also extracts twice within itself.
  gens    steer_run.py --max-new-gen 2048 (new flag; default stays 1100 so the bank stays
          comparable), replicates E1/E2, otherwise identical to A1/A2.
  mode    default, NOT --deterministic — that flag was shown 2026-08-29 to fork the corpus.
  ```
  A scheduling note worth recording: the first submission (358797) sat **PENDING for hours**
  because it was pinned with `--nodelist` to a node that had a free GPU but **all 64 CPUs
  allocated**. The pin bought nothing — same-card is guaranteed by a single job allocation on any
  node. Unpinned and asking 4 CPUs, it scheduled in under a minute. **Do not pin a node to get a
  same-card comparison.**

- **Results:**

  **H-read ✓ CONFIRMED — reads are bit-identical.**

  | comparison | max abs Δ | mean abs Δ | bit-identical |
  |---|---|---|---|
  | within-process R1 | **0.0** | 0.0 | **120/120** |
  | within-process R2 | **0.0** | 0.0 | **120/120** |
  | **cross-process, same card** | **0.0** | 0.0 | **120/120** |

  Zero drift, not small drift. (`min_cosine` reads 0.99999988 — that is float32 rounding in the
  cosine of two *identical* vectors, not evidence of divergence; `max_abs_delta` is exactly 0.)

  **H-trunc ✗ REFUTED — the floor does not move.**

  | pairing | baseline L1b | baseline L0 | steered V4 |
  |---|---|---|---|
  | A1↔A2 — budget 1100 | 0.9000 | 0.9167 | 0.8500 |
  | D1↔D2 — deterministic | 0.8333 | 0.9000 | 0.9000 |
  | **E1↔E2 — budget 2048** | **0.8833** | 0.8500 | **0.8833** |
  | E1↔A1 — 2048 vs 1100 | 0.8667 | 0.9000 | 0.9167 |
  | A1↔B1 — different cards | 0.9500 | 0.9167 | 0.8667 |

  **But the budget fixed censoring completely:** parse rate **0.8833 → 1.0000**, with L1b accuracy
  unchanged (0.5667 both) and mean reply length barely moved (1670 → 1766 chars).

  **So censoring and irreproducibility are separate problems.** Doubling the budget removed every
  parse failure and left agreement inside the same 0.85–0.90 band. The 3-of-10 parse-failure
  observation that motivated this test was a **correlate, not a cause** — long replies are both
  more likely to be truncated and more likely to diverge, and fixing the first does not fix the
  second.

- **What worked / hypothesis verdict:**
  - **H-read ✓ — the floor is GENERATION-ONLY.** `rt_cos`, `act_norm`, the P0.1 layer curves, the
    N10b read scores, the unified 14,619-row table — **no reproducibility caveat applies to any of
    them.** The caveat is confined to graded generations, i.e. the causal arm.
  - **H-trunc ✗ — truncation is not the mechanism.** The floor is now robust to *three*
    interventions: different cards, deterministic kernels, and a doubled generation budget.
    **Its cause remains unidentified**, and after three failed explanations the honest position is
    to treat 0.85–0.90 as an empirical property of greedy bf16 decoding on this stack and quote it.
  - **Independent of reproducibility, `--max-new-gen 2048` should be the default for new runs.**
    A 0.8833 parse rate means 12% of items were scored wrong for *not finishing* rather than for
    being wrong — that conflation is worth removing on its own merits, and it costs ~15% wall time.

- **Observations:**
  - **This substantially de-escalates yesterday's caveat.** "Every paired per-item table in the
    programme inherits an unmeasured noise floor" was too broad: N13's AUCs, N10b's reliability
    and every `rt_cos`/`act_norm` result are read-side and provably exact. What carries the floor
    is B4's McNemar gate, B5's cells, P0.3's P@1 tables and P0.4's criterion (a) — the graded-
    generation results, which is a much smaller set.
  - **The L1b/L0 asymmetry did not survive the wider budget.** At 1100 the obfuscated tier
    reproduced worse than clean (0.90 vs 0.92); at 2048 it reproduces *better* (0.8833 vs 0.8500).
    With 60 items a difference of 2 items is noise, so **the "instability is largest in the
    obfuscated condition" reading in the 08-29 entry is not supported by the wider data** and
    should not be carried forward.
  - Three pre-registered mechanisms have now been tested and refuted in two days at a cost of
    about four GPU-hours. That is the cheap end of this project, and each negative narrowed the
    claim rather than merely failing.

- **New questions / new hypotheses:**
  - **Re-run the graded arm at 2048?** Every banked accuracy in the programme is computed with 12%
    of replies censored. It does not change B4's or P0.4's verdicts — parse failures count as wrong
    in both arms of every paired contrast — but the *absolute* accuracies are all biased low by an
    unknown amount. A single re-run of the B4 primary cells at 2048 would bound it.
  - The unidentified residual is now the only open thread here. Remaining untested candidate:
    batch-invariance of the attention kernel under differing KV-cache buffer alignment across
    processes. Testing it properly means instrumenting logits at step 1 and diffing, which is a
    different kind of work and probably not worth it — **the practical answer (quote the floor,
    trust the reads) is already in hand.**

- **Next Steps:** the per-layer dense probe — depth for a *read*, now known to be a bit-exact
  measurement. P0.3-ext still blocked on `adversarial_rename`.
