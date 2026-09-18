### Target Date: 2026-09-18 (H-R18 RESULTS — under a deterministic decoder there is almost nothing to fix, and CodeSteer reliably harms)

- **Hypotheses / what we're testing:** as pre-registered in
  [`2026-09-17_greedy-pass-prereg.md`](2026-09-17_greedy-pass-prereg.md). Predictions on record:
  **a** `GREEDY-EXACT` · **b** the *same five* H-R14 verdict words · **c** descriptive, and I explicitly
  declined to predict the sign of `codesteer_auto`.

- **Setup:** frozen 148-snippet set / 1 742 cases, same packs, runtime, chat template and **same
  cross-fitted vectors** — only the decoder changed. All 8 arms, `--greedy --runs 1` (the runner refuses
  `--runs != 1` under greedy). `ase_steer_run.py` sha `a73a3fd93ad8b4c5…`; `--greedy` flips `do_sample` on
  **their** `run_llama`, so argmax is taken by their own `_sample_next_token`. Scored with the same
  decision code as the sampled pass. Jobs 410263 (smoke), 410379–410386, **410702 (resume)**, nodes
  g-04-02 / g-06-01, all rc=0, ≤ 3 concurrent GPU jobs.

- **Results:**

  **H-R18a `GREEDY-EXACT`** — two greedy runs: per-item agreement **1.000000**, |Δacc| **0.000000000**.
  The `--runs 3` guard refused correctly (rc=2).

  | arm | sampled acc | **greedy acc** | shift | Δ vs `unsteered` sampled | **Δ greedy** |
  |---|---|---|---|---|---|
  | `original_unsteered` (L0) | 0.7015 | **0.8507** | +0.1493 | — | — |
  | `erasure` | 0.6996 | 0.8490 | +0.1494 | +0.0195 [−0.023, +0.063] | +0.0052 [−0.009, +0.022] |
  | `swap_oracle` (oracle) | 0.6963 | 0.8490 | +0.1527 | +0.0163 [−0.031, +0.062] | **+0.0052 [−0.007, +0.017]** |
  | `unsteered` (L1b) | 0.6801 | **0.8439** | +0.1638 | — | — |
  | `ridge_map` (NLA) | 0.6770 | 0.8393 | +0.1623 | −0.0031 [−0.045, +0.038] | −0.0046 [−0.028, +0.012] |
  | `prompt` | 0.6914 | 0.8295 | +0.1382 | +0.0113 [−0.035, +0.057] | −0.0144 [−0.034, +0.001] |
  | `codesteer_auto` | 0.7179 | 0.8289 | +0.1110 | +0.0379 [−0.000, +0.076] | −0.0149 [−0.040, +0.005] |
  | `codesteer` | 0.6552 | 0.8169 | +0.1617 | −0.0249 [−0.076, +0.026] | **−0.0270 [−0.0519, −0.0073]** |

  **H-R18b — all five verdict words identical under both decoders**, as predicted:
  `DAMAGE-ABSENT` (+0.0069 [−0.0204, +0.0358] greedy vs +0.0214 sampled) · `CODESTEER-INERT` ·
  `NLA-MATCHES-CODESTEER` (**+0.0103** greedy vs −0.0409 sampled, α/3 [−0.0028, +0.0280]) ·
  `PROMPT-SUFFICES` (+0.0098 vs −0.0144) · `DAMAGE-FAR-WEAKER` (**+0.69 pts** vs their +36.29, ratio
  **0.019**). Restoration ratio withheld by the gate, again.

  **Precision won, as the whole point of the run:** CI half-widths on Δ vs `unsteered`,
  sampled → greedy — `swap_oracle` 0.0466 → **0.0121** (3.86× tighter) · `codesteer` 0.0508 → 0.0223
  (2.28×) · `ridge_map` 0.0414 → **0.0201** (2.06×) · `codesteer_auto` 0.0379 → 0.0228 (1.66×).

  **Per-case parse is 1.000 for seven of eight arms** (`codesteer` 0.992).

- **What worked / hypothesis verdict:**
  - **H-R18a SUPPORTED**, **H-R18b SUPPORTED** (same five words; H-R15's claim that the sampled
    exception was a draw artefact is confirmed from the other direction).
  - **The sampler was costing ~16 accuracy points.** Every arm gains +0.111 to +0.164 under argmax. The
    paper's own T 0.7 / top_k 7 decoder is leaving that on the table on this model.
  - **`codesteer` is now a reliable HARM: −0.0270 [−0.0519, −0.0073], the only interval in either pass
    that excludes zero.** The frozen ±0.05 magnitude rule still returns `CODESTEER-INERT`, and that rule
    is not being rewritten after the fact — but the honest sentence is *"CodeSteer costs CodeLlama-7B-
    Instruct about 2.7 points under greedy decoding, small but reliable."*
  - **The oracle ceiling collapses to half a point.** `swap_oracle` — the *true* clean-code state written
    at L7 — buys **+0.0052 [−0.007, +0.017]**, and the renaming damage it is meant to repair is
    **+0.0069**. Under a clean decoder **there is essentially nothing at this site to fix**, which
    retires the L7 latent-steering programme on this model far more decisively than H-R14's ~2 points did.
  - **Two more sign flips confirm H-R15's warning rather than contradicting it:** `prompt` +0.0113 →
    −0.0144 and `codesteer_auto` +0.0379 → −0.0149. Effects of this size in the sampled regime were the
    decoder, not the intervention.

- **Observations:**
  - **The parse story was mostly the sampler.** Sampled parse ran 0.853–0.932; greedy is 1.000 almost
    everywhere. So `codesteer`'s "worst parse rate of any arm" (0.853) and `codesteer_auto`'s "+0.0293
    compliance gain" were both artefacts of a T 0.7 lottery occasionally emitting malformed output — which
    is why H-R15 already found that gain unreplicable and why **H-R16 stays void**.
  - **A second-order bug this run exposed, fixed, and regression-checked.** `ase_r14_stats.py`'s parse
    column used the banked `parsed_frac` = `len(pred)/n_cases`, which counts case ids the model *invents*.
    Under greedy's clean formatting that produced **parse rates above 1.0** (1.020, 1.021 — impossible on
    its face). It is the same phantom-key fault corrected for the reported tables on 2026-09-15 that had
    survived in this helper. Fixed to a per-case rate over the pack's own keys (`ase_r14_stats.py`
    `bd2009919223743b…`, was `e16257bacd379aaa…`). **Accuracy was never affected** —
    `ase_bakeoff_stats.load` iterates `for c in tr`, so phantom keys cannot enter it — and re-scoring the
    sampled pass with the fix reproduces **every** H-R14 accuracy, Δ, CI and verdict unchanged, with the
    parse column now agreeing with the H-R14 report's own diagnostic values. No published number moves.
  - **My cost estimate for `codesteer_auto` was wrong and it cost a resume.** I predicted ~40 min by
    dividing the sampled cost by 3 runs; it took 3 h 32 m and stopped at 141/148 on the wall clock.
    Eq. 10 head calibration is **per snippet**, not per run, so greedy saves only generation time. The
    row-count check caught it — and this mattered because the scorer intersects arms to their *common*
    snippets, so scoring as-is would have silently re-scoped all eight arms to 141 snippets and looked
    clean while no longer being the pre-registered corpus.
  - **Declared, unchanged:** greedy is a deviation from their protocol (their Pass@k presupposes
    sampling; under greedy `pass@1 == pass@2 == pass@3`). The sampled numbers are **not** withdrawn and
    remain the protocol-faithful comparison to their tables. Nothing here favours our instrument: the
    greedy pass moves `ridge_map` from −0.0031 to −0.0046.

- **New questions / new hypotheses:**
  - **H-R20:** the ~16-point greedy gain is the largest effect in this entire ASE block — far larger than
    any steering contrast. If their Table 2/4 numbers were produced under T 0.7, then decoder choice
    dominates the intervention they are measuring, and the same gain may exist on their models. Testable
    on the permitted panel with no new machinery, and it reframes what "restoring obfuscation-induced
    loss" even means.
  - **H-R21:** with damage +0.0069 and an oracle ceiling of +0.0052 under greedy, **this corpus can no
    longer support a steering claim of any kind on this model.** Either a genuinely harder stimulus
    (H-R3's stronger renamer, or L2/L3 flattening) or a different host is required before further
    steering work. **This supersedes the CruxEval question** — running CruxEval arms now would measure a
    null with more decimal places.
  - **H-R19** (per-item stability under sampling) is untouched and now sharper: under greedy, per-item
    labels are exactly reproducible, so H-R19's remedy may simply be "use greedy for per-item analyses".

- **Next Steps:** report + artifact updated with the two-decoder table; recommend H-R21's stimulus
  question be settled before any further steering run.
