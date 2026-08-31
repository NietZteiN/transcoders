### Target Date: 2026-08-07 (N7 / HT13 — **INSTRUMENT NULL**: the gate vetoes the hypothesis, and the alignment sort stays disabled)

- **Hypotheses / what we're testing:** **HT13 (confirmatory)** — judge-scored alignment between an NLA read and the surrounding reasoning falls *specifically after* the error region in wrong runs, not globally. Frozen 2026-08-06: `after_error = 1[u_read > 0.70]` (the fixed strip, clause 3), `align = score/3`, negative `correct × after_error` interaction at p<.05 **and** the before-error simple effect's CI including 0. **Four pre-specified validity gates, any failure ⇒ N7 is reported as instrument-null, HT13 is not adjudicated, and the artifact's alignment sort stays disabled.**

- **Setup:** `nla/src/{sanitize_windows,judge_align,ar_baseline}.py` + `nla/scripts/n7.sh` (tmux `nla-n7`), analysis `src/analysis/n7_alignment.py`. Seed 20260724. GPUs 2 (judge) + 3 (AR baseline), concurrently. Judge **Llama-3.1-8B-Instruct** (cross-family vs both the Qwen subject and the Qwen NLA), temp 0, **lenient JSON parsing, not constrained decoding** (it segfaults this sglang build — 2026-08-06). **5,653 items judged, 0 unparsed (0.0%)**, ~5.0 items/s, ~19 min.
  - **Blinding by dataflow, verified.** `sanitize_windows.py` is the only module that opens the graded corpus; it emits `pack.jsonl` carrying exactly `{item_id, read, window}` and holds everything else in `keys.jsonl`. The allowlist is asserted per row, and banned keys (`correct`, `expected_output`, `tier`, `task_key`) are asserted absent from the serialized pack. Item order is randomised so condition cannot be inferred from file position. **Geometry gate: 4,421/4,421 reconstructed char spans reproduce the token strings the N5 run recorded** — windows are placed exactly where the reads were taken.
  - Windows = ±1 sentence around the read's own position (median 162 chars). Nulls reuse a real item's read text and swap only the pairing, so read quality is held fixed: **shuffled** (different case, same tier, n=500) and **distant** (same case, |Δu| > 0.4, n=500).
  - Honest limit recorded in-module: the window is the model's *own reply text*, so a judge could in principle infer that a trace is going wrong without seeing a label. That cannot be removed without destroying the data; it is neutralised by HT13 being a **within-case** contrast (both windows come from the same trace) plus the two nulls.
  - **Phi-3.5-mini crashed sglang's flashinfer KV-cache kernel** (illegal memory access — sliding-window/longrope config). Fixed with `--attention-backend triton`; 800 items, 0 unparsed.

- **Results — the gate:**

  | gate | threshold | measured | verdict |
  |---|---|---|---|
  | **G1** shuffled-null AUC | ≥ 0.65 | **0.757** (SE 0.019) | ✅ pass |
  | **G2** judge beats AR baseline | AUC_judge > AUC_AR | 0.757 vs **0.733** (+0.024) | ✅ pass (narrowly) |
  | **G3** κ(Llama, Phi-3.5), n=800 | ≥ 0.40 | **κ_quad 0.049** · κ_unweighted 0.009 · r 0.134 | ❌ **FAIL** |
  | *(reported)* distant-null AUC | — | 0.710 | judge is local, not merely topical |

  Mean scores (Llama): real **1.297** · distant 0.652 · shuffled 0.494.

  **Why G3 failed is the informative part, and it is not "two competent judges disagreed":**

  | judge | mean real | mean shuffled | AUC vs shuffled | score distribution |
  |---|---|---|---|---|
  | Llama-3.1-8B | 1.30 | 0.55 | **0.744** | {0: 213, 1: 235, 2: 345, 3: 7} |
  | Phi-3.5-mini | 2.08 | 1.95 | **0.540** | {0: 8, **2: 726**, 3: 66} |

  **Phi is a near-constant rater** — it answers "2" on 91% of items and barely separates real from shuffled (AUC 0.540, chance 0.5). Item-level agreement on real items alone: **r = 0.112, κ_quad = 0.045**.
  Corroborating the same point from the other side: **the judge and the AR baseline are essentially uncorrelated per item (Spearman ρ = 0.032)** while both separate the populations (0.757 vs 0.733).

- **What worked / hypothesis verdict:** **N7 = INSTRUMENT NULL. HT13 is NOT adjudicated.** Two independent measures (a second judge, and the AR-space baseline) both fail to reproduce Llama's per-item scores while reproducing — or failing to reproduce — the population-level separation. The pre-registered conclusion follows exactly: we can show that *something* distinguishes real pairs from shuffled ones in aggregate, but **we cannot show the alignment score is a stable property of the (read, window) pair rather than of Llama-3.1-8B**. Per the frozen rule, **the artifact's alignment sort stays disabled.**

  For completeness and explicitly **not as a verdict**, the HT13 model on the full corpus (182 cases / 4,653 reads) shows **no support anyway**: interaction **+0.0233** (CI [−0.014, +0.060], p = 0.22) — the wrong sign — with alignment declining after u>0.70 by almost exactly the same amount in *both* groups (wrong 0.461→0.403, correct 0.450→0.414). That is a **global late-trace decline, not a post-error one**.

- **Observations:**
  1. **Population separation and item-level reliability are different things, and only the first is easy.** Llama's AUC of 0.757 looks like a working instrument. It is not one — a score with κ ≈ 0.05 against a second rater and ρ ≈ 0.03 against the dense baseline cannot rank individual reads, which is precisely what the artifact wanted it for. Had N7 shipped the AUC alone, the sort would have looked principled and been noise.
  2. **The AR baseline earned its pre-registration.** Free, no new dependency, and it beat the 8B judge by only 0.024 AUC while being uncorrelated with it. That is strong evidence both are picking up something diffuse (length, register, code-ness) rather than step-level correspondence.
  3. **The κ gate did its job for a reason nobody anticipated** — not judge disagreement but judge *degeneracy*. Cohen's κ is deflated when one rater has near-zero variance, so the number understates agreement in a technical sense; but a rater that says "2" 91% of the time carries no information either way, so the gate's conclusion is right even though its mechanism differs from the one imagined. Worth noting that a κ gate should be paired with a **per-rater discrimination check** (does rater B separate the nulls at all?) so the diagnosis is immediate.
  4. **No judge substitution was attempted.** The pre-registration named Phi-3.5; swapping in a different second judge after seeing this result is exactly the post-hoc rescue the discipline forbids. A future replication may use a different (larger, less prompt-brittle) second judge — declared in advance.
  5. **BH-FDR across {HT12, HT13, HT14} is now vacuous.** HT12 was refuted by a pre-registered trigger (|β| < 0.005 and CI containing 0), HT14 by an unmeasurable criterion under 92% censoring, and HT13 was never adjudicated. **No p-value enters the family**, so there is no multiplicity correction to apply — recorded explicitly so the absence of an FDR table is not mistaken for an oversight.

- **New questions / new hypotheses:** (a) Is step-level CoT↔NLA correspondence measurable *at all* with an LLM judge, or does it need a task with a discrete ground-truth mapping (e.g. does the read name the variable the step assigns) that can be checked mechanically instead of judged? The N4 experience — where mechanical execution beat the judge decisively — suggests the mechanical route. (b) Llama's real-vs-distant AUC (0.710) is only slightly below real-vs-shuffled (0.757), meaning most of its discriminating power is *not* positional. A judge that cannot localise within a trace cannot test a localised hypothesis, which may be the deeper reason HT13 was untestable here.

- **Next Steps:** update the indexes and `docs/CHECKLIST.md` with all three verdicts; rebuild the artifact with the faithfulness sort **re-captioned** (HT12 refuted: it does not track correctness) and the alignment sort **left disabled** (HT13 instrument-null), then the F2 bidirectional token↔read linking, which is independent of every verdict above.
