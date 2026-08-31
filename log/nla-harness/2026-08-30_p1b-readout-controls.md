### Target Date: 2026-08-30 (readout controls — late readout confirmed, and obfuscation degrades the read)

Two controls proposed in [`2026-08-30_p1b-graded-labels.md`](2026-08-30_p1b-graded-labels.md),
plus a third added when the first turned out to be confounded. All CPU, no new GPU time.
Scripts: `nla/src/p1b_readout_controls.py`, `nla/src/p1b_answer_deviation.py`.

- **Results:**

  **Control A — is the emitted answer decodable at the answer line?** The LATE READOUT ONLY reading
  predicts that properties of the *answer*, which are not correctness, should be decodable there
  and not at the prompt. Target: answer character length, graded Ridge, same folds.

  | position | argmax ρ | mean ρ | reply-length baseline |
  |---|---|---|---|
  | `last_prompt` | L26 **+0.6966** | +0.4456 | +0.0762 |
  | `answer_line` | L19 **+0.8531** | +0.6002 | +0.0762 |

  The reply-length baseline of +0.0762 rules out the obvious artefact — this is not reply length in
  disguise. **But the control as specified is confounded**: the *expected* answer's length is a
  property of the item, readable from the code itself, so a probe at the prompt scoring 0.6966 may
  be decoding the **task** rather than a decided answer. That was not anticipated and is the
  reason for Control A′.

  **Control A′ — the confound-free version.** Target: **|len(answer) − len(truth)|**, the deviation
  of what the model actually emitted from what it should have. Truth length is knowable from the
  prompt; the deviation is not — it requires knowing what the model will do.

  | position | argmax ρ | truth-length baseline | beats baseline |
  |---|---|---|---|
  | `last_prompt` | L11 +0.3650 | **+0.3767** | **no — it is below it** |
  | `answer_line` | L26 **+0.5299** | +0.3767 | **yes, +0.153** |

  **At the prompt the residual stream carries no information about what the model will actually
  emit** — the probe scores *under* the baseline that knows only the task. At the answer line it
  clearly does. Δ = **+0.1649**.

  **→ LATE READOUT ONLY is confirmed on an independent target.** The answer-line correctness signal
  (ρ = 0.5702) is a corollary of the answer having been decided, not a prediction of correctness.

  **Control B — does `last_prompt` ≈ length hold on CLEAN code?** Graded k/10 targets built from
  `l0_correct` and `l1b_correct` across the same ten draws.

  | tier | mean correct | argmax ρ | mean ρ | ρ@L13 | ρ@L20 | ρ@L27 |
  |---|---|---|---|---|---|---|
  | **L0 (clean)** | 0.633 | L20 **+0.5201** | +0.4066 | +0.4457 | **+0.5201** | +0.4149 |
  | **L1b (renamed)** | 0.550 | L26 **+0.3818** | +0.2717 | +0.3583 | +0.3253 | +0.3628 |

  **Δ = +0.1383.** The L1b figure (+0.3818) sits essentially on its length baseline (+0.3626,
  measured 2026-08-30). The L0 figure is well above that number — though see the limitation.

  **This is a positive finding and it is the study's actual subject:** at the same read site, in the
  same model, correctness is substantially more decodable on clean code than on adversarially
  renamed code. **Adversarial renaming does not merely make the model wrong more often — it
  degrades the readability of the model's own correctness from its residual stream.**

  **Limitation, stated rather than worked around.** `baseline.jsonl` records `reply_chars` for the
  L1b generation only, so **no length baseline exists on disk for L0**, and the +0.5201 is compared
  against L1b's baseline rather than its own. The comparison of residual-stream ρ across tiers is
  sound; the "beats length" half of it is not established for L0. **Future baseline runs should
  record L0 reply length** — a one-line change worth making before this is quoted.

- **What worked / hypothesis verdict:**
  - **LATE READOUT ONLY ✓ CONFIRMED** on a target independent of correctness. The readout paper's
    most attractive hypothesis is closed: the model does not represent its eventual correctness
    before committing, and the large answer-line effect is readout.
  - **Obfuscation degrades correctness-decodability ✓** — Δρ = +0.1383 between tiers at the same
    site, with the baseline caveat above.
  - Control A as originally specified was **confounded and is not quoted**; A′ replaces it. The
    confound was found by asking what else could produce 0.6966 at a position before generation
    had begun.

- **Observations:**
  - **The answer-length result at the prompt is interesting in its own right and should not be
    lost:** ρ = 0.6966 for answer *length* versus ρ = 0.3730 for *correctness*, at the same site.
    The model's residual stream at the final prompt token strongly encodes what **kind** of answer
    is coming while barely encoding whether it will be **right**. Control A′ shows the length part
    is largely the task, but the dissociation between "form is determined early" and "correctness
    is not" is the sharpest version of this programme's whole finding.
  - **The tier result is the first clean positive of the week.** Everything else since Phase 0 has
    been a bounded negative. This one has a direction, a magnitude, and a mechanism-shaped
    interpretation — and it needs its length baseline before it can be published.
  - Three of this week's results now rest on the ten cheap label draws rather than on new GPU time.
    The most valuable thing built this week was not an experiment but a lower-variance label.

- **New questions / new hypotheses:**
  - **Re-run one baseline pass recording L0 reply length**, then repeat Control B properly. ~15 min
    of GPU and it converts the week's one positive from suggestive to reportable.
  - **Is the tier gap a depth effect?** L0 peaks at L20 (+0.5201) and L1b at L26 (+0.3818), and the
    L0 curve is above the L1b curve at every layer quoted. Whether renaming shifts *where* the
    signal lives or merely attenuates it everywhere is answerable from the arrays already saved.
  - The L2/L3 tiers remain unprobed, and after the tier result they are more interesting than they
    were this morning: if flattening degrades decodability like renaming does, the effect is about
    obfuscation generally; if not, it is specific to the atom-level route.

- **Next Steps:** record L0 reply length and redo Control B; then the L2/L3 tier probe.
  P0.3-ext still blocked on `adversarial_rename`.
