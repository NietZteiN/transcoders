### Target Date: 2026-09-04 (W stage 0 — two defects found at smoke, and an anisotropy rule frozen before the full data)

Amends [`2026-09-03_nla-writeback-prereg.md`](2026-09-03_nla-writeback-prereg.md). **Append-only:
that entry is not edited.** Written while only the 3-item smoke exists (job 376947, cancelled at
4:13 deliberately); the full 60-item numbers do not exist yet and none of the rules below were
chosen with knowledge of them.

- **Hypotheses / what we're testing:** no new hypothesis. One frozen **secondary** measure is added
  (H-W0c), and two implementation defects are recorded.

#### What happened

Job **376102** FAILED in 40 s: `ModuleNotFoundError: No module named 'sglang'`. sglang is installed
in **none** of the three juno envs — it did not survive the cluster migration, and the migration
table in `CLAUDE_SCRATCHPAD.md` never listed it. Fixed by removing the dependency rather than
installing it: `nla/src/local_av.py` runs the AV in-process, reusing the vendored
`NLAClient._build_embeds` injection path verbatim and replacing only the HTTP POST with
`generate(inputs_embeds=…)`. Commit `9c4a4dd`.

Job **376947** then ran and its smoke passed — **`NLA-LIVE` on 3 items / 22 spans, CJK 0.000**, so
the Gemma AV injects correctly at `injection_scale 80000` on its first use anywhere in this project.
Two defects were visible in that smoke, and both are why it was cancelled rather than left to finish.

#### Defect 1 — every read was truncated, and the code comment asserting otherwise was false

`MAX_NEW_READ` was set to **96** under the comment *"matches capture_core.MAX_NEW_READ"*. It does
not: the project's value is **180** (`overnight_capture.py:47`). **22 of 22 smoke reads hit the wall
before their closing `</explanation>`**, mid-sentence:

> `… Final token "bit_parser" opens a function signature or import statement ("import bit_parser"), immediately requiring a function body or descriptor like`

Truncation biases the round trip downward, undercounts H-W3 mentions, and in stage 1 the truncated
text is *the thing being edited*. Fixed to 180.

**This was caught only because `n_no_tags` counts tag-less reads** — the vendored client merely
prints a warning, which in a 400-read run scrolls past unread. The counter was added for tidiness
and turned out to be the detector. Worth keeping in mind: the defect was a **wrong comment**
attached to a plausible number, which is exactly the class of error the four measurement failures
of 2026-09-03 belonged to.

#### Defect 2 — the vectors were being thrown away

`cycle_rows.jsonl` dropped `h` and `rec` on the claim that *"every downstream question is answered
by the cosines."* False, as Defect 3 below shows: the anisotropy re-analysis needs the raw vectors,
and without them it would cost a **second GPU run**. Now persisted to `cycle_vectors.npz` (~12 MB).

#### Defect 3 — raw cosine may be a nearly-degenerate discriminator here (the rule, frozen now)

Smoke cosines, n = 3 items:

| | mean |
|---|---|
| matched | **+0.9850** |
| within-item | +0.9708 |
| cross-item | +0.9694 |

The CIs do not overlap, so the frozen primary rule fires `NLA-LIVE`. But **everything sits near
0.97**: residual streams are strongly anisotropic — they share a large common direction — so *any*
two activations are near-collinear and raw cosine has little dynamic range left to discriminate
with. The frozen gate could therefore fire on a round trip that carries almost no item information,
which is precisely what stage 0 exists to rule out.

- **H-W0c (secondary, frozen here, before the full data).** Recompute all three cosines after
  **centring each set by its own mean** (`H − mean(H)`, `R − mean(R)`; each by its own, because the
  AR's outputs may carry a systematic offset that centring by the activations' mean would leave
  inside `R` to dominate every cosine). Same statistic, same cluster bootstrap over items, same
  non-overlapping-CI criterion.
  - **It does not override the primary.** The prereg's rule stands as written and is reported as the
    verdict.
  - **If primary and centred DISAGREE, the family is `W0-UNRESOLVED`** — stage 1 does not launch on
    a gate whose verdict depends on which of two defensible metrics was chosen, and a properly
    pre-registered rule is written first. This clause exists because 2026-09-03 produced four
    verdicts that were artifacts of a measurement definition (R's ratio denominator, S's ungated
    veto, T's unreachable threshold, G's normalisation), every one caught by a control rather than
    by the result looking wrong.

- **Setup:** unchanged from the prereg except `MAX_NEW_READ` 96 → 180, vectors persisted, and the
  centred diagnostic computed in the same pass. Seed 20260724, 10,000 bootstrap resamples,
  `--deterministic` OFF. Host `gemma12b` L32/48, AV/AR = `kitft/nla-gemma3-12b-L32-{av,ar}`.
- **Results:** *smoke only — 3 items, 22 spans, `NLA-LIVE`, CJK 0.000, mention rates decoy 0.455 /
  true 0.273. The full 60-item run had not been executed when this entry was written.*
- **What worked / hypothesis verdict:** *pending the full run.* The smoke establishes only that the
  Gemma AV injects and the round trip is not degenerate — **not** the gate verdict, which at n = 3
  items has no meaningful bootstrap.
- **Observations:** the smoke's mention rates (decoy 0.455 vs true 0.273) point the way H-W3
  predicted, but at n = 3 items with overlapping CIs they are an anecdote, not evidence.
- **New questions / new hypotheses:** if centred cosines collapse the separation, the honest reading
  is that the NLA round trip at these positions is dominated by the shared residual-stream
  direction — which would be a bounded, publishable negative about the instrument rather than about
  belief.
- **Next Steps:** resubmit stage 0 with all three fixes; **stop at the gate** either way.
