### Target Date: 2026-09-08 (H-W36a — `W36-NOT-RECONSTRUCTION`, on a test with almost no power; and the item-level shortfall turns out to be noise)

**Thread:** nla-harness · **Job:** none (CPU, seconds) · **Resolves:**
[`2026-09-08_reconstruction-error-prereg.md`](2026-09-08_reconstruction-error-prereg.md) H-W36a ·
**Inputs:** banked `data/nla/p0/heads/gemma12b/{vectors.npz,spans.jsonl,heads_rows.jsonl}` from job
382366. Seed 20260724.

- **Hypothesis.** Is the 1.7 % causal shortfall of the NLA round trip explained by how well it
  reconstructs? Frozen: `W36-RECONSTRUCTION` if ρ(gap, cos) ≤ −0.30 with a CI excluding zero;
  `W36-NOT-RECONSTRUCTION` if the CI contains zero and |ρ| < 0.30.

- **Results.** n = 60 items, 471 spans.

  | quantity | value |
  |---|---|
  | `cos(h0, c3)` per item | mean **0.9918**, sd **0.0017**, range 0.9872 – 0.9955 |
  | `cos` per span | mean 0.9918, sd 0.0027, min 0.9795 |
  | `gap = dG(P_patch) − dG(C3pure)` | mean **+0.482**, sd **2.892**, range −4.87 … +8.62 |
  | | **positive on 29 of 60 items** |
  | **ρ(gap, cos)** | **+0.019 [−0.243, +0.299]** |
  | ρ(dG C3pure, cos) | −0.003 [−0.237, +0.248] |

- **Verdict: `W36-NOT-RECONSTRUCTION`** — the CI contains zero and |ρ| = 0.019 ≪ 0.30. The rule
  fires as written. **It should be read as almost no evidence**, for the reason pre-registered:
  `cos` has a standard deviation of **0.0017** across items. There is no dynamic range to correlate
  against, so the test could not have detected a moderate relationship even if one existed. This is
  recorded as a fired rule, not as a finding, and the honest statement is *"reconstruction quality
  does not vary enough across this corpus for the question to be asked this way"*.

- **The unplanned observation, which is the useful part of the entry.** The item-level gap is
  **positive on only 29 of 60 items**, with sd 2.892 against a mean of +0.482. **At the item level
  the C3pure-vs-P_patch shortfall is indistinguishable from noise** — it is a difference of two
  large, noisy sums (+41.52 and +41.04), and nothing about individual items survives it.

  Yet H-W31c found the shortfall **CI-clearing at five specific components** (L46H1 −1.02, L34M
  −0.58, L41H4 −0.58, L45H3 −0.47, L47H2 −0.46). Both are true, and the reconciliation matters for
  how the residue gets described:

  - the component-level contrast is **paired within an item and within a forward pass** — the same
    sequence, the same positions, one component swapped — so the item-level variance that swamps the
    aggregate gap cancels;
  - the aggregate gap is an unpaired difference of sums and inherits the variance of both.

  So "the NLA loses 1.7 %, concentrated at the load-bearing heads" is supported **only** by the
  paired per-component measurement. Any restatement of it as "items whose reconstruction is worse
  lose more" is unsupported here — and would have been the natural next sentence to write.

- **Consequences.**
  1. **`cos` is not the right instrument for what the round trip loses**, at least on this corpus.
     That is consistent with the family's standing caveat that the 98.3 % causal fidelity and the
     raw cosine 0.9877 are *the same fact viewed twice* — a norm-matched write differing by ~9°.
     A measure with range on this corpus would have to be causal, not geometric.
  2. **H-W36b's motivation is unchanged but its framing shifts.** Whether the error lies along what
     the carrying heads read is still the open question; what today rules out is the cheap proxy for
     it. Per the prereg, H-W36b stays deferred behind H-W35 — if the carrying heads turn out to be
     generic transport, an error aligned with them says little.
  3. **Method note for the thread:** this is the second time a quantity with no dynamic range has
     been correlated against an outcome and returned a clean null (cf. the AR-space instability scale
     compressed to 0.971–0.994 in the N13 work). Worth checking spread *before* freezing a
     correlation rule, not after — the rule as frozen could not fail informatively.

- **New questions.** **H-W38:** is there any per-item property that predicts the C3pure/P_patch gap
  at all? Given sd 2.892 on a mean of +0.482, the honest prior is no, and the value of asking is to
  bound how large a per-item effect the data could hide — a power statement rather than a search.

- **Bounds.** Correlational, n = 60, on a predictor with sd 0.0017. No causal claim. The verdict word
  is reported because the rule was frozen in advance, not because the measurement is informative.
