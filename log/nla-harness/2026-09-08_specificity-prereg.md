### Target Date: 2026-09-08 (H-W35 — are the eight carrying components specific to this content, or just where anything written at identifier spans arrives? Frozen before running.)

**Thread:** nla-harness · **Job:** not yet submitted · **Raised by:**
[`2026-09-07_head-mediation-results.md`](2026-09-07_head-mediation-results.md) ·
**Uses:** the banked `data/nla/p0/heads/gemma12b/{vectors.npz,spans.jsonl}` from job 382366 — no AV,
no AR, subject model only.

- **Why this is the control the localisation needs.** H-W31 returned `W31-CONCENTRATED`: eight of
  255 components carry 60 % of the transported state, led by `L41H4`, concentrated in the late
  global attention layers. That result is currently compatible with two very different readings:

  1. **A content circuit** — these heads carry *identifier semantics* from the span positions to the
     answer, and finding them is a mechanistic result about the trap.
  2. **A transport path** — these heads are simply where *anything* written at prompt positions
     arrives, because they are the late global heads that can still attend that far back. Then the
     eight components are a fact about Gemma's routing geometry, not about meaning, and calling them
     "the circuit that carries the clean state" would be overclaiming.

  Nothing in H-W31 separates these, because every arm it ran (`C3pure`, `P_patch`) carried the
  **same content** — the correct clean state for that span. The discriminating move is to write
  *different* content at the *same* positions and ask whether it travels the same way.

- **Setup.** `nla/src/nla_heads.py --stage sweep` with three added arms, all built from the banked
  `vectors.npz` so no AV/AR is loaded and the subject model is the only resident model (h100 is
  therefore eligible as well as h200). Same 471 spans / 60 items / 1,459 written positions, same
  identity gates, same seed 20260724, N_BOOT 10000.

  | arm | content written at the span positions | banked whole-effect value |
  |---|---|---|
  | `C3pure` | the NLA round trip of this span's clean state | +41.04 (H-W31) |
  | `P_patch` | this span's clean state, directly | +41.52 (H-W31) |
  | **`N_sibling`** | a **different span of the same item** | +39.94 — 87.4 % of P_patch |
  | **`N_foreign`** | a **different item's** clean span | +15.18 — 33.2 % |
  | **`N_random`** | a random unit direction, norm-matched | −365.59 — destroys |

  `N_sibling` and `N_foreign` are drawn deterministically (`crc32(snippet_id#span#arm)`), never from
  the span's own item / own span respectively. The two banked arms are **not re-run**; H-W31's rows
  are reused, which also makes the identity gate a cross-job reproducibility check.

- **Hypotheses and frozen rules.** All comparisons use H-W31c's machinery: Spearman between the
  255-component **sufficiency** profiles (item means) and Jaccard of the top-16 sets.
  - **H-W35a — the primary.** `C3pure` vs **`N_foreign`**:
    **`W35-GENERIC`** if ρ ≥ 0.80 **and** Jaccard ≥ 0.50 — wrong-item content travels the same path,
    so the eight components are the arrival path for writes at these positions and **must be
    reported as transport, not as a semantic circuit**.
    **`W35-CONTENT-SPECIFIC`** if ρ < 0.50. Otherwise **`W35-INTERMEDIATE`**.
  - **H-W35b — the graded version.** The same statistic for `N_sibling`, which carries *right item,
    wrong span*. Prediction ordered in advance:
    ρ(C3pure, N_sibling) ≥ ρ(C3pure, N_foreign). If that ordering inverts, something is wrong with
    the arm construction rather than with the hypothesis.
  - **H-W35c — the destructive control, descriptive only.** `N_random` writes a direction that
    *reduces* the score by hundreds of nats, so fractional recovery is meaningless for it and no
    verdict word is attached. What is read off it is only the **rank profile**: if even a
    catastrophic random write loads the same components, that is the strongest possible statement of
    the transport reading.
  - **H-W35d — joint transfer.** Take the top-8 selected on `C3pure` (opposite half, as in H-W31)
    and measure what fraction of **`N_foreign`'s** effect they recover. Descriptive; a high fraction
    is the same conclusion as H-W35a arriving by a second route.

- **Stated prior, recorded before running.** I expect **`W35-GENERIC`**. The carrying components are
  late *global* attention heads — layers 41 and 47 — chosen by the geometry of what can attend to a
  prompt span from a reply position, and there is no obvious reason that geometry would care what
  the written vector means. If that is right, the honest headline for H-W31 changes from "the
  circuit that carries identifier meaning" to **"the path by which any write at identifier spans
  reaches the answer, and the NLA's state uses all of it"** — which is a weaker mechanistic claim
  and a stronger engineering one, since it says the channel is not losing anything in routing.
  A `W35-CONTENT-SPECIFIC` result would be the more exciting outcome and I do not expect it.

- **What this cannot settle.** Sufficiency profiles are rank-compared, so two arms can share a path
  and still differ in how much they deliver along it — H-W35 is about *route*, not *magnitude*. It
  also cannot distinguish "these heads route everything" from "these heads route everything written
  **at identifier positions specifically**"; that needs a write at a different site, which is a
  separate experiment and is not claimed here.

- **Results / verdict:** not yet run.
