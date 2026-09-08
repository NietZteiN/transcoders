### Target Date: 2026-09-07 (H-W31 — `W31-CONCENTRATED` and `W31-SAME-CIRCUIT`: 8 of 255 components carry 60 % of the transported state, and the NLA's version routes where the raw one does)

**Thread:** nla-harness · **Job:** 382366 (h200, g-07-07, **2:27:24**, COMPLETED 0:0, 0 errors) ·
**Host:** `gemma12b` L32/48 · **Resolves:**
[`2026-09-07_head-mediation-prereg.md`](2026-09-07_head-mediation-prereg.md) ·
**Code:** `nla/src/nla_heads.py`, `nla/src/head_patch.py`, config `nla/configs/nla_heads.yaml`
(seed 20260724, N_BOOT 10000) · **Artifacts:** `data/nla/p0/heads/gemma12b/{vectors.npz,
spans.jsonl, heads_rows.jsonl, joint_rows.jsonl, heads_stats.json}`.

- **Hypotheses:** H-W31-ID (identity gate), H-W31a (localisation), H-W31b (heads vs MLPs),
  H-W31c (does the NLA-transported state route like the raw one), H-W31d (global heads),
  H-W31e (direct readers).

- **Setup.** 471 repaired spans / 60 items / **1,459 written prompt positions**. Per item, two
  reference forwards on `l1b_prompt_ids + l0_reply_ids` — unsteered (U) and steered at L32 (S) —
  then every head and MLP of layers 33–47 (255 components) patched in both directions:
  `suf_c = dG(U, c←S)` and `nec_c = dG(S) − dG(S, c←U)`, plus a read-side knockout per head against
  a per-layer mask-only reference. Joint pass with top-k selected on the **opposite crc32 half**.
  **93,060 sweep forwards at 68 ms + 5,340 joint forwards**; vectors stage mean cos(h0, c3) 0.992.

- **Results.**

  **H-W31-ID — the gate, passed exactly.** `SELF` max |Δ| **0.0000** over **960** assertions,
  `ALL` max **0.0000**, `ko_gap` max **0.0000**. Not "within the 0.05 tolerance" — bit-identical.
  Anchor: `P_patch` dG_S = **+41.52** [+36.95, +46.25], reproducing H-W23's repaired-corpus **+41.52**
  from a different script; `C3pure` **+41.04** [+36.53, +45.68].

  **H-W31a — joint pass, k selected on the other half (C3pure / P_patch):**

  | k | TOP | RAND | LRAND | recovered | TOP − RAND (frac of dG_S) |
  |---|---|---|---|---|---|
  | 1 | +7.28 / +7.86 | −0.13 / +0.30 | +1.24 / +1.52 | 0.18 / 0.19 | +7.41 / +7.56 (0.18) |
  | 2 | +10.01 / +10.91 | −0.11 / +0.17 | +0.83 / +0.84 | 0.24 / 0.26 | +10.12 / +10.74 (0.25 / 0.26) |
  | 4 | +15.41 / +17.64 | +0.29 / +0.34 | +1.20 / +1.52 | 0.38 / 0.42 | +15.12 / +17.30 (0.37 / 0.42) |
  | **8** | **+24.73 / +25.78** | +1.12 / +0.98 | +1.26 / −0.58 | **0.60 / 0.62** | +23.61 / +24.80 (0.58 / 0.60) |
  | **16** | **+32.94 / +34.53** | +2.02 / +0.48 | +4.88 / +1.58 | **0.80 / 0.83** | **+30.92 / +34.06 (0.75 / 0.82)** |
  | 32 | +39.02 / +40.91 | +4.26 / +4.74 | +13.95 / +14.87 | 0.95 / 0.99 | +34.75 / +36.17 (0.85 / 0.87) |

  Every TOP − RAND and TOP − LRAND CI excludes zero. `ALL / dG_S` = **1.000** on both arms.

  **H-W31b:** best head **L41H4** (+7.28 / +7.86), best MLP **L34M** (+4.81 / +5.39); the top-16 is
  **14 heads and 2 MLPs**. Single-component sufficiencies sum to **0.68 / 0.62** of dG_S — sub-additive.

  **Depth profile (summed necessity per layer, C3pure):** 41 **+4.78** · 47 **+5.74** · 35 +1.00 ·
  36 +0.55 · 44 +0.87, with every other layer at or below zero and **layer 34 at −2.44**.
  P_patch: 41 **+5.07** · 47 **+4.03** · 35 +1.90 · 36 +1.25.

  **H-W31c:** Spearman between the two arms' 255-component sufficiency profiles **0.838**,
  Jaccard of the top-16 **0.78**. Residue where C3pure is *below* P_patch with a CI-clearing margin:
  **L46H1 −1.02**, L34M −0.58, L41H4 −0.58, L45H3 −0.47, L47H2 −0.46.

  **H-W31d:** global-minus-local necessity **+0.291** [+0.230, +0.355] (C3pure) and **+0.277**
  [+0.220, +0.335] (P_patch). Long-item subgroup (n = 11): **+0.270** [+0.185, +0.370] vs short
  **+0.296** [+0.223, +0.371].

  **H-W31e:** Spearman(read, nec) **0.49 / 0.43**; **13 of the top-16 by necessity** have a read
  effect whose CI excludes zero.

- **Hypothesis verdicts.**
  - **H-W31-ID ✓ PASS** — exactly, on all three identities.
  - **H-W31a → `W31-CONCENTRATED`, both arms.** The frozen bar was TOP_16 ≥ 50 % of dG_S *and*
    ≥ 30 points over RAND_16 with a CI excluding zero: measured **80 % / 83 %** recovered and
    **75 / 82** points over random. It is not close. **50 % is first crossed at k = 8 — 3.1 % of the
    255 components** — and a **single head, L41H4, carries 18 %** of an effect that has to cross
    from prompt positions to reply tokens.
  - **H-W31b — descriptive.** Attention dominates (14 of 16), but the MLP directly above the write
    site is the second-strongest single component, which the pre-registration did not anticipate.
  - **H-W31c → `W31-SAME-CIRCUIT`** (bar: ρ ≥ 0.80 and Jaccard ≥ 0.50). **The NLA-transported state
    is read by the same heads, in the same order, as the raw clean state.**
  - **H-W31d — ✓ on its primary, ✗ on its secondary.** Global-layer heads do carry more necessity,
    CI excluding zero on both arms. But the *widening* prediction fails: the gap is **+0.270 on long
    items against +0.296 on short ones** — flat-to-slightly-narrower, the opposite of what was
    predicted, with n = 11 and overlapping CIs.
  - **H-W31e — descriptive.** The carrying heads mostly read the span positions **directly**:
    13 of the top 16, ρ ≈ 0.45.

- **Observations.**
  1. **The question the write-site geometry forced is now answered.** The write lands on 1,459
     prompt positions and the score is summed over 32,586 reply tokens, so every nat had to cross
     through an attention head in layers 33–47. It crosses through **eight**, and mostly through
     **L41H4** and the late global layers 41 and 47 — which are exactly the layers that can see a
     span 1,024+ tokens back.
  2. **H-W31c is the most consequential line for the programme.** 98.3 % causal fidelity left open
     the possibility that the reconstructed vector was doing its work by a *different* route — a
     way the headline number could have been hiding something. It is not: ρ = 0.838, Jaccard 0.78.
     The channel is faithful in *where* it acts as well as in how much.
  3. **But the residue is systematic and points the same way every time.** C3pure is below P_patch
     at precisely the components that carry the most — L46H1, L34M, L41H4, L45H3, L47H2 — each with
     a CI excluding zero. The ~1.7 % fidelity loss is not spread evenly; it is a small, consistent
     shortfall **at the load-bearing heads**. That is the shape a slightly-off direction makes, and
     it is the most specific mechanical description of the loss the programme has produced.
  4. **Many components are mildly antagonistic.** Summed necessity is negative at nine of fifteen
     layers, and single-component sufficiencies sum to only 0.62–0.68 of the total. Reverting an
     individual component to its unsteered value *improves* the score more often than not — so the
     effect is a concentrated positive contribution net of a diffuse drag, not a sum of small helps.
  5. **The uniform-random null is near zero while the layer-matched null is not** (+4.26 vs +13.95 at
     k = 32). Depth alone buys a third of what the selected set buys, which is why LRAND was
     pre-registered; a design with only a uniform null would have overstated the localisation.

- **New questions / hypotheses.**
  - **H-W35:** does ablating the top-8 set *at L1b with no write* damage the model's clean-code
    performance? If these are general-purpose late-layer heads, the "circuit" is where information
    from any prompt edit arrives, not something specific to identifier semantics. This is the
    specificity control the localisation result needs before it is called a mechanism.
  - **H-W36:** the residue in observation 3 predicts that the AR's reconstruction error is
    **anisotropic** — larger along the directions L41H4 and L46H1 read. Measurable from the banked
    `vectors.npz` with no GPU.
  - **H-W37:** L34M is the second-strongest component and sits one layer above the write. Is it
    doing transport, or re-normalising a vector written at 32? Distinguishable by patching it in
    isolation on an unwritten run.
  - Phase B's layer choice is now informed: `CONCENTRATED` in late global heads → a second NLA layer
    belongs **just above 41**, not spaced by depth (`docs/nla_multilayer_scoping.md` §5).

- **Bounds.** One host, one layer, one obfuscation tier, 60 items. `suf`/`nec` are single-component
  interventions patched at all positions; the joint pass is greedy-by-sufficiency, not a search over
  sets, so "8 components" is an upper bound on the minimal set. The read-knockout materialises an
  attention mask for the layer under test, which moves that layer off the flash kernel — `ko_gap`
  measured this at exactly 0.0000, so the concern is closed rather than assumed away. H-W31d's
  long-item subgroup is n = 11 and its negative result should not be quoted as more than a
  non-replication of the prediction.
