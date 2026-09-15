# 2026-09-11 · PRE-REGISTRATION — layer-count HP search, head localisation, accuracy (H-S1…H-S4)

**Thread:** nla-harness · **Experiment:** E-S (E2 steering + head attribution on the 4B all-layer pairs)
**Status:** pre-registered, nothing run. Thresholds below are frozen; changes after the first submission go in a new
dated entry.
**Builds on:** [`2026-09-10_multilayer-gate-results.md`](2026-09-10_multilayer-gate-results.md) (33 live 4B pairs,
`M-NO-GAIN`), the H-W31 head-mediation design already implemented in `nla/src/nla_heads.py` + `nla/src/head_patch.py`,
and the flippable census [`2026-09-07_flippable-census-results.md`](2026-09-07_flippable-census-results.md).

## The objective, and one honest restriction up front

Request: sweep how many layers to steer, find which attention heads carry the effect, and try to improve accuracy.
The first two are well-powered on this corpus. **The third is not, and that is already established rather than a
guess.** The flippable census found Gemma yields **6/60 flippable items**, so a *perfect* rescue is worth +0.100
accuracy; it also recorded a gate firing PASS at exactly 9 flippable items and judged that "the pass is not real",
concluding the accuracy question needs "stimuli built to have a denominator". Our 4B host scores **acc_l0 0.567 /
acc_l1b 0.533** — obfuscation costs **2 items of 60**.
**Therefore: the primary readout is `G_sum` (teacher-forced log-prob of the banked L0 reply under the L1b prompt), the
same readout as the W family and the Phase B gate. Accuracy is reported as a SECONDARY, pre-declared-underpowered
measure (H-S4) and no accuracy claim will be made from it.** Declaring this now prevents a null accuracy result being
read as evidence against steering, or a 1–2 item wobble being read as success.

## Setup (frozen)

Host: 4B text ckpt, 33 live NLA pairs (L0 PAIR-DEAD, excluded). Stimuli: the same 60 items / 471 repaired spans as every
run in this family. Vectors: already banked at `data/nla/ml/gemma4b/gate/vectors/L{K}.npz` (34 layers, uniform at
60 items / 471 spans) — **no re-extraction needed**. Seed 20260724, N_BOOT 10 000, cluster bootstrap over items.
Arms reuse the frozen definitions: `c3` (AR(AV(h_L0)), the transport arm), `edit`, `rt`, `foreign`, `random`, `self`.

## H-S1 — layer-count HP search (primary)

Greedy top-k layers ranked by single-layer `dG_c3`, for **k ∈ {1,2,3,4,6,8,12,16,24,33}**, joint write (one
`PositionReplacer` per layer, norm-matched).
**Selection honesty:** ranking is done on one split half (`crc32(snippet_id) & 1`) and evaluated on the *other* half,
both directions, then pooled — so no layer set is chosen and scored on the same items. A **depth-spaced ladder** (k
layers evenly spaced over the live range) is run alongside as a selection-free control.
Verdicts on `M_c3` relative to the single best layer, in units of that layer's `S_swap`:
- **MORE-IS-BETTER** if the best k beats k=1 by ≥ 0.10·S_swap with CI excluding 0.
- **MORE-IS-WORSE** if k=33 is below k=1 by ≥ 0.10·S_swap with CI excluding 0.
- **SATURATES** otherwise (report the k at which 95 % of the maximum is first reached).
Phase B already found the *edit* arm gets worse with more layers (`M-NO-GAIN`, −4.66 to −11.24 nats); H-S1 asks the same
question of the **transport** arm, which Phase B measured at only three fixed sets.

## H-S2 — head localisation (primary)

At the best single write layer by transport (**L7**, `S_c3` 65.75; writing early maximises the downstream component
count: 26 layers × 8 heads + 26 MLPs = **234 components**), per-component **sufficiency** `suf_c = dG(U, c←S)` and
**necessity** `nec_c = dG(S) − dG(S, c←U)`, using the existing `head_patch.py` machinery.
- **CONCENTRATED** if a top-16 joint patch (6.8 % of components), selected on the opposite half, recovers ≥ 50 % of
  dG(S) **and** beats a layer-matched random-16 null by ≥ 30 points of the effect with CI excluding 0.
- **DISTRIBUTED** if no k ≤ 32 reaches 50 %.
- **INTERMEDIATE** otherwise (report the k at which 50 % is first crossed).
Identity checks, exit 3 on failure: every `SELF_c` within 0.05 nats of dG(S); `ALL` (all 234 components ← S) within 5 %
of dG(S).

## H-S3 — which heads correlate with output change (descriptive, no verdict word)

Spearman ρ between per-head `nec` and per-head **read-knockout** `read_h` (head h at layer L blinded to the span keys
only) — `nec − read` is the head's indirect share. Also the **global-vs-local** contrast: writing at L7, the downstream
global-attention layers are **11, 17, 23, 29** (4 of 26), and information must cross positions through attention above
the write, so global heads are predicted to carry more necessity. Reported as a paired contrast with CI; **no verdict
word**, because three of my four trend predictions in this project have been refuted by measurement.

## H-S4 — accuracy (secondary, pre-declared underpowered)

Generate answers under the best H-S1 configuration and report accuracy with Wilson CIs on (a) all 60 items and (b) the
**6–7 flippable** subset. Mandatory baselines per CLAUDE.md §4: **random** and **foreign** vectors (already in the
harness) plus a **prompting baseline** ("the identifiers may be misleading; reason about the code's behaviour"). Stated
in advance: a 2-item L0→L1b gap means the maximum detectable rescue is ≈ +0.03 overall / ≈ +0.10 on the flippable
subset, so **no accuracy conclusion will be drawn in either direction**; the measure exists to check steering does not
*harm* accuracy and to size a future purpose-built corpus.

## Cost

All stages reuse banked vectors and the 36 ms/forward measured on this host.
- H-S1: 10 k-values × 2 halves × ~4 arms × 60 items ≈ 4 800 forwards ≈ **3 min**.
- H-S2: 234 components × 2 (suf, nec) × 60 items ≈ 28 100 forwards ≈ **17 min**; + read-knockout 234 × 60 ≈ 14 000 ≈ 9 min.
- H-S2 joint: k-ladder × (top-k + 3 random + 3 layer-matched) × 60 ≈ 2 500 ≈ **2 min**.
- H-S4: 60 generations × 4 conditions at ~1 100 tokens ≈ **40 min**.
**Total ≈ 1.5 h on one GPU** — the whole experiment is cheaper than a single 12B training step sequence.

## Order

Prereg (this) → H-S1 layer sweep (cheapest, and its winner defines H-S2's write config) → H-S2/H-S3 head sweep →
H-S4 accuracy + baselines → results entry with the liveness/selection caveats stated first.
