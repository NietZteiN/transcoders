### Target Date: 2026-09-04 (W stage 0 — the NLA round trip is span-specific: `NLA-LIVE`, and raw cosine was hiding it)

**Thread:** nla-harness · **Job:** 376950 (h200, g-08-01, 25 min, 0 errors) · **Host:** `gemma12b` L32/48 ·
**Resolves:** [`2026-09-03_nla-writeback-prereg.md`](2026-09-03_nla-writeback-prereg.md) stage 0 and
[`2026-09-04_cycle-gate-amendment.md`](2026-09-04_cycle-gate-amendment.md) H-W0c.

- **Hypotheses / what we're testing:** H-W0a (round trip preserves item identity), H-W0b (it
  distinguishes *this* identifier from another identifier in the same snippet), H-W0c (both survive
  mean-centring), H-W3 (descriptive: do the reads mention the decoy field more than the true one).

- **Setup:** `nla/src/nla_cycle.py` sha256 in the job log; `nla/src/local_av.py` (in-process AV — no
  sglang, see the amendment). Subject `google/gemma-3-12b-it` L32; AV/AR
  `kitft/nla-gemma3-12b-L32-{av,ar}`, `injection_scale 80000`, `mse_scale 61.9677`,
  `MAX_NEW_READ 180`. Banked 60 L0/L1b pairs, `load_pairs(None, Random(20260724))`. Span vector =
  the span's **last** token. Cluster bootstrap over **items**, 10,000 resamples, seed 20260724.
  `--deterministic` OFF. Artifacts `data/nla/p0/nla_cycle/gemma12b/{cycle_stats.json,
  cycle_rows.jsonl, cycle_vectors.npz}`.

- **Results.** 60 items, **476 spans** (7.9/item), **0 skipped**, **CJK 0.000**, **0 reads without
  closing tags** (the 96→180 fix held).

  | | raw cosine (primary, frozen 09-03) | mean-centred (H-W0c, frozen 09-04) |
  |---|---|---|
  | **matched** | **+0.9877** [+0.9873, +0.9882] | **+0.6807** [+0.6696, +0.6916] |
  | within-item mismatched | +0.9669 [+0.9659, +0.9679] | **+0.0525** [+0.0420, +0.0639] |
  | cross-item mismatched | +0.9646 [+0.9641, +0.9651] | **−0.0011** [−0.0100, +0.0077] |
  | W0a / W0b separation | ✓ / ✓ | ✓ / ✓ |

  **The two metrics agree → no `W0-UNRESOLVED`. Verdict `NLA-LIVE`; stage 1 is licensed.**

  H-W3 (descriptive, no gate), rate of reads mentioning each field: **decoy 0.489** [0.430, 0.548]
  vs **true 0.342** [0.262, 0.426].

- **What worked / hypothesis verdict.**
  - **H-W0a ✓ SUPPORTED** on both metrics.
  - **H-W0b ✓ SUPPORTED** on both, and this is the one that licenses stage 1: centred, a read's
    reconstruction sits at **+0.68** with its own span and **+0.05** with a *different span in the
    same snippet*. The round trip is not encoding "this snippet" — it is span-specific, so a
    per-span edit has a real referent.
  - **H-W0c ✓ SUPPORTED, and it strengthened rather than rescued the result.**

- **Observations.**
  - **The anisotropy correction went the opposite way from my stated worry.** The amendment was
    written expecting centring might *dissolve* a separation that raw cosine had manufactured.
    Instead raw cosine was **compressing a large effect into 0.02 of range**: matched-vs-cross goes
    from a 0.023 gap to a **0.68 gap**. The strongest evidence that centring is the right metric
    rather than a favourable choice is the null itself — **cross-item centred cosine is −0.0011,
    with a CI containing zero**, which is exactly what an uninformative comparison should produce and
    is not something a metric tuned to flatter the result would give.
  - **This bears on every `rt_cos` in the project.** `NLACritic.score` L2-normalises but does not
    centre, so every banked round-trip cosine — `ReadEngine.rt_cos`, the B6 length-vs-faithfulness
    work, the malware read stats — lives in the compressed 0.9+ regime where real differences are
    ~0.02 wide. Nothing banked is *wrong*; the differences may be far larger than they looked.
  - **H-W3 points the way the trap predicts, at the reported precision.** Reads mention the decoy
    field more than the true one (0.489 vs 0.342). Post-hoc and **not** pre-specified, the paired
    breakdown is: decoy-only **0.300**, true-only **0.153**, both 0.189, neither 0.357 — decoy-only
    is about twice true-only. A paired test is the right one and was not pre-registered; it is
    recorded here as a lead, not a result.
  - The verbalizer describes these positions in *surface* terms — "a function signature/docstring is
    being presented", "Code block structure signals a Python context" — and reaches the identifier
    itself mainly by quoting it (`The phrase "def get_bounds"`). Whether that is enough semantic
    grip for stage 1's edit to bite is precisely what stage 1 tests.

- **New questions / new hypotheses.**
  - **H-W4:** re-express the banked `rt_cos` corpus as **centred** cosine. No GPU where the vectors
    were persisted. Could change effect sizes across several finished analyses.
  - **H-W5:** the reads are dominated by surface/positional description. If stage 1 nulls, the
    diagnostic question is whether the *edit* failed or the read never carried editable semantics —
    H-W3's decoy-only rate (0.300) bounds how many spans could even be edited.

- **Next Steps:** stage 1 — read → deterministic single-term edit → AR → norm-matched **replace** at
  the span, read on `G_sum`, with C1 (unedited round-trip), C2 (foreign edit) and C3 (NLA round trip
  of the matched L0 state) as pre-registered.
