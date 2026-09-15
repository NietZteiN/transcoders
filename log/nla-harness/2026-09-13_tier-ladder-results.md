### Target Date: 2026-09-13 (the prompt-space tier ladder — `ERASURE-FLOOR` / `FLATTENING-PENALTY` / `ROUTES-COMPOUND`)

- **Hypotheses / what we're testing:** the ladder L0/L1/L1b/L2/L3 measured **in prompt space on this host**, which
  fixes the ceiling for every erasure lever the programme has or plans — NLA `edit`, the held cross-item erasure
  vector, attention masking, CodeSteer-style reallocation alike — because **L1 is perfect erasure done in token
  space** (nonsense names, original structure, no trap). Rules frozen in
  [`2026-09-12_tier-ladder-prereg.md`](2026-09-12_tier-ladder-prereg.md), all on paired per-item **sampled pass
  rate** (n = 8):
  **H-A1** `ERASURE-HEADROOM` if `rate_L1 − rate_L1b` > 0 with CI excluding 0, else **`ERASURE-FLOOR`** ·
  **H-A2** `FLATTENING-PENALTY` if `rate_L0 − rate_L2` > 0 with CI excluding 0, else `FLATTENING-FLAT`
  (Paper 3's **OR = 0.57** predicts `rate_L2` ≈ 0.448, i.e. Δ ≈ **+0.139**) ·
  **H-A3** `ROUTES-COMPOUND` if `rate_L3 − rate_L2` < 0 with CI excluding 0, else `ROUTES-NOT-ADDITIVE`.
  **Predictions recorded:** both headroom shares small · PENALTY · **NOT-ADDITIVE**.

- **Setup:** job **392094**, `sbatch nla/scripts/nla_tier_ladder_sbatch.sh` (sha256 `359cda03…`), node **g-07-03**
  (H200 NVL), `2026-09-13T05:26:02Z → 07:42:34Z`, **2:16:34 elapsed, rc=0, 2.28 GPU-h** (prereg estimated 3–4).
  `nla/src/nla_tier_accuracy.py` sha256 `66a6f8930c48a4f564c741551a40b242275c6b6cea8af2522f5ceecc8dc172e3`.
  Host Gemma-3-4B-it, **no steering, no AV/AR, no vectors** — forward generation only. Same 60 items as every run
  in this family (`select_pairs`, seed 20260724), tier rows via `nla_tiers.tier_rows` from
  `data/stimuli/dataset_{a,b}/`, truth = the tier row's own `expected_output` asserted equal to the L0 truth.
  Grading `steer_run.graded`, `MAX_NEW_GEN` 1100, `--deterministic` OFF, N_BOOT 10 000, seed 20260724.
  Sampling n = 8 via `num_return_sequences` at **batch 1, no padding**, identical to the steering runner so rates
  are comparable across the two. In-job pytest `nla/tests/test_tier_accuracy.py` passed; smoke (2 items) passed
  its own identity gate 4/4 before the full run.
  **Identity gate: 120/120 rebuilt L0 and L1b prompts bit-identical to `l0_prompt_ids` / `l1b_prompt_ids` in the
  banked traces.** Skips: `no_call` 0, `truth_mismatch` 0, `no_tier` 0 — all 60 items contributed all 5 tiers.
  **Banked sanity exact:** greedy `acc_L0` **0.5667** vs banked 0.567 and `acc_L1b` **0.5333** vs banked 0.533,
  both inside the 9/60 churn band (in fact identical to 3 dp).
  Output `data/nla/ml/gemma4b/gate/tier_ladder/{tier_rows.jsonl,tier_stats.json}`.

- **Results:** per tier, 60 items:

  | tier | greedy acc | Wilson 95 % | sampled rate | boot 95 % | greedy parse rate |
  |---|---|---|---|---|---|
  | **L0** original | 0.567 (34/60) | [0.441, 0.684] | **0.588** | [0.477, 0.694] | 0.883 |
  | **L1** nonsense rename | 0.567 (34/60) | [0.441, 0.684] | **0.504** | [0.394, 0.613] | 0.800 |
  | **L1b** adversarial rename | 0.533 (32/60) | [0.409, 0.654] | **0.540** | [0.425, 0.654] | 0.817 |
  | **L2** flattening | 0.533 (32/60) | [0.409, 0.654] | **0.498** | [0.388, 0.608] | 0.817 |
  | **L3** stacked | 0.433 (26/60) | [0.316, 0.559] | **0.438** | [0.335, 0.546] | 0.700 |

  Paired per-item contrasts on sampled rate (the frozen readout):

  | contrast | what it bounds | value |
  |---|---|---|
  | `L1 − L1b` | **everything an erasure lever can recover** | **−0.035 [−0.100, +0.027]** |
  | `L0 − L1` | what only **reconstruction** (installing true meaning) adds | **+0.083 [+0.015, +0.160]** |
  | `L0 − L2` | the flattening route | **+0.090 [+0.021, +0.160]** |
  | `L3 − L2` | does the decoy cost extra on top of flattening | **−0.060 [−0.110, −0.015]** |
  | `L0 − L1b` | the whole adversarial gap | +0.048 [−0.025, +0.127] |
  | `L1b − L2` | route vs route | +0.042 [−0.033, +0.121] |

  Greedy McNemar vs L1b: L0 7 fixed / 5 broken p = 0.774 · L1 5/3 p = 0.727 · L2 7/7 p = 1.000 ·
  L3 5/11 p = 0.210.

- **What worked / hypothesis verdict:**
  - **H-A1 → `ERASURE-FLOOR`.** `rate_L1 − rate_L1b` = **−0.035 [−0.100, +0.027]** — not > 0, CI contains 0, and
    the point estimate is *negative*. **Prediction (both shares small) SUPPORTED in magnitude**, and the verdict
    is the floor branch. **The decisive number in this entry is the pair:** of the L0→L1b gap, **erasure bounds at
    −0.035 (unresolved, possibly nil) while reconstruction bounds at +0.083 with the CI excluding 0.** All of the
    accuracy headroom on this host is in *installing the true meaning*, none of it in *removing the decoy*.
  - **H-A2 → `FLATTENING-PENALTY`, prediction SUPPORTED.** `rate_L0 − rate_L2` = **+0.090 [+0.021, +0.160]**.
    Paper 3's OR = 0.57 predicted **+0.139**, which sits inside the CI — **the behavioural flattening effect
    replicates on this host within noise**, so Step 4 (dispatcher-state reads) does not have to move hosts.
  - **H-A3 → `ROUTES-COMPOUND`. My prediction `ROUTES-NOT-ADDITIVE` is REFUTED.** `rate_L3 − rate_L2` =
    **−0.060 [−0.110, −0.015]**, CI excluding 0; greedy agrees and is larger (0.433 vs 0.533 = −0.100). I argued
    NOT-ADDITIVE from "the L1b penalty alone is within churn", i.e. from a null on one route to no interaction —
    **the same inference error as `DATA-SATURATED`** (insensitivity on one axis is not evidence about another).
    Twice in this family now; the pattern is generalising from an underpowered null.

- **Observations:**
  - **The two readouts disagree about L1, and the more sensitive one wins the verdict.** Greedy puts L1 at 0.567
    — equal to L0 and *above* L1b — while the 8-sample rate puts it at 0.504, *below* L1b. Greedy at n = 1 on 60
    items moves in 2-item quanta (0.033), so the greedy "L1 > L1b" is two items wide; the rate readout integrates
    480 generations per tier. The rule was pre-registered on the rate for exactly this reason, and it is the
    honest resolution — but the disagreement is itself the caution: **nothing in this ladder separates L0, L1, L1b
    and L2 at n = 60 by greedy accuracy at all.**
  - **Post-hoc diagnostic (descriptive, no rule, reported because it reframes H-A2/H-A3): most of the tier
    gradient is unparseable output, not wrong answers.** Conditioning on the greedy answer parsing at all:

    | tier | items parsed | greedy acc \| parsed | rate \| parsed | sampled parse rate |
    |---|---|---|---|---|
    | L0 | 53/60 | 0.642 | 0.646 | 0.867 |
    | L1 | 48/60 | **0.708** | 0.628 | 0.771 |
    | L1b | 49/60 | 0.653 | 0.651 | 0.785 |
    | L2 | 49/60 | 0.653 | 0.610 | 0.760 |
    | L3 | 42/60 | 0.619 | 0.589 | 0.698 |

    The parsed fraction falls monotonically 0.867 → 0.698 while accuracy *among parsers* is flat within noise
    (0.619–0.708, and L1 is the **highest** of any tier). So on this host the ladder is largely a
    **response-budget / formatting** effect — the model more often fails to emit a parseable answer inside the
    1100-token budget as obfuscation increases — rather than comprehension that completes and lands wrong. This
    converges with the charter's ~2 048-token System-2 plateau anchor. **Caveat stated with it:** conditioning on
    parse success is a collider, so the parsed-only column is not a clean causal contrast; it is a diagnostic
    pointing at where the variance lives, not a corrected effect.
  - **Consequence for job 391968, which was still running when this landed:** the accuracy run's `edit_band`,
    `edit_single` and `foreign_band` arms are erasure-shaped and this ladder bounds them at **≈ 0 (−0.035
    [−0.100, +0.027])**. Only `c3_band` — the transported clean state, which installs meaning — has real headroom
    above it (+0.083 [+0.015, +0.160] from `L0 − L1`). That is a pre-existing bound, derived on a separate run
    and recorded **before** 391968 reported; it is not a post-hoc explanation of whatever that job shows.
  - Telemetry clean: identity 120/120, zero skips of any kind, banked greedy accuracies reproduced to 3 dp.

- **New questions / new hypotheses:**
  - **H-A4 (free, no GPU):** is the ladder a comprehension effect or a budget effect? Re-score the banked rows with
    (a) parse failure counted as abstention vs (b) excluded, and regress `*_parsed_rate` on `*_prompt_len` — L3
    prompts are the longest and the parse rate is lowest. If length explains it, the tier "penalty" that Papers 2–3
    measure is partly an output-budget artifact on small hosts, which is a claim about the *measurement*, not the
    model, and belongs in the paper.
  - **H-A5:** raise `MAX_NEW_GEN` to 2 048 on the L3 tier only (~0.5 GPU-h) — if L3's rate recovers toward L2's,
    H-A3's `ROUTES-COMPOUND` is a truncation artifact and must be re-reported. **This is the most decision-relevant
    cheap follow-up in the entry** and should run before `ROUTES-COMPOUND` is cited anywhere.
  - **H-A6:** the reconstruction-vs-erasure split (+0.083 vs −0.035) is the programme's steering target restated.
    The held erasure-vector run ([`2026-09-12_erasure-vector-prereg.md`](2026-09-12_erasure-vector-prereg.md))
    measures `G_sum`, not accuracy, so it stays valid as written — but its accuracy add-on, if ever built, is
    pre-bounded at nil, and the reconstruction arm (`c3`-shaped writes) is where an accuracy lever could exist.
  - Does the L1-below-L1b inversion on sampled rate survive a second seed? It is the only sign inversion in the
    ladder and it decides whether "adversarial is worse than nonsense" holds here at all.

- **Next Steps:** report **391968** (accuracy) when it lands — it was at item 48/60 as this was written — then
  decide the held erasure run. Run **H-A5** (0.5 GPU-h) before `ROUTES-COMPOUND` is used; **H-A4** costs nothing.
