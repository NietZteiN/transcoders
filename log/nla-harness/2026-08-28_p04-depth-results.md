### Target Date: 2026-08-28 (P0.4 results — depth is not the constraint either)

Pre-registration: [`2026-08-28_p04-depth-prereg.md`](2026-08-28_p04-depth-prereg.md), frozen
before a single arm existed. Nothing below renegotiates it. Phase-0 context:
[`2026-08-28_p0-triage-results.md`](2026-08-28_p0-triage-results.md).

- **Hypotheses / what we tested:**
  - **H-P04a (primary).** Is the belief-injection failure a *depth* failure? P0.1 found cross-item
    coherence peaks at **L13 (0.543)** while magnitude peaks at **L20 (0.459)**, the two directions
    nearly orthogonal (cos 0.21) — yet every intervention in this programme has been written at 20
    because that is where the released NLA pair was trained. If depth is the constraint, the
    **oracle** — which holds the item's own true−decoy difference and so cannot fail for lack of
    content — should recover accuracy at 13 and not at 20.
  - **H-P04b.** If it fires at 13, does it fire at **6** as well — band or point?
  - **H-P04c.** Does the no-NLA bar (V3) move with depth?

- **Setup:**
  ```
  runner    nla/src/steer_run.py --layer {6,13,20} --positions last_prompt
            --alphas 0.25,0.5,1.0,2.0,4.0 --only-conditions V3_taskvec,V4_oracle,R_random
  corpus    the same 60 L0/L1b pairs as B4 / P0.1 / P0.2 (dataset_a + dataset_b)
  model     Qwen/Qwen2.5-7B-Instruct, bf16, greedy · seed 20260724
  scale     3 arms x 960 rows = 2,880 rows, 0 errors · 1.59-1.71 h per arm
  slurm     array 357717 (h200): task 0 -> L06 g-08-09 · 1 -> L13 g-08-11 · 2 -> L20 g-08-13
            gate job 357716 g-08-09
  out       data/nla/p0/p04/L{06,13,20}/ · verdict data/nla/p0/p04/p04_verdict.json
  sha256    steer_run.py 4062251c151d8c37 · p04_gate.py cd8b0bb2c3a22047
            p04_score.py de2600dcde13eb35 · p04_sbatch.sh 3e6ace63bf9c678f
  ```
  New this run: `--layer` on `steer_run.py` (read site and write site move together), plus
  `p04_gate.py` and `p04_score.py`. The resume key now carries the layer — without it an L13 run
  silently adopts the 420 banked L20 rows as "done" and reports L20 numbers under an L13 label,
  the identical hazard already documented for `--positions`. `steer_run` refuses `--layer != 20`
  together with any AR-derived condition: the reconstructor is trained at 20, so V1/V2/F/A are
  not omitted for cost, they are undefined off-layer.

- **Results:**

  **Gate 1 — layer indexing. PASS.** The read site and the write site are the same decoder block,
  which the codebase has asserted in comments since `steer.py` was written and never once tested.

  | ℓ | fp32 residual | positions written | other positions moved |
  |---|---|---|---|
  | 6 | **8.48e-08** | 1 (t = 69) | 0.0 |
  | 13 | **8.82e-08** | 1 (t = 69) | 0.0 |
  | 20 | **4.33e-08** | 1 (t = 69) | 0.0 |

  The bf16 pass sits at 3.68-3.82e-03 across all three layers — the 8-bit-mantissa floor
  (2⁻⁸ = 3.9e-03), which is why the gate runs fp32. Indexing is a question of *which tensor*, not
  of precision.

  **PRIMARY — H-P04a: NOT DEPTH-LIMITED.** V4_oracle, layer 13 versus layer 20, α = 1.0, paired
  on all 60 items:

  | criterion | value | threshold | passed |
  |---|---|---|---|
  | (a) beats L20 | **−0.0667** [−0.1833, +0.0500], McNemar p = 0.424 | ≥ +0.10 | ✗ |
  | (b) beats its own baseline | **0.0000** [−0.1000, +0.1000] | ≥ +0.10 | ✗ |
  | (c) beats `R_random` | **0.0000** [−0.1167, +0.1167] | ≥ +0.10 | ✗ |
  | (d) parse rate intact | **0.8333** | ≥ 0.80 | ✓ |

  **The sharpest number in the table is V4@13's Δaccuracy of exactly 0.0000 — 5 recovered, 5
  damaged.** The oracle, holding perfect information about the true semantics, written at the
  depth where the task direction is *most* coherent, produces a pure shuffle: it trades five
  right answers for five wrong ones and nets nothing.

  **This is a null, not a destruction — and that distinction is the whole point.** P0.2's negative
  came with parse rates collapsing to 0.383 (V4) and 0.100 (V1); the channel over-delivered and
  broke generation before it could carry content. Here parse rates run **0.82-0.95 across all 45
  cells**. Generation is intact. The write lands, the model keeps answering, and the answers are
  no better.

  **Criterion (a) positively excludes the pre-registered effect.** The CI upper bound is +0.05,
  below the frozen +0.10. This is a bounded negative, not a failure to find something.

  **The whole dose-response surface is flat.** Δaccuracy (parse rate) — every cell within ±0.133
  of zero, no monotone structure anywhere, `R_random` wandering as much as the oracle:

  | arm | α=0.25 | α=0.5 | α=1.0 | α=2.0 | α=4.0 |
  |---|---|---|---|---|---|
  | L06 / V4_oracle | +0.0667 (.88) | +0.0000 (.85) | +0.0500 (.85) | −0.0333 (.88) | +0.0667 (.87) |
  | L06 / V3_taskvec | −0.0167 (.88) | +0.0500 (.88) | +0.0500 (.83) | +0.0167 (.88) | +0.0167 (.88) |
  | L06 / R_random | −0.0500 (.90) | −0.0333 (.90) | +0.0167 (.85) | −0.0667 (.90) | −0.0167 (.88) |
  | L13 / V4_oracle | −0.0333 (.92) | +0.0167 (.93) | **+0.0000** (.83) | +0.0667 (.92) | +0.0167 (.90) |
  | L13 / V3_taskvec | −0.0167 (.92) | +0.0333 (.93) | +0.0167 (.90) | +0.0333 (.85) | −0.1333 (.90) |
  | L13 / R_random | +0.0333 (.90) | +0.0333 (.88) | +0.0000 (.82) | +0.0500 (.92) | +0.0667 (.90) |
  | L20 / V4_oracle | +0.0500 (.92) | +0.0167 (.93) | +0.0833 (.92) | −0.0833 (.93) | +0.0500 (.88) |
  | L20 / V3_taskvec | +0.0333 (.87) | +0.0333 (.93) | +0.0167 (.87) | −0.0167 (.92) | +0.0167 (.95) |
  | L20 / R_random | −0.0167 (.87) | +0.0667 (.93) | +0.0167 (.88) | +0.0000 (.88) | +0.0167 (.92) |

  **H-P04b: NOT EVALUATED**, by the rule's own terms — the band/point question is defined only
  when the primary returns DEPTH-LIMITED. Stating it otherwise would invite reading a distinction
  into two arms that both did nothing.

  **H-P04c: V3 is NOT DEPTH-LIMITED at either layer.** L13 vs L20 **+0.0167** [−0.0833, +0.1167];
  L06 vs L20 **+0.0500** [−0.0500, +0.1500]. No incoherence flag: V3 did not clear the rule where
  V4 failed, so the oracle-dominates-taskvec ordering is intact.

  **One secondary-alpha lead, explicitly NOT banked.** At **α = 2** criterion (a) fires:
  V4@13 − V4@20 = **+0.1667** [+0.0667, +0.2667], McNemar p = 0.00635, and it **survives BH-FDR
  across the alpha family at q = 0.0254** — the first secondary alpha in this programme to do so.
  It is not a result, for a reason visible in the table above: the contrast is driven by
  **L20 falling to −0.0833**, not by L13 rising (+0.0667). It is a difference between two nulls.
  Criteria (b) and (c) both fail there. Recorded in the same spirit as B4's V2@α=4 lead.

  **Gate 2 — cross-cluster reproduction: FAIL, and the failure is itself the finding.**

  | condition | per-item agreement | juno Δacc | banked Δacc | gap | passed |
  |---|---|---|---|---|---|
  | V3_taskvec | 0.9000 | +0.0167 | +0.0500 | 0.0333 | ✓ |
  | V4_oracle | **0.8333** | +0.0833 | +0.0833 | **0.0000** | ✗ (agreement < 0.90) |

  **V4's aggregate reproduces exactly — both arms +0.0833 — while 10 of 60 individual items
  disagree.** The population statistic is stable across A6000/torch 2.9 → H200/torch 2.11; the
  per-item outcome is not.

  **The same effect appears between juno nodes.** The unsteered baseline is layer-independent by
  construction and the three arms should agree item for item. They do not:

  | arm | L1b accuracy | n wrong | per-item agreement vs L06 |
  |---|---|---|---|
  | L06 (g-08-09) | 0.5500 | 27 | 1.00 (ref) |
  | L13 (g-08-11) | 0.5500 | 27 | **0.90** |
  | L20 (g-08-13) | 0.5333 | 28 | 0.95 |

  L06 and L13 have **identical marginal accuracy and six disagreeing items.** Greedy decoding in
  bf16 is reproducible in aggregate and not per item — non-associative reduction order and
  autotuned kernel selection differ across physical cards.

- **What worked / hypothesis verdict:**
  - **H-P04a ✗ REFUTED — NOT DEPTH-LIMITED.** Depth is not the constraint. Writing the oracle at
    the most coherent depth in the network changes nothing, with generation fully intact.
  - **H-P04b — not evaluated** (undefined given the primary).
  - **H-P04c ✗ — V3 does not move with depth either.**

  **Phase 0's licensing row 4 now has a third independent leg.** The channel is not the bottleneck
  (P0.2), the site is not inert (P0.3), and the depth is not wrong (P0.4). Three different ways of
  blaming the *apparatus* have each been pre-registered and each been closed off. What remains is
  the reading the programme has been circling: **there is no item-level belief at this site to
  edit.** This is the strongest form of the Phase-1b readout paper's central negative, and it was
  bought with 2,880 rows and a frozen rule rather than an argument.

- **Observations:**
  - The oracle's exact 0.0000 with a 5/5 recovered-damaged split is worth quoting verbatim in the
    paper. A steering direction that recovers five items and destroys five is not a weak effect;
    it is the signature of a perturbation that carries no task-relevant information at all, and
    the balanced-Δaccuracy statistic is precisely what makes it visible. Flip rate would have
    reported "5 recoveries" and looked like a result.
  - **P0.4 differs from P0.2 in kind, not degree.** P0.2 found destruction (parse 0.100); P0.4
    finds indifference (parse 0.83-0.95). A theory of the failure has to account for both: the
    write lands, it is large enough to break generation when applied broadly, and it is
    nonetheless semantically inert.
  - **The +0.10 threshold bounds what this null means.** With a 5-10% per-item cross-node noise
    floor, P0.4 excludes a +0.10 depth effect and cannot exclude a +0.05 one.
  - Re-running L20 on juno rather than quoting the bank was the right call and would have been
    hard to justify after the fact: the cross-cluster per-item agreement is 0.833 for the primary
    condition, so a depth contrast against banked rows would have been confounded at roughly twice
    the size of the effect it was looking for.
  - The `--layer` resume-key hazard was caught by analogy rather than by testing — the
    `--positions` bug was documented in `01_RESUME_PHASE0.md` and the same shape applied one level
    down. Reading the previous session's bug notes paid for itself.

- **New questions / new hypotheses:**
  - **The per-item irreproducibility is a live threat to every paired analysis in this programme.**
    B4's McNemar tables, N13's matched 2×2, and P0.4's own criterion (a) all treat per-item
    correctness as a fixed property. It is not: it is 0.83-0.95 reproducible across hardware.
    Aggregates are safe; per-item pairings inherit a noise floor nobody has measured on purpose.
    **Proposed check (cheap, no new science): re-run one arm twice on the same node and again on a
    different node, and quote the two agreements.** That separates card-to-card nondeterminism
    from run-to-run nondeterminism and gives every future McNemar table an honest floor.
  - Does the α = 2 collapse at L20 (V4 −0.0833) mean anything? It is the only cell where the
    oracle is meaningfully negative, and L20 is the one layer with high relative magnitude
    (0.0675). A magnitude-driven disruption threshold would predict exactly that. One cell,
    not banked, but it is the only structure in an otherwise flat surface.
  - P0.4 tested depth for a **write**. It says nothing about depth for a **read** — the readout
    paper's own question. L13's higher coherence may still make it the better place to *decode*
    from even though it is no better to *write* to. That is a Phase-1b experiment and it is now
    the interesting one.

- **Next Steps:** fold P0.4 into the Phase-1b framing; run the reproducibility floor check before
  any further paired per-item claim; P0.3-ext remains blocked on `adversarial_rename`, which lives
  only on csr-94608.
