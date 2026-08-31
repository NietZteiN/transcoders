### Target Date: 2026-08-31 (the relational read effect REPLICATES — with L3 weaker and one secondary check missed)

Pre-registration: [`2026-08-31_ladder-replication-prereg.md`](2026-08-31_ladder-replication-prereg.md),
frozen before any replication draw existed. Discovery:
[`2026-08-31_ladder-relational-positive.md`](2026-08-31_ladder-relational-positive.md).

- **Setup:**
  ```
  draws   array 359898, 5 NEW draws per tier (indices 5-9), same script and regime
          scored on draws 5-9 ONLY — discovery draws 0-4 are not pooled into the primary
  score   job 359981 · nla/src/p1b_ladder_replication.py
  stat    mean rho over all 28 layers minus each tier's OWN length baseline. Selection-free.
  ```

- **Results — PRIMARY: REPLICATED.**

  | | discovery (draws 0–4) | **replication (draws 5–9)** |
  |---|---|---|
  | relational mean | +0.1167 | **+0.0835** |
  | atom mean | −0.2006 | **−0.2359** |
  | **Δ (the frozen statistic)** | +0.3178 | **+0.3195** |
  | bar | — | **+0.15** |

  **Δ = +0.3195 against a bar of +0.15, and within 0.002 of the discovery value.** The relational
  mean is positive. Both conditions of the frozen rule are met.

  **Per tier, replication draws:**

  | tier | acc | length baseline | mean ρ | **beats length** | discovery |
  |---|---|---|---|---|---|
  | L0 clean | 0.657 | +0.3792 | +0.2776 | −0.1016 | −0.0593 |
  | L1 rename | 0.583 | +0.3872 | **−0.0873** | **−0.4745** | −0.3383 |
  | L1b adversarial | 0.570 | +0.2748 | +0.2774 | **+0.0026** | −0.0628 |
  | **L2 flattening** | 0.627 | +0.2533 | +0.3786 | **+0.1253** | +0.1395 |
  | **L3 stacked** | 0.567 | +0.2687 | +0.3105 | **+0.0418** | +0.0938 |

  **Permutation nulls (mean statistic, 200 draws):** L2 **p = 0.005**, L3 **p = 0.015**,
  L1b **p = 0.010**.

  **Pooled split-half over all ten draws** — the label-noise control that killed the depth
  gradient: **L2 both halves positive on 100% of splits** (+0.1275 / +0.1413, min +0.086);
  **L3 on 90%** (+0.0604 / +0.0785, min −0.023).

- **What worked / hypothesis verdict:**
  - **✓ PRIMARY REPLICATED.** The route contrast reproduces almost exactly on independent draws.
    **The residual stream at `last_prompt` carries item-level correctness information beyond reply
    length under relational obfuscation, and does not under atom-level obfuscation or on clean
    code.** This is the first positive in the programme to survive a pre-registered replication.
  - **L2 is the solid tier.** +0.1253 beats-length, p = 0.005, split-half positive on 100% of
    splits, and within 0.014 of its discovery value.
  - **L3 is weaker than discovery and should be reported as such**: +0.0418 against +0.0938, with
    one split-half draw going negative. Directionally consistent, p = 0.015, but **the stacked tier
    does not carry the claim on its own** — the effect is carried by L2.
  - **✗ A SECONDARY CHECK WAS MISSED, and it is reported rather than dropped.** The
    pre-registration said the per-tier nulls would be supportive if L2 and L3 reached p < 0.05
    **and L1b did not**. L1b reached **p = 0.010**. So the residual stream at L1b does carry
    correctness signal above chance — it just carries **nothing beyond reply length** (+0.0026).
    That is a real distinction and it sharpens the claim: the atom route is not representationally
    empty, it is **redundant with a token count**.
  - **L1 stayed signal-free** as required (mean ρ −0.0873, indistinguishable from zero at n = 60),
    so the pipeline did not change between runs.

- **Observations:**
  - **The claim, stated at the precision the data supports:** under control-flow flattening, a
    linear read of the residual stream predicts item-level correctness better than reply length
    does; under adversarial or nonsense renaming it does not, and under nonsense renaming the
    residual stream predicts nothing at all at any depth. The stacked tier points the same way with
    roughly half the effect.
  - **This gives the item-level null ledger its scope.** B4, B5, N11, N13 and P0.4 were all run on
    L0/L1b. Those are the clean anchor and the atom route — exactly where the residual stream is
    redundant with a token count. The nulls were not wrong; they were **route-specific and nobody
    knew it**, because the relational tiers had never been read.
  - **It fits the mechanism Papers 2–3 documented.** Relational obfuscation forces hidden-state
    simulation; a tracked state is *in* the residual stream in a way a lexical decoy is not. This is
    read-side evidence for the same account E3's attribution graphs were commissioned to test.
  - **Method note worth keeping.** Of five attractive positives this week, three died to controls
    (depth gradient, tier effect, L1b's argmax advantage) and one survived pre-registered
    replication. The controls that did the work were cheap: a second label draw, a per-tier
    baseline, a selection-free statistic, and a null over the statistic actually claimed.

- **New questions / new hypotheses:**
  - **What exactly is decodable at L2?** Dispatcher state, loop counter, hop index — a targeted
    probe turns a correctness correlation into a mechanistic claim and connects directly to E3.
    **This is now the most valuable experiment available.**
  - **Why is L3 weaker than L2?** The stacked tier adds renaming on top of flattening. If the atom
    route actively *destroys* the relational signal, that is the two-routes-compete-for-one-budget
    prediction from Paper 3, and it is testable by comparing L2 against L3 item by item.
  - The effect is at `last_prompt` — before any reasoning. Whether it grows or collapses across
    reply positions on L2 is the position × depth experiment again, on the tier where there is
    finally something to find.

- **Next Steps:** targeted probes for dispatcher state at L2; L2-vs-L3 item-level comparison.
  P0.3-ext still blocked on `adversarial_rename`.
