# 2026-08-14 — N12 / B4: the steering gate fails. V1 = V3 exactly, and prompting beats both.

**Goal / hypothesis.** Every prior measurement in this programme is observational. This is the
causal test: inject a direction at layer 20 of Qwen2.5-7B-Instruct and ask whether accuracy on
adversarially-renamed code recovers. **Pre-registered gate: V1 (NLA-derived) must beat V3 (a
contrastive task vector using no NLA at all).** Beating "no steering" proves only that a large
perturbation changes outputs.

**Setup.** 60 L0/L1b pairs from Dataset A+B with execution ground truth, `nla/src/steer_run.py`,
GPU 2, seed 20260724, 2.26 h, **420 rows / 0 errors**. Injection `h ← h + α‖h‖Δ̂` at the last
prompt token, α = 1.0. Baselines: **L0 0.60 / L1b 0.50** (banked reference L0 0.64–0.70), so 30
items are wrong at L1b — the recovery target. Scored by `nla/src/steer_stats.py` with balanced
Δaccuracy over *all* items, recovered and damaged reported separately.

**Results.**

| condition | acc | Δacc | 95% CI | recovered | damaged | McNemar p |
|---|---|---|---|---|---|---|
| **P_prompt** (prompt-only) | 0.600 | **+0.100** | [−0.017, +0.233] | 11 | 5 | 0.21 |
| V4_oracle (ceiling, not deployable) | 0.583 | +0.083 | [−0.033, +0.200] | 9 | 4 | 0.27 |
| **V1_gloss** (NLA) | 0.550 | +0.050 | [−0.083, +0.183] | 10 | 7 | 0.63 |
| **V3_taskvec** (no NLA) | 0.550 | +0.050 | [−0.050, +0.167] | 7 | 4 | 0.55 |
| A_antipodal | 0.533 | +0.033 | [−0.083, +0.150] | 8 | 6 | 0.79 |
| R_random | 0.533 | +0.033 | [−0.067, +0.133] | 6 | 4 | 0.75 |
| F_foreign | 0.483 | −0.017 | [−0.117, +0.083] | 4 | 5 | 1.00 |

**The gate, paired on the same 60 programs:** V1 0.550 vs V3 0.550, **delta exactly 0**,
discordant **7 vs 7**, McNemar **p = 1.00**. Verdict: **tie — the gate FAILS.**

**Verdict — B4 NOT SUPPORTED.**

1. **The NLA direction does not beat the no-NLA baseline.** Not "beats it narrowly" — an exact
   tie, with symmetric discordance (7 items each way). On this evidence the verbalizer and
   reconstructor buy nothing causal over a difference of activations that requires neither.
2. **Prompt-only is the best condition** (+0.100), above even the oracle. This is the AxBench
   result the project charter explicitly anticipates — prompting beats representation steering —
   now reproduced in-house on our own task. It is also the honest competitor any steering claim
   has to clear, and it wasn't cleared.
3. **The antipodal control is indistinguishable from random** (+0.033 vs +0.033). If V1 carried
   a real directional signal, reversing it should push the other way. It does not. That is the
   single most damaging number here, and it is independent of statistical power: it says the
   direction has no consistent sign, not merely a small effect.
4. **Nothing is significant.** Every CI spans 0; every McNemar p is 0.21–1.00.

**The interim reading was wrong, and the way it was wrong is now familiar.** At n = 34/condition
this looked like the pre-registered success: V1 +0.086 tracking the oracle +0.088, with V3 at
−0.029. At full n = 60 it is V1 +0.050, V3 +0.050. Three items of noise, reversed. This is the
same shape as N11's coupling cell — 3/3 becoming 3/10 once it had a denominator — and it is the
second time this month an encouraging partial result has evaporated at full n. **Do not read
partial-run tables as results.**

**A geometry check ran before the numbers and found no bug** — worth recording because it rules
out the obvious alternative explanation. `cos(V3, V4) = +0.418`: the task vector and the oracle
share direction, as they must, since V3 is the held-out mean of what V4 is per item. So V3 is not
sign-flipped or misconstructed. More interestingly, **V1 is near-orthogonal to both**
(`cos ≈ +0.02`, `+0.01`): the AR-reconstructed *language* direction and the empirical
*activation-difference* direction live in different subspaces. They are genuinely different
interventions, not two estimates of one thing — which is what makes the tie meaningful rather
than tautological. Norms (V1 70, V3 7, V4 13) cannot confound anything: `steer.py` normalizes Δ
to unit before scaling by α‖h‖, verified in source.

**Limitations, and they are real.**
- **Underpowered.** 60 items, 30 wrong; CIs are roughly ±0.12. This excludes a *large* effect,
  not a moderate one. The exact V1=V3 tie is the informative part; the individual Δaccuracies
  are not precisely estimated.
- **One α.** α = 1.0 injects a step the size of ‖h‖ — aggressive, and the controls show the
  model is sensitive at it (random alone flips items). A sweep was budgeted away. So the claim
  is "V1 does not beat V3 at α = 1.0", not "at any α".
- **One position** (last prompt token), one layer (20), one model, greedy, single seed.
- V2 (the paper's own read-edit protocol, Δ = AR(edited read) − AR(read)) was **not run** — it
  needs an AV call per item and the budget went to the V1/V3 contrast. V1 is a cheaper,
  read-independent form of the same idea, so a V2-specific effect is not excluded.

**Four bugs were fixed in this runner before it produced a number**, all caught by the control
condition rather than the treatment: a 400-token budget that truncated every reply before the
answer line (L0 scored 0/3); a normalization that differed from the established harness; a prompt
that never named the call, so the model invented its own inputs; and a re-derived call builder
that produced `get_bounds(get_bounds(3))` for a program defining `make_a_pile` — 17% of L0 rows
have an `fn_name` absent from their code. `task_bank.build_call` already handled every one of
those cases and is now imported instead. **Clean-code accuracy is the tripwire**: a task that
fails on unobfuscated code is broken, not hard.

**Next steps.** The causal question the programme was built to answer now has an answer, and it
is negative at this α: NLA-derived steering does not outperform a difference vector, and neither
outperforms a sentence in the prompt. Before spending more, the two cheap things that would
change the picture are (a) an **α sweep** — the single frozen α is the biggest unforced
limitation here — and (b) **V2**, the paper's own read-edit protocol, which remains untested.
Absent those, the honest summary of Instrument 2 is that it reads themes and does not steer.
