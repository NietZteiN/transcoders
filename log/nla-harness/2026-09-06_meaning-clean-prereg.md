### Target Date: 2026-09-06 (H-W17 — the clean nonsense-vs-truth contrast, and a correction to what I claimed it would fix)

Follows [`2026-09-06_tier-source-results.md`](2026-09-06_tier-source-results.md), where H-W16b's
verdict was found to be an artifact of `tier_anchor()`'s fallback. **Committed before the run.**

#### Correction, filed before running: my proposed fix does not do what I said it does

That entry proposed H-W17 as:

> restrict `T_L1_sub` and its matched `T_L0_sub` to the **131 `rename_map`-anchored spans only** —
> a *span-level* restriction, not the item-level selection used post-hoc here, **so it does not
> inherit the selection bias**.

**The clause in bold is false, and I checked it before writing the code rather than after.** All
**131** `rename_map`-anchored spans live in **exactly the same 20 items** (6.55 spans each); the
other 29 items contain **zero**. A span-level restriction therefore lands on precisely the item set
the post-hoc analysis used. **The confinement to 20 of 49 items is a property of the stimuli — L1
renaming is thorough on 20 items and absent on 29 — not of the analysis**, and no restriction rule
can escape it.

What H-W17 can still do is worth the job: turn a **post-hoc** number into a **pre-registered** one,
remove the 21 unrenamed spans (14 %) still diluting the 20-item estimate, and **measure how far the
20 differ from the 29** so the bound is stated rather than assumed.

- **Hypotheses.**
  - **H-W17a (the contrast, at last uncontaminated):** `T_L0_ren − T_L1_ren` ≥ **+12.11** nats with
    a CI excluding 0, over the **131 renamed spans**, every arm writing at exactly those positions.
    **CONFIRM → identifier meaning carries a real part of the item component**, and the
    "removal suffices" reading is refuted. **REFUTE → nonsense repairs as well as truth**, and the
    mechanism is interference removal.
  - **H-W17b (representativeness, descriptive, frozen now):** the 20 anchored items vs the 29
    others on five properties fixed in advance — `G_sum` unit, baseline L0 accuracy, prompt token
    length, spans per item, language (py/js). **If they differ on ≥ 2 of the 5 at q < 0.05
    (BH-FDR, Mann–Whitney), H-W17a's result is labelled `corpus-bounded` in every later citation.**
    This does not gate H-W17a; it fixes the caveat before the estimate exists.

- **Setup (frozen).** `nla/src/nla_tiers.py --strict-anchor`, which disables the `tier_anchor`
  fallback so only `rename_map`-anchored spans qualify. Arms `T_L0_ren`, `T_L1_ren` at the identical
  131 positions, plus `SELF` (re-asserted, `|ΔG| ≤ 1.0`, **exit 3** on failure) and the unsteered
  noop. Generation for the veto on `T_L1_ren` and noop. `G_sum`; paired bootstrap 10,000; seed
  20260724; `--deterministic` OFF; host `gemma12b` L32/48.

- **Expected effect, stated in advance.** The post-hoc 20-item estimate was **+21.48** (ratio 0.584)
  including 14 % diluting spans; removing them should move the estimate **up**, so H-W17a is
  expected to clear. **I record that this is now the direction I expect — the opposite of what I
  pre-registered on 2026-09-06 for H-W16b — and that I have already been wrong once in this exact
  hypothesis, in the direction of my own prior.** If it clears, the correct framing is
  *"meaning and removal both contribute"*, never *"meaning wins"*: `T_L1_ren` is expected to remain
  a large positive, which is the removal component and is not diminished by this result.

- **Bounds.** n = 20 items, the smallest denominator any W result has rested on, and a
  stimulus-selected subset. **A single job cannot fix that**; only stimuli with thorough L1 renaming
  across the corpus could. Every citation of H-W17a carries n = 20.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
