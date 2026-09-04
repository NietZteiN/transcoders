### Target Date: 2026-09-04 (Filed mid-run: the frozen +12.11 support threshold sits BELOW the banked random-noise benchmark)

Amends the reading — **not the rule** — of [`2026-09-03_nla-writeback-prereg.md`](2026-09-03_nla-writeback-prereg.md).
**Append-only.** Written while job **377399** is at **11 of 60 items**, from banked R2 data plus the
progress lines; the full stage-1 numbers, every control's mean, and all H-W2′ statistics do not
exist yet. The change made here makes a positive claim **harder**, never easier — which is the only
direction a criterion may be tightened mid-flight.

- **Hypotheses / what we're testing:** no hypothesis changes. An interpretive floor is added to
  H-W1, and a design gap is recorded.

#### What prompted it

Stage 1's progress lines show both the protocol arm and its round-trip control moving hard:
`dG_W1` ≈ **+24 nats** and `dG_C1` ≈ **+10 nats** over the first 11 items, against a frozen support
threshold of **+12.11**. Before reading that as a result, I re-expressed the **banked R2 arms**
exactly in the same `G_sum` units — same items, same clean traces, per-item differences of log-prob
sums, no approximation and no new compute:

| banked arm | G_sum (nats) |
|---|---|
| `l0prompt` — the whole prompt swap, i.e. the unit itself | **+121.13** |
| `P_prompt` — prompting | **+27.53** |
| **`R_random @ 0.149 id_spans` — a RANDOM vector at these very positions** | **+19.60** |
| `V1_gloss @ id_spans` | +5.00 |
| `V3_taskvec @ id_spans` | +2.66 |
| `V5_replace @ last_prompt` (all 33 layers, exact clean state) | +0.11 |

**A random vector at the identifier spans scores +19.60 — it clears the frozen +12.11 support
threshold on its own.** The threshold was set as "10 % of mean G_sum" by analogy with R2's
`SUPPORT_FRAC`, and R2 applied that fraction to a *per-token* quantity where the generic arms sat at
18–23 %. Carried over to `G_sum` it lost its relation to the generic benchmark.

#### The consequence, stated before the data

- **Clearing +12.11 is necessary but NOT sufficient** for any claim that the NLA's edit did
  belief-specific work. The rule stays exactly as frozen and the verdict it computes will be
  reported verbatim; but a `W-STEERS` verdict that does not also beat the generic benchmarks is to
  be written up as **generic perturbation**, not as steering.
- **The benchmarks to beat, fixed now:** `R_random @ id_spans` = **+19.60** (operator-generic) and
  `P_prompt` = **+27.53** (the mandated prompting baseline that no arm in this programme has ever
  beaten). Both are banked, both are on these 60 items.
- **In-design content control unchanged:** C2 (foreign edit) already blocks a steering claim when it
  matches W1. What C2 cannot do is rule out that the *operator* — norm-matched replacement of
  identifier activations with **any** plausible reconstruction — carries the effect. That is what C1
  measures, and C1 at ≈ +10 is already most of the way to the +12.11 bar.

#### The design gap, recorded rather than patched

Stage 1 has **no operator-level null**: an arm writing a *random* vector through
`PositionReplacer` at the same spans, norm-matched. The banked `R_random` is **additive** steering
at matched energy, not replacement, so it is an indicative benchmark and not a substitute.
**No arm is being added to a running experiment** — that is the forking path this project has
avoided six times. It is pre-registered here for the follow-up, and its absence bounds what stage 1
can conclude:

- **H-W6 (for the follow-up, frozen now):** a random unit vector written by `PositionReplacer` at
  the same identifier spans, norm-matched, scores **< W1 − 12.11 nats**. CONFIRM → W1's effect is
  not merely the operator. REFUTE → the effect is the *replacement of decoy identifier
  representations by anything at all*, which is a result about the trap rather than about the NLA,
  and a genuinely interesting one: it would say the decoy's damage lives in the identifier
  activations being *specifically wrong* rather than in what replaces them.

- **Setup:** unchanged. Banked figures from `data/nla/p0/trace_llr/gemma12b/llr_rows.jsonl`,
  clean-half rows only, paired against `noop#1` per item, n = 60.
- **Results / verdict:** *pending — job 377399 was at 11/60 when this was filed.*
- **Next Steps:** let 377399 finish untouched; report the frozen verdict **and** its position against
  +19.60 and +27.53.
