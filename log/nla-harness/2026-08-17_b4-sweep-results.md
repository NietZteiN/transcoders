# 2026-08-17 — B4 v2: the gate fails at every alpha; one exploratory cell (V2 at α=4) is the only lead

**Pre-registration:** `2026-08-16_b4-sweep-prereg.md` — primary gate **V1 > V3 at α = 1.0,
`last_prompt`**; secondary alphas BH-adjusted; the primary excluded from its own correction.

## Setup

```
env    nla-mi (run) / transcoders-mi (score) · seed 20260724 · layer 20
split  GPU 2 alphas {0.25, 0.5, 1.0} · GPU 3 alphas {2.0, 4.0} — disjoint keys, one JSONL
rows   2,160 total · 0 errors · 60 L0/L1b pairs (30 wrong at L1b = the recovery target)
score  nla/src/steer_stats.py --primary-alpha 1.0
```

Integrity checks during the run: 0 corrupt lines from concurrent append, 0 duplicate keys,
baseline reproduced the banked reference exactly (L0 0.600 / L1b 0.500).

## The pre-registered gate — FAILS, again

| | V1 | V3 | Δ | discordant | McNemar p |
|---|---|---|---|---|---|
| **α = 1.0 (primary)** | **0.550** | **0.550** | **0.000** | 7 v 7 | **1.00** |

Identical to the banked single-alpha run, as it must be — those 420 rows were resumed, not
recomputed.

## And it fails at every other alpha too

| α | V1 | V3 | Δ | p | q (BH) | verdict |
|---|---|---|---|---|---|---|
| 0.25 | 0.583 | 0.567 | +0.017 | 1.000 | 1.00 | V1 > V3 |
| 0.5 | 0.517 | 0.550 | −0.033 | 0.727 | 1.00 | V3 ≥ V1 |
| **1.0** | 0.550 | 0.550 | 0.000 | 1.000 | *(primary)* | tie |
| 2.0 | 0.633 | 0.617 | +0.017 | 1.000 | 1.00 | V1 > V3 |
| 4.0 | 0.550 | 0.550 | 0.000 | 1.000 | 1.00 | tie |

**"V1 does not beat V3" now holds across a 16× range of α**, with every |Δ| ≤ 0.033 and every
q = 1.00. This is what the sweep was for: it converts a single-point null into a claim we own.
As pre-registered, it strengthened the negative rather than overturning it.

## The confound the sweep exposed: everything drifts up with α

| α | mean Δacc across all 7 conditions | R_random | V3 |
|---|---|---|---|
| 0.25 | +0.045 | +0.017 | +0.067 |
| 0.5 | +0.031 | +0.017 | +0.050 |
| 1.0 | +0.050 | +0.033 | +0.050 |
| 2.0 | +0.076 | +0.083 | +0.117 |
| 4.0 | +0.083 | +0.083 | +0.050 |

A **random, norm-matched direction** gains as much as most real ones by α = 4 (+0.083). Any
sufficiently large perturbation nudges accuracy on this task. This is precisely why the
pre-registered bar is V3 and not zero, and it is the single most important line in the table for
anyone reading a raw Δaccuracy column.

## The one exploratory lead — labelled as such

**V2 (the minimal single-word edit) is the only condition with a monotone dose-response**, and
at the top of the range it separates from V3:

| α | V2 | V3 | discordant | p |
|---|---|---|---|---|
| 0.25 | 0.500 | 0.567 | 2 v 6 | 0.289 |
| 0.5 | 0.533 | 0.550 | 6 v 7 | 1.000 |
| 1.0 | 0.617 | 0.550 | 9 v 5 | 0.424 |
| 2.0 | 0.617 | 0.617 | 7 v 7 | 1.000 |
| **4.0** | **0.683** | 0.550 | **12 v 4** | **0.077** |

V2's Δacc runs 0.000 → 0.033 → 0.117 → 0.117 → **0.183**. V3 does *not* drift (0.050 at both
α = 1 and α = 4), so the V2–V3 gap at α = 4 is not the general perturbation drift above.

**Do not bank this.** It is exploratory, not the pre-registered comparison; p = 0.077 does not
clear 0.05; under the same BH correction applied to the alpha family it would be q ≈ 0.385; and
it is **one cell out of 25**. It is also exactly the shape of the two results this programme has
already lost — N11's coupling 3/3 → 3/10, and B4's own n = 34 interim → an exact tie at n = 60.

It is nonetheless the most interesting thing in the sweep, and the direction is theoretically
sensible: V2 is the *more* NLA-dependent intervention (it needs the reconstructor to render one
specific lexical substitution), while V1 — the whole-gloss swap that could differ in any of up
to eight terms — shows nothing.

**It cannot be confirmed on this corpus.** n is hard-capped at 70 pairs by `dataset_a`+`dataset_b`,
and 60 are already spent. A frozen confirmatory test needs the 862-snippet Java corpus, i.e. the
B5 composed runner. That is now the single best reason to build it.

## Verdict

- **B4 NOT SUPPORTED**, on a 16× α range, at the pre-registered α, against seven controls.
- Prompt-only remains +0.100 — competitive with every steering condition.
- The banked antipodal-equals-random result stands: the V1 direction has no consistent sign, and
  the sweep confirms scaling a sign-free vector does not help.
- **One lead survives**: V2 at high α, exploratory, unconfirmable here, carried to B5.

## Notes on the run itself

- **V2 was silently broken until this session**: its decoy→true pairing zipped the two tiers'
  identifier-term lists, which differ in length on 52 of 60 pairs because adversarial renaming
  creates more identifier sites than the original has terms. Only 8 directions were being built.
  Using `meta.rename_map` gives 60/60. Had this not been caught, the one interesting cell in the
  sweep would have been reported on 8 items.
- The AR was pinning ~11 GB on the generation GPU after its caches were built; freeing it
  returned ~12 GB of headroom (`del ar` after cache construction).
