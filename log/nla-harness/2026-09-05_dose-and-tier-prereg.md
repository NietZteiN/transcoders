### Target Date: 2026-09-05 (H-W13 dose curve + H-W14 tier contrast + a harness identity check. Frozen before running.)

Raised by [`2026-09-05_null-battery-results.md`](2026-09-05_null-battery-results.md): the clean-state
effect is **item-specific but not span-specific** — a sibling span retains **87.4 %**. Two readings
survive that result and this family separates them. **Committed before the run.** One job, because
every arm writes at subsets of the same 375 locatable positions.

#### The two readings

- **Redundant.** Every identifier span carries the *same* item-level signal, so any one of them
  restores most of the effect.
- **Accumulative.** Each span carries a fraction and they sum, so a sibling looks nearly as good
  only because ~7.7 spans are patched either way and one substitution changes little.

These predict identical sibling-swap numbers and different **dose curves**, which is what H-W13
measures.

- **Hypotheses.**
  - **H-W13a (redundant):** `P_1 ≥ 0.50 × P_all`, where `P_k` patches **k** seeded-random locatable
    spans with their own clean states. CONFIRM → the item-level signal is present at *every* span.
  - **H-W13b (accumulative):** `P_1 < 0.50 × P_all` **and** the curve is monotone increasing over
    k ∈ {1, 2, 4, all}. **k = 2 and k = 4 are descriptive shape only and carry no verdict** — the
    gates are the endpoints.
  - **Dose confound, controlled by design:** k also sets **how many positions are written**, so
    delivered perturbation is confounded with dose. `F_1` and `F_all` (foreign clean states at the
    same k) are therefore run as matched controls, and the quantity that isolates content from
    operator at each dose is the **contrast** `P_k − F_k`, not `P_k` alone.
  - **H-W14 (tier):** is the item-level signal *"this is the clean program"* or *"this program"*?
    `F_all` (a foreign item's **L0** span) vs `F_L1b_all` (a foreign item's **L1b** span).
    CONFIRM a generic clean-code component if `F_all − F_L1b_all ≥ +12.11` with a CI excluding 0;
    if the gap is ≈ 0, a foreign vector supplies generic *program-ness* regardless of tier, and the
    measured +15.18 says nothing about cleanliness.
  - **SELF — a hard harness identity check, not a hypothesis.** `PositionReplacer` writes
    `‖h‖ · unit(v)`, so writing a position's **own** current activation is *exactly* the identity.
    `|ΔG_SELF| ≤ 1.0` nat is **asserted**; a violation means the write path, the position mapping or
    the scorer is faulty and **the run is refused rather than reported**. This costs one extra
    scoring pass and tests the entire pipeline end to end at every written position — something no
    W run has had.

- **Setup (frozen).** The 375 locatable spans, 49 items, `PositionReplacer` (direction-only, no α).
  Arms: `SELF`, `P_1`, `P_2`, `P_4`, `P_all`, `F_1`, `F_all`, `F_L1b_all`, plus the unsteered noop.
  Span subsets and foreign pairings seeded from `crc32(snippet_id#k)`. Readout `G_sum`; paired
  percentile bootstrap 10,000; seed 20260724; BH-FDR across the gated contrasts.
  **Generation (for the behavioural veto) is run only for noop, `P_1` and `F_L1b_all`** — `P_all` is
  already measured with a passing veto (+45.71, acc 0.714, job 378019) and the intermediate k are
  descriptive. Arms without a generation are reported with `veto: null` and **cannot carry a
  verdict**, which is the same clause that made stage 1 refuse a verdict without a baseline.
  `--deterministic` OFF. Host `gemma12b` L32/48.

- **What each outcome means, fixed in advance.**
  | H-W13 | H-W14 | reading |
  |---|---|---|
  | redundant | tier matters | every span carries "this specific clean program"; one span suffices |
  | redundant | tier does not | every span carries "a program-shaped state"; the +15.18 foreign floor is generic and **the item-specific gap is the whole result** |
  | accumulative | tier matters | the signal is distributed across spans and clean-specific |
  | accumulative | tier does not | distributed and generic — the weakest reading, and it would bound H-W7's fidelity claim to a distributed, tier-insensitive quantity |

- **Power / bounds.** n = 49, the same easier subset (baseline acc 0.714, swap unit +111.57).
  The battery's contrasts had CIs of ±5 nats, so a 12.11-nat gap is resolvable; **a null H-W14 is
  therefore informative rather than merely underpowered.** One host, one layer, one tier of
  obfuscation, 375 identifier-span positions. `P_1` writes ~3 positions against `P_all`'s ~25, and
  no claim is made that the two deliver equal perturbation — that is what `F_1`/`F_all` are for.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
- **Next Steps:** implement as `nla/src/nla_dose.py`; smoke, then 49 items.
