### Target Date: 2026-09-08 (H-W33 — reclassify L3 spans by what the tier DID, not by what the name looks like. Frozen before running.)

**Thread:** nla-harness · **Job:** none (CPU) · **Raised by:**
[`2026-09-07_l3-is-l1-not-l1b.md`](2026-09-07_l3-is-l1-not-l1b.md) ·
**Enabled by:** [`2026-09-07_pairing-recovery-results.md`](2026-09-07_pairing-recovery-results.md).

- **The defect.** `src/convert_stimuli.py:133` labels a renamed span `l1_neutral` if its new name
  matches `^(?:var|func|fn|v|f)_[0-9a-fA-F]{2,8}$` and **`adversarial` otherwise** — a decision made
  on the *shape of the string*. Python L3's `var_68f8` matches and is labelled correctly; JavaScript
  L3's mangled names (`a`, `b`, `uepoi`) do not match, so they fall through to `adversarial`.
  Measured now over `data/stimuli/dataset_{a,b}/*.jsonl`:

  | tier · language | `adversarial` | `l1_neutral` |
  |---|---|---|
  | L1 · javascript | **0** | 523 |
  | L1 · python | **0** | 675 |
  | L1b · javascript | 506 | 0 |
  | L1b · python | 709 | 0 |
  | **L3 · javascript** | **1,327** | **0** |
  | **L3 · python** | **376** | 675 |

  L1 — whose renames L3 inherits — has **zero** adversarial spans in either language, while L3 has
  1,703. On the 2026-09-07 finding that L3 = L1 ∘ L2, every one of those 1,703 is suspect.

- **The frozen rule.** A span is classified by **what the tier did to that identifier**, verified
  per span rather than guessed from its shape. Using the validated pairings from H-W28 (L1 anchored
  on L0; L3 anchored on L2, which carries L0's names):

  | new class | condition |
  |---|---|
  | `unchanged` | the L3 name equals the original |
  | **`nonsense`** | the L3 name equals **the L1 name for the same original** — the tier applied L1's meaningless rename |
  | `adversarial` | the L3 name equals **the L1b name for the same original** |
  | `unresolved` | no pairing for that original in the tier being compared against |

  Order matters and is fixed: `unchanged` → `adversarial` → `nonsense` → `unresolved`. Putting
  `adversarial` ahead of `nonsense` means that if L3 ever *did* carry a decoy, this rule finds it
  rather than absorbing it into the nonsense class — the rule is built so that its own premise can
  fail.

- **H-W33a — the prediction.** If L3 = L1 ∘ L2 holds at span level, **essentially all** currently-
  `adversarial` L3 spans reclassify to `nonsense` and **none** to `adversarial`.
  **`W33-CONFIRMED`** if ≥ 0.95 of resolvable L3 renames match the L1 name and **zero** match an L1b
  name. **`W33-MIXED`** if any resolvable L3 rename matches an L1b name — which would mean the
  2026-09-07 finding is too strong and some L3 items do carry decoys. **`W33-UNRESOLVED`** if fewer
  than half of L3 renames can be paired at all.

- **H-W33b — the control, on the tier where the current labels are believed correct.** The same rule
  applied to **L1b** must return `adversarial` for essentially all of its 1,215 spans. If it does
  not, the rule is broken rather than the labels — this is what stops H-W33a's result from being an
  artifact of the classifier. **Required: ≥ 0.95 of resolvable L1b renames classify `adversarial`.**

- **Output.** A sibling file `data/stimuli/span_classes_v2.jsonl` carrying the old class, the new
  class and the evidence for each span. **The stimulus files are not edited in place**, for the same
  reason as H-W28: they are the shared input to three instruments.

- **Bounds.** Only spans whose identifier can be paired in both tiers are classifiable; the rest are
  `unresolved` and are reported, not guessed. Pairing itself is 1.000-precision *where checkable*
  (H-W28), which is an upper bound.

- **Results / verdict:** not yet run.
