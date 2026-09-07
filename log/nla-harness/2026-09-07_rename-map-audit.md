### Target Date: 2026-09-07 (Audit of the `rename_map` defect's blast radius — the generator's own algorithm, the consumers that guard, and the one banked arm that did not)

**No GPU.** Follows [`2026-09-06_unpaired-sentinel-defect.md`](2026-09-06_unpaired-sentinel-defect.md),
which established that `?unpairedN` marks a **lost** pairing rather than an absent one.

- **The generator's algorithm, now read rather than inferred.** `src/convert_stimuli.py`,
  `derive_rename_map()`:

  > *"Renaming preserves token structure, so equal-length identifier sequences pair 1:1."*
  > `if len(sf) == len(st):` … positional zip … else
  > `return {f"?unpaired{i}": n …}, False`

  **Equal-length positional matching, exactly as predicted** on 2026-09-06 from the coverage rates
  before the source was read. The stated assumption is false for JavaScript L1, which rewrites
  member access (`lst.length` → `c['length']`) and so changes the identifier-sequence length, and
  for L3, which stacks renaming on flattening. That is why the sentinel rate is 97.3 % for JS L1 and
  91.8 % for JS L3 while L1b — a pure rename — is 0.0 % on JavaScript.

- **The generator *records* the failure, and most consumers respect it.** `derive_rename_map` returns
  `(map, pairing_ok)` and the flag is written into every stimulus row as `meta.pairing_ok`
  (dataset_a: **39 True, 21 False, 40 None**). Three consumers check it:
  `task_bank.py:44`, `task_bank.py:58`, and `overnight_capture.py:351`; `task_bank.glosses()`
  additionally filters `?`-prefixed keys and documents the skip. **The information was never
  missing — it was recorded, flagged, and honoured by the older code.**

- **⚠️ One banked arm does not honour it, and it is measurable.** `trace_llr.py`'s **V2** arm builds
  its "minimal single-word edit toward the true name" by iterating `rename_map.items()` as
  `(true_name, decoy_name)` with no `?unpaired` guard. When the key is a sentinel it substitutes the
  **literal string `?unpaired0`** into the gloss handed to the reconstructor.

  Measured over the 60 banked items using the arm's own deterministic tie-break:

  | V2's chosen substitution | items |
  |---|---|
  | a real true name | **49** |
  | **the sentinel string** | **11 (18 %)** |

  Examples of the text V2 actually reconstructed: `_node_to_msg → ?unpaired0`,
  `logmsg → ?unpaired0`, `accepted_answer → ?unpaired0`.

  **Scope of the damage — bounded, and it does not overturn anything.** V2 is a *secondary* arm
  throughout: R2's belief-arm ladder (where it sits at −0.093 of the unit, negative like every other
  belief write) and the N/C battery (where swapping one word does what swapping the whole gloss
  does: nothing). On 18 % of items it was measuring "substitute a nonsense token", which is at least
  as harmful as the intended edit, so the negative verdict is unaffected in direction. **What is
  affected is the claim's cleanliness:** "swapping one word toward the truth does nothing" is
  cleanly supported on **49 of 60** items, not 60. Any future citation of V2 carries that.
  Guard added at the site with the measurement in the comment.

- **Observations.**
  - **This is the same defect I made, one level up.** The W-family scripts also never check
    `pairing_ok`; they survive because a sentinel key fails `term_spans(code_l0, "?unpaired0")` and
    the span is dropped — **correct behaviour by accident, not by design.** The older code
    (`task_bank`, `overnight_capture`) was written defensively and the newer code, mine included,
    was not.
  - **The right fix is upstream and now exists.** `repair_pairs.recover()` reconstructs the
    correspondence at precision 1.000 / recall 1.000 on L1 and L1b in both languages, so the
    fallback the converter reaches for is no longer the best available answer. **H-W28:** re-derive
    `rename_map` for the whole stimulus set offline and record it as a sibling field, leaving the
    original untouched — no regeneration, no GPU, and it retires the sentinel for every consumer at
    once.
  - **Blast radius outside this repo is unresolved.** Instrument 1's analysis path is not in this
    tree, so the audit could not reach it. What can be said precisely: **any consumer that reads
    `rename_map` without checking `meta.pairing_ok` inherits a 42–97 % hole**, and the flag needed to
    detect that has been sitting in every stimulus row the whole time.

- **Also corrected here:** the skip reason added to `test_steer.py` on 2026-09-06 said the fixture
  model "was never fetched into the HF cache". Wrong — it **is** cached, with config and tokenizer
  and **zero weight shards**, which is why transformers raises a missing-file error rather than a
  missing-repo one. Comment fixed; the skip behaviour is unchanged.

- **Next Steps:** H-W28 (offline re-derivation), and the flippable census on Llama-3.1-8B — the
  corpus question that bounds every accuracy claim.
