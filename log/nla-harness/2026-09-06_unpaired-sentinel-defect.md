### Target Date: 2026-09-06 (H-W19 — the `?unpaired` sentinel is a **stimulus-pipeline pairing failure**, not a property of the obfuscation; it holes identifier correspondence tier- and language-dependently, and it reaches Papers 2–3)

**No GPU.** Corrects two claims I made earlier this week. **Append-only — neither prior entry is
edited.**

- **What I set out to test.** [`2026-09-06_hw17-language-confound.md`](2026-09-06_hw17-language-confound.md)
  asked whether the missing JavaScript L1 anchoring was a generation-pipeline asymmetry or a real
  language property. **It is a pipeline defect, and it is worse and wider than the question assumed.**

- **First, two hypotheses of mine that the data refutes.**
  1. *"JavaScript L1 items lack a `rename_map`."* **False.** **30/30** JS L1 items have one, exactly
     like **40/40** Python.
  2. *"L1 renaming is thorough on 20 items and absent on 29"* (my framing in the language addendum).
     **False.** Measured as the fraction of L0 identifiers absent from L1, renaming is **identical**
     across languages: **0.521 Python vs 0.522 JavaScript**. The obfuscation ran. Only the
     *bookkeeping* failed.

- **The actual defect.** `?unpairedN` appears as a **rename_map KEY**, meaning the generator renamed
  an identifier and **could not recover what the original was**. Share of keys that are sentinels:

  | tier | Python | JavaScript |
  |---|---|---|
  | **L1** | 48.7 % | **97.3 %** |
  | **L1b** | **41.7 %** | **0.0 %** |
  | **L3** | 45.4 % | **91.8 %** |

  Two things stand out. The failure is **large everywhere** — even the best cell loses 42 % of its
  pairings. And it **inverts between tiers**: L1/L3 fail almost totally on JavaScript, while L1b
  fails only on Python and is *perfect* on JavaScript. Whatever matcher recovers original↔renamed
  correspondence behaves differently per tier and per language.

- **Correction 1 — to [`2026-09-04_nla-fidelity-prereg.md`](2026-09-04_nla-fidelity-prereg.md).**
  That entry reported 88 of 476 spans mapping to `?unpairedN` and concluded:

  > adversarial renaming did not only *rename* identifiers, it **introduced new ones** … there is no
  > clean state to patch.

  **That inference is wrong.** The sentinel means the pipeline lost the original name, not that no
  original exists — the L0 code plainly contains one. Those 88 spans **do** have a clean counterpart;
  we simply do not know which identifier it is. The *operational* consequence for H-W7 is unchanged
  (they still cannot be anchored, so excluding them was right), but the **stated reason was wrong**
  and the sentence about renaming "introducing" identifiers must not be quoted.

- **Correction 2 — to the language addendum.** H-W17's 20-item subset is not "items where L1 renaming
  was thorough". It is **items whose L1 pairing metadata survived**, which is ~19/20 Python purely
  because L1's matcher fails on 97.3 % of JavaScript keys. The result is still **Python-only** and
  still cited that way; the reason is different, and the reason matters, because a metadata failure
  is repairable whereas a language property is not.

- **Why this leaves this thread.** Any analysis anywhere in the study that uses `rename_map` to pair
  an original identifier with its obfuscated form inherits a **tier- and language-dependent hole of
  42–97 %**. That includes **Instrument 1's identifier-level attention measures** and any
  semantic-displacement scoring keyed on original↔decoy pairs, if they are built on this field.
  **This is not a claim that Papers 2–3 are wrong** — I have not inspected their code — but it is a
  concrete, checkable risk, and the check is an hour of work: grep for `rename_map` in the analysis
  path and ask what it does with a `?unpaired` key.

- **The repair is probably cheap, which is the good news.** The originals exist in the L0 source and
  the renamed forms exist in the tier source; only the correspondence was lost. Position- or
  AST-based re-alignment could rebuild it offline, with no re-generation of stimuli and no GPU.
  **H-W22:** attempt that re-alignment on the 88 W-corpus spans and measure recovered coverage. If
  it works, H-W7's C3 ceiling and H-W17's contrast can both be re-run on ~100 % of spans instead of
  79 % and 28 %, and the JavaScript half of the corpus becomes usable for the L1 question for the
  first time.

- **Hypotheses / verdict:** H-W19 ✓ resolved — **pipeline defect, not language property.**
- **Next Steps:** (a) flag the `rename_map` risk to the Instrument-1 analysis path; (b) H-W22
  re-alignment, CPU-only, before any further tier-contrast GPU work.
