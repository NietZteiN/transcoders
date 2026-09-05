### Target Date: 2026-09-05 (H-W4 is not a cheap re-analysis: the banked read corpora kept the scalar and discarded the vectors)

Corrects a claim in [`2026-09-04_cycle-gate-results.md`](2026-09-04_cycle-gate-results.md).
**Append-only — that entry keeps its text.** No GPU was used to establish this.

#### The claim being corrected

Stage 0 raised **H-W4** — re-express the banked `rt_cos` corpus as **centred** cosine, since
`NLACritic.score` L2-normalises but never centres and the raw metric was shown to compress a 0.68
effect into 0.02 of range. That entry filed it as:

> **H-W4:** re-express the banked `rt_cos` corpus as **centred** cosine. **No GPU where the vectors
> were persisted.** Could change effect sizes across several finished analyses.

The qualifier was carrying the whole claim, and it is now checked. Inspecting the banked read
corpora — `n13/read_instability.jsonl`, `n13/act_norm.jsonl`, `n10b/benign_reads_{neutral,security}.jsonl`,
`n11/{coupling,deception}_reads.jsonl` — **not one stores an activation vector.** They carry the
read text and derived scalars; no field in any of them holds more than a handful of numbers.

- **Consequence.** A centred cosine needs both `h` (the activation) and `rec` (the AR's
  reconstruction). `rec` can be recomputed cheaply from the banked read text with the AR alone, but
  **`h` requires re-extracting activations from the subject model for every row.** H-W4 is therefore
  a **GPU re-extraction job over ~14,600 rows**, not a CPU re-analysis, and it should be scheduled
  and justified as such rather than treated as free.

- **The general problem, which is the part worth keeping.** This project **systematically persists
  derived scalars and discards the vectors they were derived from.** The same defect has now
  appeared three times in different clothes:
  - **B5** ran with no per-case rows at all, so its six cells "cannot be re-analysed from disk —
    not re-scored, not clustered differently, not given a second seed without re-running from
    scratch" (`nla/continuation/03_LEDGER_DEBTS.md` §1).
  - **W stage 0** shipped dropping `h`/`rec` on the explicit claim that "every downstream question is
    answered by the cosines" — false within the same day, and fixed only because the anisotropy
    question arrived before the vectors were gone
    ([`2026-09-04_cycle-gate-amendment.md`](2026-09-04_cycle-gate-amendment.md)).
  - **The read corpora**, here.

  The cost is always the same and always invisible at the time: **a metric-definition question that
  should cost minutes costs GPU-hours**, and the sharpest lesson of 2026-09-03 — that four wrong
  headlines in one day came from measurement definitions rather than from the science — is exactly
  the kind of question that arrives *after* the run.

- **Recommendation, for anything new in this thread:** persist the vectors whenever a stored scalar
  is a *function of vectors*, as `cycle_vectors.npz` now does (~12 MB for 476 spans). `/work` has
  105 TB free; the storage argument for dropping them does not survive contact with the arithmetic.

- **Hypotheses:** none new. **H-W4 stays open**, re-classified from *cheap* to *GPU re-extraction*.
- **Results:** n/a — an audit of what is on disk.
- **Next Steps:** leave H-W4 queued behind the null battery; it is worth doing but not before the
  claims it would revise are settled.
