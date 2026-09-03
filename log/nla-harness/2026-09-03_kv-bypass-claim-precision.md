# 2026-09-03 — KV bypass: correcting an overstatement in today's pre-registration

**Thread:** nla-harness · **Corrects:** [`2026-09-03_kv-bypass-prereg.md`](2026-09-03_kv-bypass-prereg.md) §1 · **Status:** correction, filed before any result exists

Append-only, per §6. The pre-registration file is unchanged; this is the correction of record.

---

## The overstatement

The pre-registration says:

> "The intervention **never** changes what the model subsequently *reads*; it changes only what one
> position contributes upward, in 7 of 28 layers."

The first clause is **too strong, and contradicts the second**. The steered position's K/V entries
at layers 21–27 **are** edited — the hook fires at layer 20's output, so everything above it is
downstream of the write and gets cached in its corrected form. Later tokens attending at those
layers do read the correction.

## The accurate statement

> A write at layer 20 reaches later tokens through layers 21–27 only. At layers 0–20 — **21 of 28,
> or 75% of the depth** — every later token attends to the **un-edited decoy**, because those K/V
> entries were computed before the hook fired.

The gate's measurement is unaffected and was always the precise claim: max |Δ| at the edited
position across layers 0–20 is **exactly 0.000** (Qwen), and 33 of 48 layers likewise on Gemma.
What was wrong was the prose gloss, not the number.

## Why this matters enough to file separately

It changes the strength of the mechanism, not its direction. The bypass is **partial**, not total —
so H-C1's prediction is that closing the remaining 75% adds enough to clear +0.10, not that the
current instrument delivers nothing at all. Stated the original way, a reviewer who noticed the
top-7-layer path would be entitled to discount the whole argument, and would be right to.

**No decision rule changes.** H-C0/H-C1/H-C2, the energy-matching rule, the UNINFORMATIVE guard and
the V5 ceiling rule all stand exactly as filed. This is a correction to how the mechanism is
described, and it is being made **before any steering result exists**, so it cannot be an
after-the-fact reinterpretation of an outcome.

## Consequence for the write-up

Any text derived from the pre-registration must use the corrected form. The phrase "never changes
what the model reads" is not to appear in a report, an abstract, or a figure caption. The defensible
sentence is the one above: **75% of the depth, not all of it.**
