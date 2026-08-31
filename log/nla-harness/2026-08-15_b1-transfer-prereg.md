# 2026-08-15 — B1 pre-registration: does the released NLA read a different base model?

**Status: PRE-REGISTRATION. Written and committed before the full run.** The only B1 numbers
seen at the time of writing are from a 3-snippet / 6-read plumbing smoke, disclosed in full
below. No full-corpus number exists yet.

## Goal / hypothesis

`kitft/nla-qwen2.5-7b-L20-{av,ar}` was trained on `Qwen/Qwen2.5-7B-Instruct` layer-20
activations. `Qwen2.5-Coder-7B-Instruct` shares its architecture. **HB1: the released NLA
produces reads of the Coder's residual stream that are as faithful as its reads of its own
training host, on identical inputs.**

This is the paper's branch point. Pass → the subject model for the whole paper is
Qwen2.5-Coder-7B-Instruct, matching CodeSteer's primary model and the 244 banked replication
runs. Fail → the subject falls back to Qwen2.5-7B-Instruct and the CodeSteer conditions are
regenerated on it (~25–30 GPU-h; affordable precisely because the replication established that
the source paper's stored numbers were never reusable as anchors).

**Both outcomes are reportable and neither wastes the run.** The NLA release is silent on
cross-model transfer, so a clean negative — "the released NLA does not transfer even to a model
sharing its tokenizer, chat template, and every architectural dimension" — is a contribution in
its own right. Recorded here so there is no incentive to squint at the number afterwards.

## Why the pairing is exact

The two models ship the **same `tokenizer.json` blob** and a byte-identical chat template, and
agree on `hidden_size=3584`, `num_hidden_layers=28`, `num_attention_heads=28`,
`num_key_value_heads=4`, `vocab_size=152064`. For any input string they therefore produce the
identical token sequence and identical position indices. **The only variable in this comparison
is the weights.** That is what licenses the deterministic even-spacing position rule: the same
rule selects the same constructs in both arms.

## Design

- **Corpus.** 200 original (C0) Java sources from `allocation_replication/data/obf/humaneval`,
  sampled with seed 20260724 — the same corpus B3 read and B4/B5 will steer, so a transfer
  number is about the *model* and not about a different pile of code.
- **Positions.** 5 per snippet, evenly spaced over the code region, excluding positions below
  `extract.MIN_POSITION = 50` (the reference's own datagen floor — earlier positions decode to
  noise) and single-character / pure-digit pieces (which verbalize as numerology; established
  noise source since 2026-08-04).
- **Arms.** Control (`Qwen2.5-7B-Instruct`, the NLA's own host) runs **first**: if the
  instrument cannot reproduce its own host's numbers on this corpus, the subject arm is
  uninterpretable and the run stops. The control condition is the tripwire — the rule that
  caught four bugs in B4.
- **One AV server for both arms**, so the instrument is held fixed across the comparison.
- ~2,000 reads total at a measured 3.83 s/read sequential ≈ 2.1 h + ~0.5 h extraction.
  The read path stays sequential: batching was measured 6.3× faster but only 15%
  byte-identical (`data/nla/n5/async_gate.json`).

## Frozen decision rule

Primary: **median round-trip cosine on the subject, against the control on identical inputs.**

| # | criterion | threshold |
|---|---|---|
| 1 | median `rt_cos` on the subject | ≥ 0.70 (the established code-text band is 0.70–0.96) |
| 2 | median-cosine drop below the control | ≤ **0.05** |
| 3 | own vs foreign activation, 95% CI | lower bound > 0 |
| 4 | subject median `act_norm` | within 2× of the ~100–170 Qwen L20 band |
| 5 | mostly-CJK reads | < 1% |

All five must pass. Thresholds live in `nla/configs/b1_transfer_gate.yaml`; the gate is computed
by `src.transfer_gate compare` and cannot be recomputed with different numbers without a diff.

**Criterion 3 is the null attached to the deciding statistic**, not to a descriptive companion —
the HT14 error. Each explanation is scored against its own activation *and* against an
activation from a **different snippet** (foreign at the snippet level: two reads of the same
program share its subject matter, so a within-snippet "foreign" vector would smuggle in the
signal the null exists to strip). The CI is a cluster bootstrap resampling **snippets, not
reads** — reads within a snippet are not independent, and read-level bootstrapping is exactly
what made the 2026-08-06 faithfulness result a length artifact.

Why a null at all when the metric is a cosine rather than a rate: reads of Java activations
share structure, so a respectable-looking own-cosine is not evidence of item-specific content.
The smoke already shows this — foreign cosine sat at 0.443, far from 0.

## Declared deviations from plan v0.3

1. **`fve_nrm` is demoted from gate criterion to reported diagnostic.** v0.3's gate led with
   `fve_nrm ≥ 0.55` and `≥ 0.75× control`. Verified against the reference example
   (`vendor/nla-repo/examples/qwen7b_layer20_step4200.txt`, 101 tokens): `mse_nrm = 2(1 − cos)`
   holds to within the file's own 3-decimal rounding, and its `fve_nrm` is `1 − mse/denom` with
   `denom` constant at ≈0.734. So FVE against a shared denominator is a monotone restatement of
   mean cosine and carries no independent information; against a per-split denominator the extra
   signal is how dispersed each model's activation cloud is — a property of the model, not of
   how well the NLA reads it. It is also unstable at low n. Reported for comparability with the
   reference's published band, and guarded to n ≥ 100 reads.
2. **The layer sweep stays struck.** The AV/AR are trained for layer 20
   (`nla_meta.yaml: extraction_layer_index: 20`) and cannot be swept. A stated limitation, not a
   discovery to be made later.

## What was seen before freezing, disclosed

A 3-snippet / 6-read plumbing smoke ran before these thresholds were committed. It reported
subject median cosine 0.703 vs control 0.852 — **a 0.149 gap that would fail criterion 2.** The
δ = 0.05 threshold was chosen on general grounds before that smoke and was **not** adjusted
after seeing it. It is disclosed here rather than quietly re-tuned, because a threshold set
after seeing data is not a threshold. n = 6 reads over 3 snippets is a plumbing check and is not
evidence about HB1 in either direction.

The smoke also found and fixed two real defects: a `KeyError` on the sidecar's `d_model` (it is
top-level in schema_version 2, not nested), and the FVE instability above.

## What would make this null

The Coder's activation norms sit outside the band `injection_scale = 150` was tuned for, so
injection is out of distribution and reads degrade for a boring reason (criterion 4 catches
this; the decision rule is then to re-derive `injection_scale` from the subject's own norm
distribution, re-run, and report transfer quality as a first-class limitation). Or the reads
stay fluent and on-topic but stop being item-specific, which criterion 3 catches and criteria
1–2 would not.

## Setup

```
env         nla-mi (transformers 5.12.1, torch 2.11.0)
config      nla/configs/b1_transfer_gate.yaml
runner      nla/src/transfer_gate.py
driver      NLA_GPU=0 bash nla/scripts/b1_transfer.sh
seed        20260724   (foreign-null draw seed+1, cluster bootstrap seed+2)
layer       20
GPU         0   (GPU 1 is running N13 stages 2-3)
```

## Next

Run both arms, then `src.transfer_gate compare --a subject --b control`. The verdict is a
**human-gated checkpoint**: it selects the subject model for B0, B4, and B5, and no downstream
run should be sized until it lands.

---

## Addendum, 2026-08-16 — a position-selection defect found mid-run, and the remedy fixed in advance

**Found during a debug pass while the control arm was running, before any subject-arm read
existed and before any gate number was computed.**

`run()` calls `pick_positions(align, code_start, len(align.full), k)`. The third argument is
meant to be the end of the code region; `len(align.full)` is the length of the whole templated
string, which extends past the code into the chat-template tail
(`<|im_end|>\n<|im_start|>assistant\n`). The evenly-spaced grid's final point therefore always
lands on the last prompt token.

Measured on the control arm: **100 of 500 reads (20.0%) sit on the token `assistant`**, and they
are **slot 4 of 5 in 100 of 100 snippets** — systematic, not sampling noise. Their round-trip
quality differs from the code reads (mean `rt_cos` 0.787 vs 0.859), which is expected: they are
reads of a different token class.

**What this does and does not damage.** Position selection is deterministic and the two models
share a tokenizer, so *both arms receive the identical contamination at the identical indices*.
The paired comparison — which is what the gate rests on — remains apples-to-apples. What it
damages is the interpretation: a median described as "round-trip cosine on code reads" would be
diluted by one-fifth non-code reads, and the foreign-activation null would be computed over a
mixture.

**Why it is NOT being fixed in the running code.** Correcting `code_end` now would give the
subject arm different positions from the control arm and destroy the exact pairing, which is the
single property that makes this design a paired test rather than two unrelated samples. The
defect is therefore left in place for the duration of this run.

**Remedy, specified now, before the subject arm's data exists:**

1. Both arms run to completion with the current positions.
2. Reads whose token is a chat-template artifact (`assistant`, `user`, `system`,
   `<|im_start|>`, `<|im_end|>`) are **excluded** before computing every summary statistic and
   the gate. Each arm keeps 4 code reads per snippet (~656 per arm), which is ample.
3. The exclusion is by token identity, is applied identically to both arms, and is declared here
   *before* the subject arm has produced a single read — so it cannot be a filter chosen to move
   a number.
4. `pick_positions`' caller is corrected for future runs after this one completes.

The gate thresholds are unchanged. No threshold is being renegotiated in response to this.
