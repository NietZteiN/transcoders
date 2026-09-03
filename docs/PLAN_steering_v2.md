# Steering v2 — a plan

*Last updated: 2026-09-02. Design document, not a pre-registration. Each experiment below gets
its own frozen rule before it runs.*

## The problem in one paragraph

Every causal result in this programme comes from **one vector, at one token position, at one
layer**: `α·‖h‖·Δ̂` written to `layers[20]` at `last_prompt`, on `Qwen2.5-7B-Instruct`, tier L1b.
B4 refuted it, B5 found no composition, and Phase 0 established the failure is not the channel
(P0.2), not a dead site (P0.3) and not the wrong depth (P0.4). The natural reading is that there
is no item-level belief to edit. But that reading rests on an intervention so narrow that it is
worth asking what it could have moved even in principle.

## The reframing that makes this worth redoing

**Every steering experiment was run at a site now known to carry nothing.** The read-side work
(2026-08-30 → 09-02) established that at `last_prompt` a linear probe on the raw residual stream
**ties a token count** — +0.010 over reply length on Qwen, +0.04–0.06 per tier. We steered, for
weeks, at the one position where the probe says there is nothing item-specific to steer.

Meanwhile there is now exactly one place across three hosts and nine cells where the residual
stream demonstrably carries item-level information beyond every surface baseline:
**Llama-3.1-8B, tier L2, peaking at layers 5–9** (ρ = +0.42 at L8; +0.1062 over the strict
baseline, replicated, p = 0.0149).

> **The principle for v2: steer where the probe says something is represented.** Not at the site
> the source protocol happened to specify.

## Four axes, none of which the existing work varied

### A. Coverage, with energy held constant — *the confound in P0.2*

P0.2 widened from one position to `all_reply` **at the same α = 1.0**, so total injected energy
rose by roughly the reply length. It did not under-deliver; it **over**-delivered and destroyed
generation — V1's parse rate fell 0.917 → **0.100**. That tells us nothing about coverage,
because coverage and magnitude moved together.

The missing experiment is a **coverage sweep at matched total energy**: positions
`{1, 4, 16, 64, all_reply}` with `α` scaled as `α₀/√k`, so `Σα²` is constant. Any interior
maximum is a coverage effect; a monotone decline is a magnitude effect.

*Needs:* a `--positions first_k:<n>` spec and an `--alpha-schedule energy_matched` flag.

### B. Site selection by content — *never tried, and the annotations already exist*

Steering has only ever targeted `last_prompt`, a position chosen for protocol reasons. But the
stimuli carry **`identifier_spans`** (11 per L1b item) marking exactly where the decoy names sit,
and **`dispatcher_spans`** (11 per L2 item) marking the indirection sites. N11 established that
the decoy's influence is a **local echo** — 30.9% on the identifier, 3.0% off it, 0.0% past 400
characters. If the decoy is local, the correction should be local too.

*Experiment:* inject at the identifier spans (L1b) or dispatcher spans (L2) rather than at
`last_prompt`, energy-matched against a same-count set of **random** positions — the null that
decides whether it is the *sites* or merely *more sites*.

### C. The KV bypass — *the mechanism the report already names but never tested*

A write at `layers[20]`, position *p*, propagates only to layers 21–27 **for that position**.
The KV entries at layers 0–20 for *p* were computed before the hook fired and are left unedited,
so **every later token still attends to the un-edited decoy through the bottom 21 layers**. The
intervention does not change what the model subsequently reads; it changes only what that one
position contributes upward, in 7 of 28 layers.

This is a sufficient mechanical explanation for the null, and it is independent of whether a
belief exists. Two ways to test it:

1. **Edit the KV cache** at the target position for layers ≤ 20, so later tokens read the
   corrected state. Directly removes the bypass.
2. **Steer during prefill at every layer** with per-layer vectors — legitimate because **V3 and
   V4 are activation differences defined at every layer**, so no NLA is needed and P0.1's HARD
   verdict (which forbids *re-using the L20 vector* elsewhere) does not apply.

*This is the highest-value item on the list.* If steering works once the bypass is closed, every
prior null is a statement about the hook, not about beliefs.

### D. Read-time versus write-time

`last_prompt` is the boundary between reading and generating. The Llama L2 effect peaks at
**layers 5–9** — early, while the model is still reading the code, before any answer exists to
read out. Every prior intervention landed at layer 20 at the end of the prompt: late in depth,
late in position. Steering early-and-during-reading has never been attempted, and the only site
with a demonstrated signal is exactly there.

## Order of work

| # | experiment | cost | why first |
|---|---|---|---|
| 1 | **KV-bypass test** (C) on the banked Qwen L1b setup | ~6 GPU-h | Cheapest decisive test. Re-uses banked stimuli and the existing α=0 identity gate. If the bypass explains the null, everything downstream changes. |
| 2 | **Coverage at matched energy** (A) | ~10 GPU-h | Removes P0.2's confound. Also produces the dose–response curve B4 never had for position. |
| 3 | **Span-targeted injection** (B), L1b identifiers | ~8 GPU-h | Tests the locality N11 measured. Random-position null is mandatory. |
| 4 | **Llama L2, layers 5–9, spans** (B+D) | ~10 GPU-h | The only site with a replicated read-side signal. Highest prior of the four. |

## Non-negotiables, from what this programme has already learned

- **Parse rate is a primary criterion, not a diagnostic.** P0.2's negative was *destruction*, not
  absence. Any condition below 0.80 parse is reported as DESTRUCTIVE and never as a null.
- **Energy-matched controls throughout.** A random norm-matched direction gains +0.083 at high α;
  more positions is a bigger perturbation before it is anything else.
- **V3 remains the bar, not zero.** An NLA-derived direction that does not beat a contrastive
  difference vector has bought interpretability, not capability.
- **Graded k/N labels and the did-not-terminate category**, per 2026-08-29 and 09-02. Single
  greedy grades flipped a curve's argmax once already.
- **The α=0 identity gate before every run** — byte-identical to unsteered, edit at the target
  position, neighbours bit-unchanged.
- **One frozen rule per experiment, written before the run.** Six positives have been retracted
  here; the ones that survived were pre-registered.

## What would change the conclusion

If (1) or (2) recovers accuracy, the Phase-0 licensing table is reopened: the failure was the
*intervention*, not the representation. If neither does, the negative gets substantially stronger
— it would then hold across coverage, site selection, depth, and the KV bypass, which is a much
harder claim to dismiss than "one vector at one token did not work."

## Changelog
- **2026-09-02** — Written after the read-side programme showed the historic steering site carries
  nothing beyond reply length, and after Llama L2 gave the first site with a replicated
  item-level signal.
