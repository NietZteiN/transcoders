# Costing a flippable corpus — can the NLA steering work ever be scored on accuracy?

*Last updated: 2026-09-12 (corrected — see Changelog)*

**Decision this supports.** Every steering result in the `nla-harness` thread is scored on `G_sum`
(teacher-forced log-prob of the banked L0 reply under the L1b prompt), never on task accuracy, and
each prereg has restated the same caveat: the corpus is too small for an accuracy claim. This doc
replaces that recurring caveat with arithmetic, an inventory of what is actually on this host, and a
costed go/no-go. It is a **scoping document — nothing here has been run.**

Written alongside the H-S5…H-S9 pre-registration
([`../log/nla-harness/2026-09-12_beta-layerset-prereg.md`](../log/nla-harness/2026-09-12_beta-layerset-prereg.md)),
whose H-S9 accuracy stage is gated and pre-declared unable to support a conclusion either way.

---

## 1. The accuracy question is not underpowered — on this corpus it is unanswerable

Gemma-3-4B-it on the 60-item bank ([`../log/nla-harness/2026-09-09_phase-b-go.md`](../log/nla-harness/2026-09-09_phase-b-go.md),
job 384619): `acc_l0` **0.567** → `acc_l1b` **0.533**, **7 flippable** (items the host gets right at
L0 and wrong at L1b). Net penalty is **2 of 60 items**.

The binding constraint is not the flippable count but that it sits **at the noise floor**. The greedy
reproducibility floor on this harness is 0.85–0.90 — **6 to 9 items of 60 disagree between two runs
of the same condition** ([`../log/nla-harness/2026-09-07_flippable-census-results.md`](../log/nla-harness/2026-09-07_flippable-census-results.md);
[`../log/nla-harness/2026-08-29_determinism-floor-structural.md`](../log/nla-harness/2026-08-29_determinism-floor-structural.md)
established the floor is structural and not fixable by deterministic kernels). So the *signal ceiling*
(7 items) and the *noise* (6–9 items) are the same size.

Exact McNemar (two-sided, the right test for paired flips) at n = 60:

| scenario | (rescued, counter-flipped) | p |
|---|---|---|
| perfect rescue of all 7, **zero** counter-churn | (7, 0) | **0.016** |
| perfect rescue, 2 counter-flips | (7, 2) | 0.180 |
| perfect rescue + the observed same-condition churn | (10, 3) | **0.092** |
| realistic partial rescue | (4, 1) | 0.375 |
| partial rescue inside churn | (4, 3) | 1.000 |

**Read the third row.** Even an intervention that rescues *every* flippable item cannot reach p < 0.05
once the measured same-condition churn is present. Only the zero-churn row clears, and zero churn is
not what this harness delivers on the 4B. Any accuracy number produced on this corpus is therefore
uninterpretable in both directions — which is exactly why every prereg has declared it in advance
rather than reporting it and arguing.

### How much corpus would be enough

Target: detect a rescue of **half** the flippable items against ~3 counter-flips, two-sided p < 0.05.

| flippable items | (b, c) | p | items to screen @ 11.7 % yield |
|---|---|---|---|
| 7 (today) | (6, 3) | 0.508 | 60 |
| 12 | (9, 3) | 0.146 | 103 |
| **16** | (11, 3) | **0.057** | **137** |
| **20** | (13, 3) | **0.021** | **171** |
| 25 | (15, 3) | 0.008 | 214 |
| 30 | (18, 3) | 0.002 | 257 |

**~20 flippable items ⇒ ~170 screened items** is the honest minimum; ~25–30 (215–260 screened) buys
margin for a yield below 11.7 % and for partial rather than half rescue.

---

## 2. Inventory — what is actually on this host

| | count |
|---|---|
| Dataset A (`data/stimuli/dataset_a/dataset_a.jsonl`) | 100 rows = **20 items** × 5 tiers (L0/L1/L1b/L2/L3) |
| Dataset B (`data/stimuli/dataset_b/dataset_b.jsonl`) | 250 rows = **50 items** × 5 tiers |
| **total distinct items** | **70** — all 70 have L0, L1b and `expected_output` |
| usable by `steer_run.load_pairs` | **60** |
| banked 4B traces (`data/nla/p0/trace_llr/gemma4b/traces.jsonl`) | 60 — the bank is already complete |
| usable items **not** in the trace bank | **0** |

Language split of the usable 60: 30 Python / 30 JavaScript. Sources: `humaneval-x-{python,javascript}`,
`cruxeval-x-{python,javascript}`, `leetcodedataset` (10 items each in Dataset B).

**The corpus on this host is exhausted.** There is no untapped margin to trace; 60 is all of it.

### The 10 lost items are all of Dataset A's Python half

`load_pairs` drops exactly 10 items because `build_call` returns nothing — and every one is a
**Dataset A Python** item (`Python/{31,36,40,43,74,101,104,121,125,154}`). Their stored `input` field
is `None`, so no executable call can be constructed, even though `expected_output` is present.

Two consequences worth separating:

1. **Dataset A contributes only its 10 JavaScript items** to every steering run in this thread
   (60 = 10 from A + 50 from B). This is a corpus fact that no entry in the thread has recorded.
2. Dataset A is the **human-labeled** half — `dataset_a_human_labels.csv`, 600 graded responses over
   98 snippet-tier cells. So the missing inputs are also what blocks human-difficulty triangulation
   (the charter's ρ = 0.30–0.47 alignment split) on the Python side.

Recovering them is **not free and not safe by guesswork**: HumanEval-X is public and its canonical
inputs are recoverable, but `expected_output` was stored for *some specific* input, and pairing a
guessed input with a stored output would silently mis-grade. It needs the provenance, not a
reconstruction. Value if recovered: +10 items ⇒ ~1 extra flippable item. **Not a route to power** —
worth doing for Dataset A completeness and the human-label join, not for accuracy.

### What is and is not reachable — **corrected 2026-09-12**

[`../data/DATA_SOURCES.md`](../data/DATA_SOURCES.md) (written 2026-08-04, describing the original
monorepo) names two artifacts that would make expansion cheap. An earlier version of this section
said both were absent. **One claim was wrong and is corrected here:**

- **`tasks_unified_500.json` = 2,500 rows = 500 base problems × 5 tiers** — described verbatim as
  *"Supersets exist … if more items are ever needed."* **Still not found**, after searching
  `/work/jvl210002`, `$HOME` and `/scratch/juno/jvl210002`. If retrieved, expansion costs one run of
  the existing converter `src/convert_stimuli.py` plus the screening GPU time below.
- **The obfuscation pipeline IS reachable — and it is the WRONG TOOL.** It is present in
  `~/allocation_replication_HEAD_2026-09-09.bundle` (a git bundle, HEAD `47b6cfe7`, 29 551 files) at
  `pipeline/obfuscation/{base,rename,flatten,deadcode,indirection,builder,cli,validate}.py`. Three
  facts kill it for this purpose:
  1. **It is Java-only.** `base.py:1` — *"Shared infrastructure for the Java source-to-source
     obfuscator"*; it parses with `tree_sitter_java` and requires the output to stay
     `javalang`-parseable. Dataset A/B are **python and javascript**.
  2. **Its renaming is NEUTRAL, not adversarial.** `rename.py` emits `_obf_t1_<stem>_<6digits>` /
     `m_O0lI1`-style fresh names — that is the **L1** tier. **L1b, the adversarial decoy rename that
     the whole trap depends on, is not implemented anywhere in that repo**: zero matches for
     `adversarial renam` / `decoy` / `misleading name` / `L1b` across its Python and Markdown, and
     **zero matches for `smoothArea`**, the study's flagship `fibfib → smoothArea` trap.
  3. `artifact/obfuscation/README.md` says outright that the artifact *"does not include:
     obfuscation generation tooling, semantic-equivalence checking, dataset-building or
     post-verification utilities."*

  So that repo is a **different project** (attention steering on obfuscated **Java**), not the
  Papers 2–3 python/javascript stimulus pipeline.

**Corrected bottom line:** the blocker is not merely retrieval of a data file — **no adversarial-rename
generator is reachable at all**, so new L1b tiers cannot be minted with anything on this machine. That
makes retrieving `tasks_unified_500.json` (whose L1b tiers already exist) the *only* cheap route, and
strengthens rather than weakens the recommendation in §5.

### What the bundle DOES give: the source corpus, and 10 recoverable items

The bundle carries a large **un-obfuscated** source corpus with per-task `.meta.json`:
**Cruxeval/python 799 · Cruxeval/javascript 743 · Humaneval/python 164** (~1 700 python/js tasks, plus
17 other languages). Two consequences:

1. **The 10 dropped Dataset A Python items are recoverable with real provenance.** Their sources are
   all present (`artifact/Source/Humaneval/python/Python_{031,036,040,043,074,101,104,121,125,154}.py`)
   and each carries the canonical HumanEval `check()` function with concrete assertions — e.g.
   Python/101: `words_string("Hi, my name is John") == ["Hi","my","name","is","John"]`. Dataset A
   **already has all five tiers** for these items (13 identifier spans on L0 and L1b for Python/101);
   only `input` is `None` and `expected_output` is the literal string `'None'`. So the obfuscation work
   is done — what is missing is one call per item, and it can be taken from the canonical test and
   **validated by execution** with the existing `nla/src/exec_oracle.py` rather than guessed.
   *Value:* +10 items ⇒ ~1 extra flippable. **Still not a route to accuracy power**, but it ends the
   situation where Dataset A contributes only JavaScript, and it unblocks the human-difficulty join
   (the charter's ρ = 0.30–0.47) on the Python side.
2. It is a source pool for a purpose-built corpus **if** an adversarial renamer were ever written —
   which §5 argues against, and the corrected finding above makes a larger job than it looked.

---

## 3. Three options, costed

### Option A — retrieve `tasks_unified_500.json`, convert, screen (**recommended**)
1. Retrieve the file from the canonical hub (`model_understanding/base/data/corpus/` per
   `DATA_SOURCES.md`) — **no GPU, no generation**; it is already tiered.
2. Run `src/convert_stimuli.py` over it. The converter's span→token resolution was validated at
   **1.0000** and it derives `rename_map` cross-tier from the stimuli themselves, so it does not
   depend on the mismatched alignment parquets. Expect the same `build_call` attrition (~14 %, and
   worth measuring: if the `input: None` pattern is Dataset-A-specific it may not recur).
3. Screen for flippability: 2 greedy generations per item (L0 and L1b) at `MAX_NEW_GEN = 1100`. The
   banked census cost **25 min for 60 items on one h200** (job 381294) ⇒ ~**3.5 GPU-h for 500**, or
   ~1.2 GPU-h if screening stops at the first ~170 usable items.
4. Re-bank traces for the retained items and re-run the vector stage at the layers of interest.
   **This is the real cost**: `nla_ml_gate.py`'s vector stage is ~0.8 min/item/layer (measured on the
   12B; the 4B is cheaper), so vectors at a handful of layers for ~200 items is **~10–20 GPU-h**, not
   the 44.6 GPU-h of retraining pairs.

**Total: ~15–25 GPU-h plus a file retrieval.** Yields ~20–30 flippable items and makes accuracy
answerable for the first time in this programme.

### Option B — screen a different host for a bigger denominator (**closed**)
Already tested and refuted. H-W30 screened Llama-3.1-8B-Instruct: 9 forward flips against a threshold
of exactly 9.0, so the frozen gate fired PASS — **and the pass was not real.** Llama also had **6
reverse flips** (obfuscation making items *correct*), a direct churn estimate, drawn from a pool of 15
items moving either way, net penalty 5 points. Gemma-12B had **6 forward and 0 reverse**, net 10
points. *A smaller denominator with no counter-flips beats a larger one that is half noise.* Gemma was
already the better trap host; there is no permitted candidate left to try.

### Option C — keep `G_sum` as the only readout (**status quo**)
Zero cost, and it is what every result in this thread already does. The cost is that the programme
cannot state the thing a reader most wants — whether steering *helps the model answer correctly* —
and the charter's E2 ("test accuracy recovery on L1b traps") stays unfulfilled.

---

## 4. Do these regardless of the option chosen

1. **Replace binary greedy accuracy with an n-sample pass-rate.** A per-item rate over n = 8 samples
   has ~**2.8×** smaller per-item SE than one greedy draw (0.177 vs 0.500 at p = 0.5) — the same
   precision as ~8× the items, for ~2 GPU-h. This is the single cheapest lever on the churn floor and
   it is already the design of H-S9. **Implementation trap:** `answer_entropy.py:186-190` batches
   multi-sample generation with **left padding**, which breaks `PositionReplacer`'s absolute-position
   targeting; use `num_return_sequences` at batch 1 with no padding.
2. **Measure the 4B's reverse flips.** Only forward flips (7) were recorded for this host. The 12B's
   0 and Llama's 6 are what made H-W30 decidable; the 4B's reverse count is unknown and is the number
   that says whether it is a clean trap host. Free — it falls out of any screening pass.
3. **Record the Dataset A Python gap** in the thread ledger, so no future entry assumes the 60 items
   are a balanced 30/30 draw from both datasets when Dataset A contributes only JavaScript.

## 5. Go / no-go

**Go on Option A if and only if `tasks_unified_500.json` can be retrieved.** It converts a
~15–25 GPU-h job into a worthwhile one and is the only path that makes the charter's E2 answerable.
If it cannot be retrieved, then — per the correction above — **no adversarial-rename generator is
reachable at all**, so minting new L1b tiers means writing one from scratch whose renaming character
must match the banked stimuli (`fibfib → smoothArea`, not the parquets' `fibfib → rns`). At that point
**Option C is the honest choice** and the accuracy claim should be dropped from the programme's stated
goals rather than pursued at that cost.

**Do regardless:** recover the 10 Dataset A Python items from the bundle's canonical tests (validated
by `exec_oracle.py`). It is ~1–2 h of work and near-zero compute, it does not help accuracy power, and
it fixes a real corpus defect plus the Python-side human-label join.

## Changelog
- **2026-09-12 (correction)** — §2 said the obfuscation pipeline was absent from this host. **Wrong.**
  It is present in `~/allocation_replication_HEAD_2026-09-09.bundle`, but it is the wrong tool: Java-only
  (`tree_sitter_java`), **neutral** renaming (`_obf_t1_…`, i.e. the L1 tier) with **no adversarial/L1b
  mode anywhere** (zero matches for `decoy`/`adversarial renam`/`smoothArea`), and its own README says it
  excludes generation tooling. The corrected conclusion is *stronger*: no adversarial-rename generator is
  reachable at all. Added the bundle's ~1 700-task python/js source corpus and the finding that the 10
  dropped Dataset A Python items are recoverable with provenance from the canonical HumanEval tests.
- **2026-09-12** — Created alongside the H-S5…H-S9 prereg. Establishes that a perfect rescue cannot
  reach p < 0.05 on the present corpus (McNemar (10,3) → p = 0.092); inventories 70 items / 60 usable
  and identifies the 10 dropped items as all of Dataset A's Python half (`input: None`); finds that
  the documented 500-item superset and the obfuscation pipeline are **absent from this host**, which
  makes retrieval rather than GPU the blocker.
