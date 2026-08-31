# START HERE — cluster handoff for the **NLA line** (Instrument 2)

**Written 2026-08-28 on `csr-94608.utdallas.edu`, for a fresh Claude Code session on a different
cluster.** Read this file in full before touching anything.

> **Scope.** This folder covers the **Natural Language Autoencoder** work in `transcoders/nla/` —
> the belief-readout and belief-steering programme (N-series, B-series, Phase-0 triage).
> It does **not** cover the `obtune/` project, which has its own separate handoff in
> `obtune/continuation/`. Two different research lines, two different folders. Don't merge them.

| file | what it is |
|---|---|
| `README.md` (this) | orientation + the transfer checklist that must happen first |
| `00_STATE.md` | where the research actually stands — every verdict, the numbers, what is closed |
| `01_RESUME_PHASE0.md` | the experiment that was mid-flight when we moved, and how to restart it |
| `02_ENVIRONMENT.md` | what will break on a new cluster and exactly how to fix it |
| `03_LEDGER_DEBTS.md` | owed fixes and unlogged results, already diagnosed, not yet applied |
| `artifacts/` | small files that must survive even if nothing else does (see below) |

---

## ⚠️ TRANSFER CHECKLIST — do this before anything else

Unlike `obtune`, **both repos here have GitHub remotes**, so code can travel by push/pull.
Verified 2026-08-28:

| repo | remote | HEAD | uncommitted |
|---|---|---|---|
| `transcoders` | `github.com/NietZteiN/transcoders.git` | `b54b7ee` | **69** |
| `allocation_replication` | `github.com/NietZteiN/allocation_replication.git` | `bbd0ef3d` | **36** |

**Code only travels if it is committed and pushed first.** These files are *untracked* and would
be silently lost — they are the entire Phase-0 apparatus:

```
transcoders/nla/src/layer_rotation.py     # P0.1, the experiment that already produced a result
transcoders/nla/src/p03_score.py          # the scorer whose absence was a real bug (see 03)
transcoders/nla/src/p0_summary.py         # applies the frozen decision rules
transcoders/nla/scripts/p03_site.sh       # P0.3 driver
transcoders/nla/scripts/p0_autopilot.sh   # the unattended chain
transcoders/nla/continuation/             # this folder
```

**The cross-repo trap.** P0.3 cannot run without a patch that lives in the **other** repo:
`allocation_replication/artifact/obfuscation/main.py` gained a `--steer-layers A:B` flag.
It is easy to forget because it is not in `transcoders/` at all. A copy is saved here as
`artifacts/steer_layers_flag.patch` — apply it with
`git apply` from the `allocation_replication` root if the commit did not travel.

### Data that is NOT in git and must be copied separately

| path | size | why it matters |
|---|---|---|
| `transcoders/data/nla/p0/` | 1.3 MB | **the Phase-0 results so far** — P0.1 complete, 8 of 10 P0.3 cells |
| `transcoders/data/stimuli/` | 884 KB | the 60 L0/L1b pairs every steering experiment uses |
| `transcoders/data/nla/n12/` | 976 KB | the banked B4 run — the comparison baseline |
| `transcoders/data/nla/unified/` | 15 MB | the 14,619-reading unified table |
| `allocation_replication/artifact/artifacts/obfuscation/result/` | ~400 MB | per-case rows for the finished P0.3 cells |

### Data that does NOT need copying

- **`transcoders/nla/data/checkpoints/` (25 GB)** — re-download from Hugging Face:
  `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar`. Frozen released
  checkpoints, nothing was fine-tuned here.

---

## The 60-second version of the science

A **Natural Language Autoencoder** is a pair of frozen models that translate one layer of a
language model's internal state into English and back: **AV** (vector → English) and **AR**
(English → vector). Round-tripping gives `rt_cos`, a self-reported faithfulness score.

We pointed it at `Qwen2.5-7B-Instruct` at **layer 20 of 28** while it read deliberately obfuscated
and malicious code, across ~20,250 reads.

**The finding, in one line: the reads carry theme-level content reliably and item-level content
not at all.** They separate populations ("this is hostile code") but never identify *which*
capability, *which* algorithm, or *whether this run was misled*. Every negative in the programme
is an item-level claim; the one positive is population-level.

**The causal arm failed**, and the current open question is *why* — which is what Phase 0 exists
to decide. See `00_STATE.md` §Phase 0.

---

## Read next

1. `00_STATE.md` — do not re-derive any of this; it is all sourced from dated ledger entries.
2. `01_RESUME_PHASE0.md` — there is a half-finished experiment with frozen decision rules.
3. `02_ENVIRONMENT.md` — before running anything.

**The single most important discipline in this project:** decision rules are written down and
frozen *before* a run, and they are not renegotiated once the numbers are visible. Six results
here have reversed between a partial run and full sample size. Do not report a partial table.
