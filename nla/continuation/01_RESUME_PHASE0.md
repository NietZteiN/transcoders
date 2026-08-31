# 01 — Resuming Phase 0

Phase 0 was mid-flight when the move happened. **It was stopped deliberately, not by a crash**,
and everything is resumable. Read `00_STATE.md` §Phase 0 first for what the experiments *are*.

---

## Why Phase 0 matters, in one paragraph

B4 (belief steering) is refuted and B5 (composition) is null. Both are currently being read as
*"there is no item-level belief to edit"* — but a second explanation predicts identical tables:
**the injection channel cannot deliver anything, whatever it carries.** Phase 0 decides which,
and the decision changes **which paper gets written**. Do not spend GPU time on the
position-allocation work or B5's second seed until this resolves.

---

## The decision rules — frozen, and not to be renegotiated

Full text in `artifacts/2026-08-27_p0-triage-prereg.md`. Summary:

| P0.2 | P0.3 | conclusion | next |
|---|---|---|---|
| channel-limited | site live | the channel was the bottleneck | **Phase 1a** — position-allocation paper; re-run the B4 gate at the better allocation |
| not channel-limited | site dead | single-layer late intervention cannot move this task | **Phase 1b** — readout paper with a bounded, mechanistic causal negative |
| channel-limited | site dead | contradictory — diagnose, report neither | — |
| not channel-limited | site live | the site works but belief-shaped writes do not | strongest support for "no item-level belief" |

- **P0.2 CHANNEL-LIMITED** iff V4 improves by **≥ +0.10** with a CI excluding zero **and**
  `R_random` does not match the gain. A V4 gain that random matches is generic perturbation, not
  delivery.
- **P0.3 SITE LIVE** iff [20,20] differs from the unsteered baseline by more than the baseline's
  own seed-to-seed spread, in the same direction at both seeds. **SITE DEAD** iff [20,20] is within
  seed noise while [20,27] is not. If [20,27] is *also* within seed noise, the curve is flat and
  P0.3 is **UNINFORMATIVE** — explicitly *not* to be read as SITE DEAD.

---

## What is already done, and what is left

| | state | remaining cost |
|---|---|---|
| **P0.1** layer rotation | ✅ complete — verdict **HARD**, argmax coherence L13 | — |
| **P0.2** channel | baseline 60/60 done, **300 steering rows on disk, unscored** | resumes; ~30–60 min |
| **P0.3** site | **8/10 cells complete** at full scale | only the two `[20,27]` cells, ~1.5 h on 2 GPUs |

---

## How to resume

Everything is idempotent — completed P0.3 cells carry `.done` markers and are skipped; P0.2 reuses
its finished baseline and its existing rows.

```bash
cd <repo>/transcoders
setsid nohup bash nla/scripts/p0_autopilot.sh >/dev/null 2>&1 &
```

`setsid` matters: it detaches the process group from any controlling terminal, so the chain
survives logout, a dropped SSH connection, or a killed tmux server. Verify with
`ps -o pid,ppid,sid,tty,cmd -p <pid>` — you want **`TT=?`** and its **own SID**.

Watch it:

```bash
tail -f data/nla/p0/autopilot.log
cat  data/nla/p0/P0_SUMMARY.md      # written when the chain completes
```

`P0_SUMMARY.md` applies the frozen rules and prints the verdict it computes — including
`UNINFORMATIVE` when the rule says so — and reports missing stages as **MISSING** rather than
omitting them.

### Before you launch, on a new cluster

1. **GPU check is mandatory** — this is a shared box with no scheduler. `p03_site.sh` hard-codes
   **GPUs 0 and 3**; edit that if the new cluster's free cards differ. Never launch onto a card
   another job is using.
2. **Apply the cross-repo patch** if it did not travel:
   `cd allocation_replication && git apply <this folder>/artifacts/steer_layers_flag.patch`.
   Without `--steer-layers A:B`, P0.3 cannot target layer 20 (`--steer-last-n-layers 1` gives
   layer **27**, not 20).
3. **Smoke-test first**, per the project rule. `layer_rotation.py --limit 2 --gate-only` runs the
   layer-index gate in ~1 min and proves the extraction path is correctly indexed.

---

## Two bugs were found and fixed in this pipeline — do not reintroduce them

Both were **silent-failure class**: the chain would have reported success while producing nothing
or the wrong thing.

**1. `steer_stats.py` has no `--out-dir`, and all three of its path defaults point at the banked
B4 run** (`data/nla/n12/…`). Calling it without explicit paths would have scored the *old*
experiment and reported a plausible number for a run that never happened — and `--out` alone would
have **overwritten `n12/steer_stats.json`**, destroying the banked B4 result. Always pass
`--results`, `--baseline` and `--out` explicitly.

**2. Nothing wrote `cells_scored.json` for P0.3.** `p0_summary.py` reads it; `p03_site.sh` only
called the generic aggregate, which regenerates the RQ1 grid instead. P0.3 would have burned ~5
GPU-hours and produced **no verdict** while the chain reported success. Fixed by
`nla/src/p03_score.py` — walks the per-run `score.json` tree, groups by run tag, and counts
unscorable runs into `runs_unscorable` rather than dropping them (a cell whose denominator quietly
shrank looks identical to one that did well).

**A third hazard, still live:** if a P0.3 cell fails, the autopilot re-runs the driver, which
claims GPUs 0 and 3 unconditionally rather than waiting. It only triggers on a failed cell; it was
left alone rather than adding untested logic to a running pipeline.

**Do not edit a running bash script.** Bash reads scripts incrementally by byte offset, so an
in-place edit corrupts execution. Stop the chain, patch, relaunch — it re-adopts work in flight.

### Stopping it

Kill the **autopilot first**, or it will see the driver exit and immediately relaunch it:

```bash
pkill -f "bash nla/scripts/p0_autopilot.sh"   # 1. the chain
pkill -f "p03_site.sh"                        # 2. the driver AND its worker subshells
pkill -f "obfuscation/main.py"                # 3. the workers
pkill -f "steer_run.py"                       # 4. P0.2
```

The driver's worker subshells survive a single pass and will spawn the *next* cell — kill by
explicit PID (shells before children) and verify with `nvidia-smi` that the cards actually freed.

---

## P0.4 — proposed, NOT pre-registered, wired but disabled

P0.1 found cross-item coherence peaks at **layer 13**, not the instrument's layer 20, while
magnitude peaks at 20. Since **V3 and V4 are pure activation differences defined at every layer**,
running them at layer 13 vs layer 20 is testable with no autoencoder. If V3@13 beats V3@20, "the
instrument is at the wrong depth" stops being an inference and becomes a causal result.

It is deliberately **off by default** in `p0_autopilot.sh` because adding an unregistered
experiment to an automated chain is exactly the forking path this project has otherwise avoided.
**Write its decision rule first**, then:

```bash
P0_WITH_L13=1 setsid nohup bash nla/scripts/p0_autopilot.sh >/dev/null 2>&1 &
```

(The stage is a stub — it logs and returns. Implement it after the rule is frozen.)
