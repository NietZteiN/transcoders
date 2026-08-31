# 2026-08-13 — Unattended one-GPU pipeline, and a bug check that found a silent data-corruption class

**Goal.** Make the remaining B3 work run without supervision on a single GPU, and audit the code
written this session before it executes unattended. Nothing that runs overnight should depend on
a human noticing something.

**Setup.** Three detached tmux sessions; `loginctl show-user` reports **`Linger=yes`** and
`KillUserProcesses` is unset (default `no`), so user processes survive logout — that setting, not
tmux itself, is what determines whether this outlives an SSH session.

| session | stage | GPU |
|---|---|---|
| `nla-screen` | behavioural screen, 1,724 batched generations | 0 |
| `rq4-corpus` | four RQ4 obfuscation corpora | CPU |
| `nla-autopilot` | wait → select → capture → score | 0 |

New: `nla/scripts/deception_1gpu.sh` (AV server + runner co-located; AV at
`--mem-fraction-static 0.30` ≈ 14 GB, runner ≈ 27 GB, ~42 GB of 48), `nla/scripts/autopilot.sh`,
`nla/src/select_coupling_set.py`.

---

## The bug check

### 1. Snippet names collide across datasets — three sites, silent corruption

`comm` on the two source trees: **142 snippet names are shared** between humaneval and cruxeval
(`Java_000` exists in both). Three places keyed on the bare name:

- `deception_capture.py` — `task_key = f"{snippet}|{cond}"`. On resume, a completed humaneval
  `Java_000` would mark the cruxeval `Java_000` as already done, so it would never run and the
  file would look complete.
- `deception_capture.py --snippets` — matched the bare name and loaded only `--dataset`, so a
  selection file spanning both datasets would have **silently dropped half of it** and mislabelled
  what remained.
- `deception_stats.py` — `{(snippet, condition): row}` overwrites one dataset's capture with the
  other's, with no error.

All three now key on `(dataset, snippet)`, and `--snippets` loads every dataset its file mentions
and warns on any requested snippet it cannot find. **N11 escaped this only because it was
humaneval-only**; the coupling set spans both datasets, so it would have fired on the very next
run. Re-scoring N11 after the fix reproduces the published numbers exactly (C0 3.0% / C1 2.0% /
C2 9.0%, p = 0.039), confirming the fix is inert on single-dataset data.

### 2. A completed exit did not mean a completed run

`deception_capture` returns 0 when it stops on its wall-clock guard, so `autopilot` stage 3 would
have succeeded on a truncated capture and stage 4 would have scored it as final. The capture now
records `run_complete` in its manifest, and the autopilot compares actual rows against
`n_selected × 3` and refuses to score a short run, pointing at the idempotent resume path
instead. This is the same silent-failure shape the ledger keeps recording — a green signal that
does not mean what it appears to.

### 3. A GPU race between the screen and the capture

Stage 3 triggered on the screen's row count, but the count reaches 1,724 while the screen process
is still exiting and still holding ~25 GB. Adding ~42 GB on top would OOM, and the symptom would
have been a mysterious sglang crash rather than a recognisable race. Added stage 1b: wait until
the card reports < 2 GB before claiming it.

### 4. Direction normalized at activation precision

`steer.py` cast Δ to the activation dtype *before* normalizing, computing the norm of a 3584-dim
vector in bf16 (8 mantissa bits) and so applying a coefficient that differs from the requested
alpha. Now normalized in fp32 and cast afterwards; the AR returns fp32 anyway, so it costs
nothing. Regression test asserts the realized edit magnitude matches `alpha·‖h‖` within 2%.

### 5. `pkill -f` matched its own shell

Recorded for the record because it cost a launch: `pkill -9 -f "sglang.launch_server.*--port
30002"` matches any process whose command line contains that string — **including the shell
running the command** — so the `&&` chain died mid-way with no output. Both drivers now use a
bracketed pattern (`[s]glang...`) that cannot match itself.

**Test suite: 64 passing**, including the new fp32-normalization regression.

---

## Design choices in the autopilot worth recording

- **It stops itself when the science does not justify the spend.** If the screen yields fewer
  than 6 deceived items, stage 2 exits cleanly saying the corpus has no behavioural variance,
  rather than spending ~5 GPU-hours measuring coupling on four items.
- **Controls are matched inside the selection**, not after: each `stated_hit` item is paired with
  a non-hit sharing the **injected algorithm** and nearest in **code length**. Deceived items are
  not a random sample, and without this "deceived items have more capability reads" could just
  mean "deceived items are longer".
- **Stratified selection is declared as such.** `coupling_selection.json` carries the screen's
  own marginal rate and a note that read rate must be reported *within* stratum — the selection
  supports a coupling claim, never an unconditional deception rate.

**Next steps.** Screen → selection → capture → score runs unattended. If the coupling cell
survives its stratified test, the next planned item is the steering arm (B4), whose gate is
whether V1 (NLA explanation-edit Δ) beats V3 (contrastive task vector, no NLA) — if it does not,
the steering claim collapses to "a difference vector works" and should be reported that way.
