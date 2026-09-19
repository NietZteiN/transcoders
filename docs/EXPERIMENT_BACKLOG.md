# Experiment backlog — Instrument 3

*Last updated: 2026-09-19 · supersedes nothing; `CHECKLIST.md` remains the hypothesis ledger*

Ordered by what unblocks the most. Costs are GPU-hours unless marked CPU. Status is what a run would
actually hit today, verified 2026-09-19, not what the charter assumes.

---

## Tier 0 — unblock the charter (nothing below Tier 0 can run first)

The `nla/` sub-thread (ASE/CodeSteer) consumed this week and is **finished**. The experiments this
sub-project actually exists for — **E1/E2/E3/E7** in `CLAUDE.md` §3 — have **never run**, and the reason
is mechanical, not scientific.

| # | task | cost | status | what it unblocks |
|---|---|---|---|---|
| **T0.1** | **Rebuild the `transcoders-mi` env** from `environment.lock.txt` (162 pinned lines incl. `torch==2.13.0`, `sae-lens==6.47.0`, `transformer-lens==3.2.1`, `circuit-tracer` @ pinned sha) | ~1 h **CPU** | **BLOCKER** | everything |
| **T0.2** | Fix the stale path in `scripts/env.sh` — it points at `/data/jvl210002/conda_envs/transcoders-mi`, the env is at `/work/jvl210002/conda_envs/transcoders-mi` | minutes **CPU** | **BLOCKER** | everything |
| **T0.3** | Dictionary smoke: load one Llama Scope SAE + `Llama-3.1-8B-Instruct`, reconstruct one prompt, report **FVE / L0 / dead-feature share** | ~0.3 h | ready after T0.1 | E1, E2, E7 |

**Verified state.** The env directory exists but is **pip-empty** — 696 MB containing only python + pip +
packaging, 25 site-packages entries, no torch. The conda create succeeded on 2026-08-03 (`data/env_create.log`
exits 0) on the **old A6000 host**; the pip layer did not survive the move to juno. Everything else E1
needs is present: `configs/experiments/e1_semantic_capture.yaml`, `configs/dictionaries.yaml` (pinned
revisions, verified via `HfApi.model_info`, with the exact SAELens loading recipe), stimuli
(`dataset_a/b/c`, alignment parquets), and **both dictionaries already cached** —
`OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` (residual SAEs) and `LXTC-8x` (transcoders), 518 MB each.

**T0.3 is a real gate, not a formality.** The dictionaries are trained on `Llama-3.1-8B` **base**, and
every E1/E2 claim would be read on **Instruct**. CLAUDE.md §4 names base→instruct transfer as a
silent-failure mode; if reconstruction is degenerate on Instruct, E1's design changes before it runs.

---

## Tier 1 — the committed experiments

| # | task | cost | status | decides |
|---|---|---|---|---|
| **T1.1** | **E4 — dispatcher state-binding probe** across hops | ~2 h | **ready now, needs no dictionary** | whether state-tracking is linearly decodable; the one committed experiment not blocked by T0 |
| **T1.2** | **E1 — semantic-capture feature diff** (L0 vs L1b, `dataset_a`, layers 12/16/20) | ~4 h | blocked on T0 | HT1: do decoy-semantics features fire while true-semantics features are suppressed? |
| T1.3 | E1 prerequisite: **auto-interp labelling** to populate `decoy_true_feature_sets` (currently `null` in the config) | ~2 h | blocked on T0 | makes E1's Semantic-Capture Score computable at all |
| **T1.4** | **E2 — causal feature steering** + the **mandatory** dense-steering and prompting baselines | ~6 h | blocked on T1.2 | whether an SAE feature edit recovers L1b accuracy — and whether it beats a prompt |
| T1.5 | **E3 PoC — attribution graphs** on a circuit-tracer-supported small model (Gemma-2-2B / Llama-3.2-1B), matched L0/L2/L3 | ~6 h | blocked on T0 | feasibility + **error-node mass**, which decides whether E3 is readable at all on OOD code |
| T1.6 | **E7 — cross-instrument triangulation** (SAE vs attention A_id/A_struct vs NLA/ISF) | ~2 h | blocked on T1.2 | the thing that makes this a third instrument rather than a standalone |

**Sequencing note.** T1.1 (E4) is the only committed experiment that needs no dictionary and no env
rebuild. If T0.1 stalls, **run E4 first** — it is panel-agnostic and tests the relational route on the
model side, which is exactly the route the ASE work could not reach from the stimulus side.

---

## Tier 2 — ASE follow-ups (the sub-thread is closed; these are the parts still worth money)

| # | task | cost | status | why |
|---|---|---|---|---|
| **T2.1** | **H-R20 — the decoder effect.** Greedy vs sampled on the panel, same corpus | ~2 h | **ready** | **the largest effect measured in the whole ASE block: ~16 points**, dwarfing every steering contrast. If their tables were produced at T 0.7, decoder choice dominates the intervention they measure |
| **T2.2** | **H-R19 — per-item stability.** Re-read Papers 2–3 per-item claims against the measured **38 % label-flip rate** under resampling | ~0 (analysis) | **ready** | per-item HCI correlations and flippable-item subsets from single sampled runs are partly measuring the draw; this is a validity check on already-published work |
| **T2.3** | **H-R24 — scope decision** (human): write the ASE line up as a bounded negative replication, or fund H-R29 | — | **needs a human** | five models, two stimuli, oracle ceiling +0.0052, no deficit anywhere |
| T2.4 | H-R29 — deeper flattening (nested CFG, ~20–40 states) **+ a differential-testing harness first** | ~2 d build + 1 h | gated on T2.3 | the only untested version of the flattening claim; failure mode is a **silent semantics change**, so the harness is not optional |
| T2.5 | H-R5 — head comparison: do CodeSteer's calibrated heads carry the NLA write? | ~3 h | ready | the one ASE question that is mechanistic rather than behavioural |

---

## Tier 3 — NLA core line (separate thread, paused)

The thread README lists **73 open hypotheses**; most are stale and should be archived rather than run.
The ones that still gate something:

| # | task | cost | why it still matters |
|---|---|---|---|
| T3.1 | **H-A9** — re-run all accuracy arms at 2 600 tokens | ~4 h | **until it exists, no accuracy figure in that thread may be quoted**, including `edit_single`'s −0.121 harm |
| T3.2 | **H-S16** — does the adversarial-rename effect exist on a **reasoning-tuned** host? | ~3 h | the accuracy arm of the NLA line lives or dies here; also the natural sequel to `NO-PERMITTED-HOST` |
| T3.3 | H-E11 — is the trained edit's advantage over a mean-difference vector non-linearity, or a bad linear estimator? | ~0.3 h | decides what NLA training actually buys |
| T3.4 | H-E16 — GPU-architecture assertion in every 12B runner | ~0 (CPU) | without it a legitimate re-run refuses and looks like a science failure |

---

## Housekeeping (cheap, pays for itself)

- **Archive the stale two-thirds of the 73 open hypotheses** in `log/nla-harness/README.md`. A ledger
  nobody can read is not a ledger.
- **`--mem` / `--time` audit is done** (2026-09-19): measured peak RSS is 15.4 GB (7B) / 26.7 GB (13B);
  all 45 sbatch scripts moved 200 G → 48 G, and walls now match measured runtimes. This was costing
  ~9 h of queue delay per submission round on the busy h200 nodes.
- **Budget context.** This repo is **23.4 %** of the account's GPU-hours (187.6 of 802.7 since 09-11);
  the whole ASE block is **5.7 %**. A single 12B layer run (`nla12b_layer`, 56.5 h) remains the largest
  single consumer — 30 % of this repo's usage, more than the entire ASE block.

## Changelog
- **2026-09-19** — created after the ASE block closed on both obfuscation routes; Tier 0 written from a
  live check of the env, dictionaries, configs and stimuli rather than from the charter's assumptions.
