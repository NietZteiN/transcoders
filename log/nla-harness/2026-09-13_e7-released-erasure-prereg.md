# 2026-09-13 · PRE-REGISTRATION — the released 12B pair vs the training-free erasure vector (H-E7 / H-C11)

**Thread:** nla-harness · **Experiment:** `E_erasure_vector` at Gemma-3-12B-it **L32** ·
**Status:** pre-registered, nothing run. Rules frozen here before any code that could show new data.

## The question

Today's headline — a leave-one-item-out mean of `h0 − h1b` reproduces the trained NLA edit
([`2026-09-13_erasure-vector-results.md`](2026-09-13_erasure-vector-results.md): `erase_1.0` +35.75 vs `edit`
+32.06 at 4B L7) — carries one qualification: **our** pairs are data-limited (`DATA-LIMITED`, `DATA-NOT-STEPS`;
[`2026-09-12_dose-response-results.md`](2026-09-12_dose-response-results.md),
[`2026-09-13_same-steps-results.md`](2026-09-13_same-steps-results.md)). A pair trained at ~10× our volume with RL
might carry content the erasure vector cannot. The one such pair we can run is the released
`kitft/nla-gemma3-12b-L32-{av,ar}`, whose transport fidelity is 0.983 [0.967, 0.999]
([`2026-09-05_nla-fidelity-results.md`](2026-09-05_nla-fidelity-results.md)) against our 12B pair's 0.739
([`2026-09-12_12b-budget-results.md`](2026-09-12_12b-budget-results.md)). Its edit has only ever been measured on the
**old** anchoring (375 spans / 49 items: `W1_edit` +19.47, `C2_foreign` +10.68, `W1 − C2` +8.79 [+6.35, +11.41],
[`2026-09-04_writeback-results.md`](2026-09-04_writeback-results.md)) — the open H-C11.

So: run the released pair through the **same** gate stages as our 12B pair (repaired anchoring, 471 spans / 60
items, same host checkpoint, same traces), then write the erasure vector at L32 next to **both** trained edits in
one job. Three things fall out: H-C11's missing number, the like-for-like erasure-vs-best-available-edit test, and a
same-host-same-layer replication of the removal share (H-W23's 0.493 was measured **at 12B L32**, so H-E2's
cross-host agreement at 4B becomes a within-host test here).

## Arms (per span `s` of item `i`, L32, β = 1, full replacement — the operator of every banked run)

`E^{(−i)}` and `erase_α` exactly as in [`2026-09-12_erasure-vector-prereg.md`](2026-09-12_erasure-vector-prereg.md),
α ∈ {0.5, **1.0**, 2.0}; `erase_own` (identity, vector-gated); `swap`; `self`.

- **`edit` / `foreign`** — the **released** pair's AR(edited read) / AR(foreign-substituted read), produced by
  `nla_ml_gate.py --stage vectors` on a root whose `L32/{av,ar}` are the released snapshots. Same `substitute_terms`,
  same foreign-term seeding (crc32 of `sid#span`), same 180-token greedy reads as our pair.
- **`edit_ours` / `foreign_ours`** — **our** 12B pair's banked bytes
  (`/scratch/juno/jvl210002/nla_ml_gemma12b/gate/vectors/L32.npz`, job 391152), written in this job at the same
  positions. Identity-gated against our banked `dG_S_edit_L32` = +16.17 / `dG_S_foreign_L32` = +11.56.
- **`erase_flip` (NEW directional control)** — the LOO mean of `σ_s · Δ_s` with a seeded random sign σ_s ∈ {−1, +1}
  per span (seed 20260724 + crc32(`sid#L32#flip`)), unit-normalised, written at α = 1.0. **Why:** at 12B L32 a
  norm-matched Gaussian direction scores **−359** (job 391152), so `erase_rand` cannot serve as the directional null
  there — anything beats a vector that breaks the model. `erase_flip` stays inside span(Δ) with the same per-span
  scale statistics and destroys only the shared direction, which is the null the directional claim needs.
  `erase_rand` is still run and reported descriptively.

`edit`'s **editable** spans (read contains the decoy) are pair-specific: an unreadable span gets `v_rt`. The released
pair's editable count is reported next to ours (139/471) — a different denominator is part of the answer, not a
nuisance.

## Readout, pairing, bootstrap

`G_sum` per item (teacher-forced logp of the banked L0 reply under the L1b prompt), paired per item over the 60
items, cluster bootstrap **N_BOOT 10 000, seed 20260724** — unchanged. Traces:
`data/nla/p0/trace_llr/gemma12b/traces.jsonl`.

## Frozen rules

| id | contrast | verdict |
|---|---|---|
| **H-E7a** (primary) | `erase_1.0 − edit` (released), paired | CI lower > 0 → `ERASURE-BEATS-EDIT` · upper < 0 → `EDIT-CARRIES-CONTENT` · else `ERASURE-MATCHES-EDIT`; CI width reported |
| **H-E7b** | `erase_1.0 − edit_ours`, paired | same three verdicts, suffixed `-OURS` |
| **H-E7c** (gate on a/b) | `erase_1.0 − erase_flip` ≥ **+3.0** with CI lower > 0 | `ERASURE-DIRECTIONAL` else `ERASURE-NOISE` — voids a/b/d as in H-E3 |
| **H-E7d** | `erase_1.0 − 0.55·swap` upper < 0 | `ERASURE-WITHIN-SHARE`; share = mean/mean reported against **0.493** (H-W23, same host & layer) |
| **H-C11** | released `edit − foreign` ≥ **+3.0** with CI lower > 0 | `SPEC-LIVE` else `SPEC-FLAT`; reported next to ours (+4.61 [+3.13, +6.19]) and the old-anchoring +8.79 |

Descriptive, no verdict: released `c3/swap` (expect ≈ 0.98), `edit − edit_ours` paired, `erase_1.0 − foreign`,
Spearman(gain, cos(Δ_i, Ê)), the α ladder, `erase_rand`.

**Predictions, stated now.** H-E7c `ERASURE-DIRECTIONAL`. H-E7d `ERASURE-WITHIN-SHARE` with share in [0.40, 0.58]
(erase_1.0 ≈ +20 against swap +41.5). H-C11 `SPEC-LIVE` (the old-anchoring +8.79 should survive the repair as
+21.82 → +21.05 did). **H-E7a `ERASURE-MATCHES-EDIT`** — the released edit ≈ +20 and the erasure vector ≈ +20; a
CI of ±5 will not separate them. **H-E7b `ERASURE-BEATS-EDIT-OURS`** (our +16.17 sits under the predicted +20), held
weakly. **What would qualify the headline:** `EDIT-CARRIES-CONTENT` on H-E7a — the released pair beating the
vector with a CI excluding 0 — is the outcome under which "training buys nothing beyond erasure" is **false at
scale** and must be written that way.

## Gates — any failure ⇒ exit 3, `E-HARNESS-FAULT`, nothing reportable

- **G1 anchoring identity:** the released root's `L32_spans.jsonl` and ours resolve the **same 471 spans with the
  same positions, span for span**; `h0`/`h1b` agree (same host, same capture) — cos ≥ 0.999 per span, max |Δ|
  reported. This is what licenses `edit_ours` to be written from another root's bytes.
- **G2 score identity:** released `edit`/`foreign`/`swap` reproduce the released score rows to ≤ 0.05 nats; `edit_ours`
  and `foreign_ours` reproduce our banked rows to ≤ 0.05; `self` ≤ 1.0; vector gate (`VEC_TOL` 1e-3, bf16 fraction
  5e-4) on `erase_own` as fixed on 2026-09-13. `erase_own`'s score-level drift is the bf16 diagnostic, not a gate.
- **Liveness:** the released pair has no trainer `eval.json`/`check.json`, so the gate's score stage runs with
  `--ignore-liveness`; its `reportable` is therefore False **by construction** and is not read. The released pair's
  liveness is the 2026-09-04 stage-0 `NLA-LIVE`. Verdicts come from the erasure stats only.

## Setup

- Host `google/gemma-3-12b-it`, text-only ckpt `/scratch/juno/jvl210002/nla_ml_gemma12b/host_text` (job 391152's).
- Released root `/scratch/juno/jvl210002/nla_ml_gemma12b_released/L32/{av,ar}` → symlinks to the HF snapshots
  (`$HF_HOME/hub/models--kitft--nla-gemma3-12b-L32-{av,ar}/snapshots/*`, both with `nla_meta.yaml`).
- Stages: `vectors --layer 32` (~50 min) → `score --ignore-liveness` (~3 min) → `nla_erasure.py --layer 32
  --compare-root /scratch/juno/jvl210002/nla_ml_gemma12b` (~10 min). Smoke (3 items, all three stages, separate
  root) inside the sbatch before the full run. One H200 (`gpu:nvidia_h200_nvl`, matching 391152's kernels), ≈ 1.2 GPU-h.
- Script shas, job id, node and timestamps go in the results entry.

## Cost

≈ 1.2 GPU-h. I called this "nearly free" in the memo; the released pair's vectors stage is the same 50-minute stage
our pair needed, so the honest figure is above.
