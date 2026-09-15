# 2026-09-09 · Phase B training halted at L18 by a filesystem quota, not by compute

**Thread:** nla-harness · **Experiment:** Phase B (all-layer NLA on Gemma-3-4B-it) · **Status:** recovered, partially
**Entries this supersedes:** none (append-only; corrects a wrong claim in my own run-line debug pass, see *Correction* below)

## What happened

Array `384655` (34 layers, 4-wide under QOS `juno` MaxJobsPU=4) trained **L0–L17 cleanly**, then:

| task | fate |
|---|---|
| 384655_18 | **FAILED exit 1** at 20:10Z — 68 min of AV training completed, died writing the shard |
| 384655_19 | FAILED exit 1, same signature |
| 384655_20 | FAILED exit 1, same signature (had already started before the hold landed) |

```
File "nla/src/nla_train.py", line 628, in stage_sft_av
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O error: Disk quota exceeded (os error 122)
```

`mfsgetquota /work/jvl210002` at the time of failure:

```
 size  | 1099675779072 | soft 1000000000000 (109.97 %) | hard 1100000000000 | 99.97 %
```

i.e. **~380 MB of headroom against a 1.1 TB hard quota**. Each remaining layer needs ~13–16 GB
(AV 7.6 GB constant + AR growing with the K+1 trunk), so all 16 remaining tasks were certain to fail
the same way. The AV holdout metrics were computed and logged *before* the save, so L18/L19/L20's
`holdout_gap_permuted_minus_real` (+0.164 / +0.158 / +0.153) are known but their pairs do not exist.

## Correction to the run-line debug pass earlier the same day

That pass (recorded in `CLAUDE_SCRATCHPAD.md`) asserted **"Disk fine: `/work` has 101 T free"**. That was
`df` output — free space on the MooseFS filesystem — and it is **not the binding constraint**. The binding
constraint is a per-directory quota on `/work/jvl210002` that `df` does not show. A pass that set out to
find exactly this class of problem missed it by measuring the wrong thing. **Rule going forward: check
`mfsgetquota <project root>`, never `df`, before any run that writes tens of GB.**

## Blast radius of the recovery (all human-approved, two rounds, via AskUserQuestion)

Approved and executed by me:
- `data/nla/ml/gemma4b/acts/L{0..17}.npy` — 18 files, **34.3 GB**. Safe: `nla_ml_gate.py` never reads
  `acts/` (the vectors stage reads AV/AR + `traces.jsonl`); only re-training one of L0–L17 would need them.
  Verified `acts/L{18..33}.npy` + `norms.json` + `norms.npy` all still present.

Approved but **NOT executed** — the permission classifier blocked recursive deletion, handed to the human:
- `migration/results/sweeps` (34 GB, last modified Dec 2025, contains a `prompt_sweep_v1 copy` duplicate)
- 6 Qwen-lineage HF caches (**47.4 GB**): `Qwen2.5-0.5B-Instruct`, `Qwen2.5-7B-Instruct`,
  `Qwen2.5-Coder-7B-Instruct`, `Qwen3-0.6B`, `DeepSeek-R1-Distill-Qwen-1.5B`, `DeepSeek-R1-Distill-Qwen-7B`.
  (The `kitft/nla-qwen2.5-7b-L20-{av,ar}` pair is **not** in the cache at all — nothing to reclaim there,
  contrary to a note in the repo README.)

**Observed, unexplained:** deleting 34.3 GB moved the quota counter from 1099.7 GB → **928.5 GB** (−171 GB).
The extra ~137 GB is most plausibly MooseFS reclaiming the three failed jobs' partial shards plus async
trash; recorded because it means the counter is not a reliable instantaneous measure of a delete's effect.

**Observed, needs a decision:** `models--Qwen--Qwen2.5-Coder-1.5B-Instruct` (3 GB) was present in a listing at
~20:25Z and gone at ~20:35Z, and the held obtune job `dl_sc2` was released by something other than me and is
now RUNNING and downloading into the same quota. `hf_home` is a **shared cache mutated by other sub-projects
concurrently** — treat its contents as not exclusively ours.

## Recovery state

- `scontrol hold 384655_[20-33]` (L20 had already started, hence its failure), then
  `scontrol release 384655_[21-26]` → **L21–L24 running, L25/L26 queued, L27–L33 still held**.
- `sbatch --array=18-20 nla/scripts/nla_ml_layer.sh` → **job 388490** for the three lost layers.
- Committed footprint ≈ 9 layers × 13.6 GB ≈ **122 GB** against 171 GB headroom — deliberately under-committed
  so a second quota hit cannot happen while L27–L33 stay held.
- **The `afterok` chain is dead**: `384656` (vectors) and `384657` (score) depend on `afterok:384655_*`, which
  can never be satisfied now that 18/19/20 are terminal FAILED. Both must be resubmitted by hand. No work is
  lost: `stage_score` builds `have_vec` from the npz files that exist (`nla_ml_gate.py:327`) and excludes the rest.

## Scientific consequence

If L27–L33 are never trained, **every pre-registered set still resolves**: primary `{8,22,24}` and depth
`{8,17,26}` are all inside L0–L26, and `all` becomes L1–L26. That must be reported as a **quota-forced scope
change**, with the frozen rules untouched — not as the pre-registered `all`.

## Liveness through L17 (rules frozen in `nla/configs/nla_ml_gate.yaml`)

15 of 16 live; **L0 PAIR-DEAD on rule (b) alone** (`holdout_fve` 0.035 < 0.20). AR fve: 0.035 · 0.231 · 0.358 ·
0.383 · 0.376 · 0.406 · **0.415 (peak, L6)** · 0.398 · 0.396 · 0.385 · 0.374 · 0.361 · 0.340 · 0.307 · 0.304 ·
0.304 · 0.322 · **0.336 (L17)**.

## Hypothesis verdict

None settled. **A forecast I raised today is already contradicted:** I fitted the L6→L13 fve decline
(−0.0138/layer) and predicted a crossing of the 0.20 liveness floor at L22, with a *steepening* past L17 on
spectral grounds. L16/L17 came in at 0.322/0.336 — **rising**, against predicted 0.277/0.263. The
extrapolation over-read an 8-point trend and is withdrawn; the spectral measurement it rested on
(participation ratio ~1.3 at L8–L17 rising to 3–9 by L22–L33, so FVE is not comparable across layers and the
single absolute 0.20 floor applies to targets of very different difficulty) stands, and is still untested
where it matters.

## Next steps

1. Human runs the two blocked `rm -rf` commands (or declines → stop at L26).
2. Decide whether `dl_sc2` should be held until the array finishes; it writes into the same quota.
3. On array completion: resubmit `nla_ml_vectors.sh` with no dependency, then `nla_ml_score.sh`
   `--dependency=afterok:<vectors>`.
4. Results entry: liveness table first, then gate verdicts, with the scope change stated up front.
