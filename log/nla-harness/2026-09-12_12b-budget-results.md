# 2026-09-12 · Phase C RESULTS — H-C1 → **BUDGET-EXPLAINS** (our pipeline at 12B L32 reaches C3 fidelity 0.739, not 0.98)

**Thread:** nla-harness · **Experiment:** Phase C, `ML_multilayer_gate` on Gemma-3-12B-it · **Status:** H-C1 resolved;
H-C2 **not adjudicated** (scope cut by the user mid-run: *"ok let's stop training once we get one layer trained"*);
H-C3 reported for the one trained layer.
Pre-registration: [`2026-09-10_12b-budget-prereg.md`](2026-09-10_12b-budget-prereg.md) — rules applied unchanged.
*(Naming note: the Phase-C ids H-C1…H-C3 collide with the 2026-09-03 KV-bypass ids H-C0…H-C2 in this ledger. Everywhere
below "H-C1" means the Phase-C budget hypothesis. Future entries should use a fresh prefix.)*

### Target Date: 2026-09-12 (12B L32 pair — liveness, gate, H-C1 verdict)

- **Hypotheses / what we're testing:** frozen in the prereg.
  **H-C1** — compare our 12B L32 pair's `S_c3/S_swap` with the released `kitft/nla-gemma3-12b-L32` pair's **0.98**
  on the same items: **BUDGET-EXPLAINS** if ≤ 0.80 · **HOST-EXPLAINS** if ≥ 0.90 · **INDETERMINATE** in (0.80, 0.90).
  **H-C2** — needs L2 and L7 at 12B; **not run** (see scope note). **H-C3** — liveness rules (a)–(d), descriptive.

- **Setup:**
  - Host `google/gemma-3-12b-it`, text-only ckpt `/scratch/juno/jvl210002/nla_ml_gemma12b/host_text` (48 layers,
    d 3840). Acts for all 48 layers extracted to scratch (138 GB). Corpus + 174 423 explanations **reused unchanged**
    from the 4B run (shared tokenizer, `㈜` = 246566).
  - **Training — job 389994** (`nla12b_layer`, h200, node `g-07-08`, GPUs 0,1 `NVIDIA H200 NVL`, 2026-09-10T23:32 →
    2026-09-12T03:48 local, elapsed **1-04:16:08 = 56.5 GPU-h**). `nla/scripts/nla_ml12b_layer.sh` sha `16992897…`
    → `torchrun --nproc_per_node 2 nla/src/nla_train.py --fsdp` with `NLA_FSDP_CPU_OFFLOAD=1 NLA_FSDP_NOSYNC=0
    NLA_NCCL_TIMEOUT_SEC=3600 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--mem 350G`.
    Config `nla/configs/nla_ml_12b.yaml` sha `c44d7ca2e60bf2be…`; trainer sha (from `check.json`) `64c8faedc8b430b3…`;
    git HEAD `3c52f26e`; seed **20260724**; every hyperparameter identical to the 4B run (lr 1.41e-5, global batch 128
    = micro 8 × accum 16, 1 epoch, no RL). Stage timings from the log: **AV** 86 822 rows / 858 holdout,
    injection_scale 74 000, **678 steps, 04:35Z → 22:31Z ≈ 18.0 h** (95–96 s/step, 73→81 GB/GPU with CPU offload);
    **AR** 85 895 rows / 848 holdout, 33-layer trunk, **671 steps ≈ 10.1 h** (58 GB/GPU); **check** 353 s.
  - **Gate — job 391152** (`nla12b_gate`, h200, `g-07-08`, 1 GPU, 2026-09-12T08:48:20Z → 09:42:16Z, **0.90 GPU-h**, rc=0).
    `nla/scripts/nla_ml12b_gate.sh` sha `ec3aa153…` → `nla/src/nla_ml_gate.py` sha **`1f7a9d3191a9ce32…`** (the same
    scorer that produced the 4B results), `--config nla/configs/nla_ml_gate_12b.yaml` sha **`076e3724816ecabf…`**
    (`train_config: nla_ml_12b.yaml`, `sets: primary: [32]`, thresholds inherited), **`--traces
    data/nla/p0/trace_llr/gemma12b/traces.jsonl` (60 items)** passed explicitly — the gate's default points at the 4B
    traces, which would have scored silently against the wrong host's banked replies. `n_boot` 10 000, seed 20260724.
  - **Dictionary identity:** AV `/scratch/juno/jvl210002/nla_ml_gemma12b/L32/av`, AR `…/L32/ar`, vectors
    `…/gate/vectors/L32.npz`, stats `…/gate/gate_stats.json` (large artifacts on scratch, not committed).
    Slurm logs `log/slurm/389994_nla12b_L4294967294.out`, `log/slurm/391152_nla12b_gate.out`.

- **Results:**

  **H-C3 liveness — L32 is LIVE on all four rules** (4B L32 alongside, same corpus, same rules):

  | rule | 12B L32 | 4B L32 | threshold |
  |---|---|---|---|
  | (a) AV `holdout_gap_permuted_minus_real` | **+0.1556** (1.2411 real vs 1.3967 permuted) | +0.1093 | > 0 |
  | (b) AR `holdout_fve` | **0.3438** (shuffled −0.374, train-mean −0.011) | 0.3381 | ≥ 0.20, > both baselines |
  | (c) no-tag / CJK reads | **0/96, 0.000** | 0/96, 0.000 | ≤ 10 % |
  | (d) `cos_cycle − cos_other` | **0.9899 − 0.9812 = +0.0086** | 0.9778 − 0.9661 = +0.0118 | > 0 |

  `mean_words` 90.0; `cos_gold_expl` 0.991; 0 duplicate reads. Vectors: **471 spans / 60 items, editable 139**
  (4B L32: 30; 4B median 97), `cos(h0,c3)` **0.982**, `cos(h1b,rt)` 0.976, `cos(h0,h1b)` 0.960.

  **Gate arms at L32** (summed `G_sum` nats, cluster bootstrap over 60 items, 95 % CI):

  | arm | 12B L32 (ours) | 4B L32 (ours) | 4B L22 (depth-matched, 0.65) |
  |---|---|---|---|
  | `G_prompt_swap` | +121.13 [110.94, 132.14] | +163.67 | (same host as 4B L32) |
  | `S_swap` (raw clean state; pair-independent) | **+41.47** [36.82, 46.33] | +0.11 | +59.87 |
  | `S_c3` = AR(AV(h_L0)) | **+30.65** [26.98, 34.51] | −0.24 | +40.77 |
  | `S_edit` | +16.17 [14.07, 18.36] | −0.56 | +21.24 |
  | `S_foreign` | +11.56 [9.91, 13.28] | −0.57 | +18.68 |
  | `S_rt` (unedited round trip) | +10.12 [8.52, 11.82] | −0.57 | +17.80 |
  | `S_random` | −359.44 [−507.38, −233.96] | +8.22 | −16.60 |
  | `SELF` | **0.000** [0, 0] | 0.000 | 0.000 |
  | **`S_c3/S_swap`** | **0.739 [0.695, 0.781]** | −2.15 (undefined; swap ≈ 0) | **0.68** |

  Gate block (set `primary` = [32]): gate0 `S_c3/G_prompt_swap` **0.253 [0.215, 0.294]** ≥ 0.25 → **ok**;
  H-M1 `M_edit − best single edit` = +0.00 (one layer — vacuous); H-M2 `M_edit − M_random` +375.61 vs threshold
  +9.19 → ok (inflated comparator, as flagged on 2026-09-10); specificity `M_edit − M_foreign` **+4.61 [3.13, 6.19]**
  → ok; SELF 0.000 (tol 1.0) → ok; label **`M-NO-GAIN`**. H-M3 block: 0 of 1 live layers clear CI-lower ≥ 0.80,
  **median ratio 0.739 → REFUTE**; `swap_peak_layer` 32; `best_single_edit_layer` 32 (both trivially, one layer).
  `identity_passes: true`, `reportable: true`.

- **What worked / hypothesis verdict:**
  - **H-C1 → BUDGET-EXPLAINS.** Measured ratio **0.739, CI [0.695, 0.781]**; the whole interval sits under the
    frozen 0.80 bar. Our pipeline, on the *same host and the same layer* as the released pair, transports **74 %** of
    the clean state's causal effect where the released pair transports **98 %**. Host size therefore does not account
    for the 4B's 0.68: a 12B host trained with our budget lands at 0.74, a CI-overlapping distance from the 4B's
    depth-matched 0.68, and **0.24 below the released artefact**. The gap follows the training recipe (86 k SFT rows,
    1 epoch, local explanations, no RL vs 100 k docs × 10 positions with RL).
  - **H-C2 → not adjudicated.** L2 and L7 were never trained; the prereg declared this outcome in advance ("If only
    L32 completes … H-C2 is simply not adjudicated"). EARLY-INVARIANT vs EARLY-IS-4B-ONLY stays open.
  - **H-C3 → prediction partly unverifiable.** The stated prediction concerned L2 ("L2 will be live"); L2 was not
    trained. The layer that *was* trained is live on all four rules, with rule-(b) fve 0.3438 comfortably above 0.20.

- **Observations:**
  1. **Fidelity did not improve with host size; editability did.** 12B L32 fve 0.3438 vs 4B L32 0.3381 (+0.006, noise);
     AV gap +0.156 vs +0.109. Yet `editable` is **139 vs 30** — 4.6× more spans where the English read actually
     mentions the identifier — and `S_swap` at 12B L32 is a full-sized **+41.47** where 4B L32's was **+0.11**. So the
     4B's top-of-stack collapse (L30–L33 inert; 2026-09-10) is a *relative-depth* effect: L32/48 = 0.67 is mid-late at
     12B and behaves like 4B's L22/34 (ratio 0.68 vs 0.74, edit/swap 0.35 vs 0.39, rt/swap 0.30 vs 0.24).
  2. **The edit arm is still not the channel.** `S_edit − S_rt` = +6.05 and `S_edit − S_foreign` = +4.61 [3.13, 6.19]:
     the edit carries a small, CI-clearing, item-specific increment over the unedited round trip — but `S_c3 − S_edit`
     = +14.5, i.e. the raw transported clean state is worth 1.9× what the English edit delivers. Same picture as Phase B
     and the W family: the channel transports, the edit step loses most of it. (The released pair's W1 edit was +19.47
     vs random +19.60 — no gain at all — so our edit arm is *relatively* better than the released pair's, but against
     a much weaker transport.)
  3. **`S_random` = −359 at 12B L32** vs +8 at 4B L32 and −17 at 4B L22. A norm-matched random vector written at 1 459
     positions destroys the 12B reply logp. This is why H-M2 (+375.61 vs thr +9.19) is uninformative here — the
     comparator inflation flagged on 2026-09-10 is worse at 12B; specificity (+4.61) is the real test.
  4. **`S_swap` is pair-independent and lands near the W family's.** +41.47 on the repaired 471-span anchoring vs
     +45.71 (P_patch, original 49-item anchoring) and the repaired-set +41.52 clean-state effect reported on 2026-09-07 —
     consistent, which is a useful check that the traces/spans/host wiring of this run matches the banked W-family runs.
  5. **Cost reality vs the prereg's estimate.** The prereg budgeted "~4–5 h each at 2 GPUs" per 12B layer; actual was
     **28.3 h wall / 56.5 GPU-h** for one layer (AV 18.0 h at 96 s/step under CPU offload, AR 10.1 h). Offload was the
     only configuration that fit (see [`2026-09-11_12b-fsdp-blocked.md`](2026-09-11_12b-fsdp-blocked.md)); the 5–6×
     miss is what drove the user's scope cut. Total Phase C spend incl. extraction and gate ≈ 60 GPU-h for one
     decidable hypothesis.
  6. Silent-failure checks: SELF exactly 0.000 (capture-shape fix holds on a second host); `--traces` pointed at the
     gemma12b bank (60 items confirmed in the job header); `identity_passes` true; no no-tag or CJK reads.

- **New questions / new hypotheses:**
  - **H-C4 (budget dose):** does fidelity rise with SFT data at fixed host/layer? A 2× / 4× row count at 4B L7 (the
    cheapest live layer: 1.2 h/pair) would give a slope; if 0.93 → ~0.98 is reachable with data alone, RL is not the
    missing ingredient. This is the direct follow-up H-C1 makes worth running, and it costs ~5 GPU-h at 4B, not 56.
  - **H-C2 remains open** (12B L7 at 56 GPU-h, or the 16-GPU path). Given H-C1, the cheaper informative experiment
    is H-C4, not more 12B layers.
  - **Relative-depth hypothesis (from obs. 1):** the fidelity profile is a function of relative depth, so 12B's peak
    should sit near L5–L10 (0.10–0.20), matching the 4B's L2–L7. Testable only with 12B early layers — same cost issue.

- **Next Steps:** index this entry (thread README ledger + `log/README.md`); update `CLAUDE_SCRATCHPAD.md` (Phase C
  state → closed on H-C1, H-C2 open). (The obtune jobs held during Phase C — 389288/389289/389290 — already ran to COMPLETED on
  2026-09-10; no hold remains.) H-S4 (accuracy) stays unrun unless asked — pre-declared underpowered.
