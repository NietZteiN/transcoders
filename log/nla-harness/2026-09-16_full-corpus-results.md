### Target Date: 2026-09-16 (H-R14 RESULTS — the whole corpus answers the question, and the answer is no)

- **Hypotheses / what we're testing:** as pre-registered in [`2026-09-16_full-corpus-prereg.md`](2026-09-16_full-corpus-prereg.md)
  (H-R14a–e, thresholds frozen before any of these rows existed). Predictions on record there:
  `DAMAGE-PRESENT` · `CODESTEER-INERT` · `NLA-MATCHES-CODESTEER` · `PROMPT-SUFFICES` · `DAMAGE-FAR-WEAKER`.

- **Setup:** full aligned HumanEval-X Java corpus, **148 snippets / 1 742 cases**, 3 runs/case
  (4.0× the 50/434 of H-R7). `codellama/CodeLlama-7b-Instruct-hf` in the ASE-2026 artifact's own
  `SteeredCausalLM` with the chat template installed; sampler T 0.7 / top_p 1.0 / top_k 7;
  `--deterministic` OFF. Snippet set frozen to `full/snippets_148.json`
  (sha `9056b6b6aba9b0b6…`) before submission; PACKS-PAIRED clean (1 742 cases in both conditions).
  Runner `ase_steer_run.py` sha `cad05d864ac91eaa…` (unchanged from H-R7),
  vectors `ase_vectors.py` sha `2eb5109568691d41…`, config `ase_vectors_full.yaml` sha `18695d2e051b9a54…`,
  scorer `ase_r14_stats.py` sha `e16257bacd379aaa…`, arm script sha `3b9afdc27acbe191…`.
  Jobs 408488/408489 (unsteered pair), 408492–408494 (`swap_oracle`, `prompt`, `erasure`),
  408497–408499 + 408501 (`codesteer` shards 0–3), 408500 + 408502 + 408503 + 408504
  (`codesteer_auto` shards 0–3), 408505 (`ridge_map`), on nodes g-04-02/g-06-01/g-07-08/g-07-09/g-07-11/
  g-08-04/g-08-05, all rc=0, in three `--dependency=singleton` lanes (**peak 3 concurrent GPU jobs,
  verified from `sacct` start/end intervals — never 4**). Smoke 408486 first (3 never-scored snippets ×
  4 code paths). Cluster bootstrap over snippets, `N_BOOT 10 000`, `SEED 20260724`.

  **`ridge_map` vectors, nested grouped cross-fit** (job 408487, CPU, the one real code change): 5 outer
  folds by snippet; per fold, `mu` / mean delta / RRR map / role prototypes **and** the (λ, rank) choice
  fitted on the ~118 out-of-fold snippets only. **All 5 folds PASS**, all select λ=100 / rank=256, held-out
  cosine 0.3908–0.4081 vs mean-only 0.1820–0.1914, margins **+0.2088…+0.2227** (min +0.2088, mean +0.2159)
  against the frozen 0.05 gate; coverage exact (4 799 spans / 148 snippets, every span fitted out-of-fold).
  `crc 1456495885`. Residual arms aligned **148/148, 0 excluded, 4 799 spans** at K=7.
  **Effect gate, both directions:** every `codesteer` shard 2 595–2 866 effective level-2 calls/run
  (L1 attention shift 0.230–0.237), every `codesteer_auto` shard 3 128–3 396 (shift 0.266–0.283). Their
  steering was demonstrably live on all 148.

- **Results:** primary statistic `c/n` (accuracy over cases × all 3 runs, case-weighted). Chance 0.500.

  | arm | acc | Δ vs `unsteered` (95 %) | parse | acc\|parsed | Δ acc\|parsed |
  |---|---|---|---|---|---|
  | `codesteer_auto` | **0.7179** | **+0.0379 [+0.0009, +0.0766]** | 0.932 | 0.7706 | +0.0170 |
  | `original_unsteered` (L0) | 0.7015 | *(+0.0214 = the damage)* | 0.899 | 0.7805 | +0.0269 |
  | `erasure` | 0.6996 | +0.0195 [−0.0233, +0.0634] | 0.917 | 0.7633 | +0.0097 |
  | `swap_oracle` (oracle) | 0.6963 | +0.0163 [−0.0312, +0.0621] | 0.899 | 0.7743 | +0.0207 |
  | `prompt` | 0.6914 | +0.0113 [−0.0346, +0.0572] | 0.924 | 0.7485 | −0.0051 |
  | `unsteered` (L1b) | 0.6801 | — | 0.902 | 0.7536 | — |
  | `ridge_map` (NLA) | 0.6770 | −0.0031 [−0.0446, +0.0383] | 0.899 | 0.7528 | −0.0008 |
  | `codesteer` | 0.6552 | −0.0249 [−0.0765, +0.0249] | 0.853 | 0.7681 | +0.0145 |

  Verbatim verdicts (`full/stats/2026-09-16_r14_full_stats.json`):
  - **H-R14a `DAMAGE-ABSENT`** +0.0214 [−0.0274, +0.0690] (threshold +0.03).
  - **H-R14b `CODESTEER-INERT`** — `codesteer` −0.0249, α/2 CI [−0.0831, +0.0332]; `codesteer_auto` +0.0379,
    α/2 CI [−0.0060, +0.0824]. *At the uncorrected 95 % level `codesteer_auto`'s CI does exclude 0
    ([+0.0009, +0.0766]); the frozen rule requires |Δ| ≥ 0.05 at α/2, which it misses on both counts.*
  - **H-R14c `NLA-MATCHES-CODESTEER`** — `ridge_map − codesteer_auto` = **−0.0409**, α/3 CI
    [−0.0860, **+0.0049**]. **The sign is negative and the uncorrected 95 % CI [−0.0774, −0.0046] excludes
    zero**; only the Bonferroni α/3 interval (barely, by 0.0049) keeps this out of `CODESTEER-BEATS-NLA`.
  - **H-R14d `PROMPT-SUFFICES`** — `ridge_map − prompt` = −0.0144, α/3 CI [−0.0625, +0.0349].
  - **H-R14e `DAMAGE-FAR-WEAKER`** — ours **+2.14 pts** vs their renaming stratum's **+36.29 pts**,
    ratio **0.059**. Ours: orig 70.15 → obf 68.01 → `codesteer_auto` 71.79 / `codesteer` 65.52 /
    `ridge_map` 67.70. Theirs (Qwen2.5-7B, Table 4): 76.49 → 40.20 → 78.30, restoration 104.99 %.
  - **Restoration ratio WITHHELD** by the pre-registered gate (H-R14a returned `DAMAGE-ABSENT`).

  **The 50/98 split — H-R13 settled** (Δ vs `unsteered`, same arms, disjoint snippet halves):

  | arm | the 50 of H-R7 | the 98 NEVER scored | all 148 |
  |---|---|---|---|
  | `ridge_map` | +0.0061 [−0.061, +0.076] | −0.0061 [−0.056, +0.044] | −0.0031 |
  | `codesteer_auto` | +0.0154 [−0.052, +0.087] | **+0.0454 [+0.001, +0.090]** | +0.0379 |
  | `codesteer` | +0.0177 [−0.055, +0.095] | −0.0390 [−0.102, +0.023] | −0.0249 |
  | `swap_oracle` | +0.0515 [−0.025, +0.131] | +0.0046 [−0.053, +0.059] | +0.0163 |
  | `erasure` | −0.0184 [−0.097, +0.067] | +0.0321 [−0.018, +0.082] | +0.0195 |
  | damage | +0.0369 [−0.051, +0.127] | +0.0163 [−0.040, +0.074] | +0.0214 |

  CI half-width on `ridge_map − unsteered`: **0.0739 → 0.0414** (ratio 1.78 vs √(148/50) = 1.72 expected).

- **What worked / hypothesis verdict:**
  - **H-R14a REFUTED my own prediction.** I predicted `DAMAGE-PRESENT`; the corpus returned
    `DAMAGE-ABSENT` (+0.0214, below the +0.03 threshold). The n=50 damage (+0.0584) was **not
    representative** — on the 98 unseen snippets it is +0.0163. Adversarial renaming costs this model
    ~2 points, not the ~36 their Qwen2.5-7B loses.
  - **H-R14b SUPPORTED** (`CODESTEER-INERT`), **H-R14d SUPPORTED** (`PROMPT-SUFFICES`),
    **H-R14e SUPPORTED** (`DAMAGE-FAR-WEAKER`).
  - **H-R14c SUPPORTED as a word, reversed as a fact.** I predicted `NLA-MATCHES-CODESTEER` and got it —
    but at n=50 NLA led by +0.048/+0.063 and at n=148 it **trails by −0.041 with the uncorrected CI
    excluding zero**. The verdict word survives only because Bonferroni widens the interval by 0.0049.
    **The honest reading is that the NLA arm does not beat CodeSteer and may lose to it.**
  - **H-R13 RESOLVED ✗ NOT REPLICATED.** `ridge_map`'s +0.0461 [−0.027, +0.120] from H-R7 is **+0.0061**
    on those same 50 snippets in this run and **−0.0061** on the 98 it had never seen. Two changes could
    carry this (fresh sampling; cross-fitted rather than single-split vectors) and **both point the same
    way**: the +0.046 was winner's curse — the best of six steered arms, selected after the fact, CI
    already spanning zero.
  - **H-R12 ABSORBED.** 148 clusters delivered the predicted precision (half-width ratio 1.78 ≈ √2.96).
    The power calculation is moot: the effect it was sizing does not exist.
  - **The oracle bound is the load-bearing result.** `swap_oracle` — writing the *true* clean-code state at
    L7 — buys **+0.0163**. Every oracle-free L7 latent arm must sit below that, and they do
    (`erasure` +0.0195 is within noise of it, `ridge_map` −0.0031). **There is ~2 points of headroom at
    this site, so no amount of better vector construction can produce a CodeSteer-beating result here.**

- **Observations:**
  - **`codesteer_auto`'s gain is mostly format compliance, not reasoning.** Its Δacc is +0.0379 but its
    Δparse is **+0.0293** and its Δ(acc | parsed) only **+0.0170** — roughly half the headline. Mirror
    image on plain `codesteer`: parse **drops** −0.0494 (0.853, the worst of any arm) which is what makes
    its accuracy −0.0249, while its conditional accuracy is *positive* (+0.0145). **Both variants nudge
    conditional accuracy by +0.014/+0.017; what separates them is what they do to whether the model
    answers at all.** This is the same trap that voided `role_proto`'s apparent win at H-R9 and it is why
    parse rate is reported beside every arm.
  - **Run-to-run variation exceeds what the snippet bootstrap reports.** On the *same* 50 snippets,
    between H-R7 and today: `ridge_map` +0.0461 → +0.0061, `codesteer_auto` −0.0023 → +0.0154,
    `codesteer` −0.0169 → +0.0177. That is ±0.02–0.04 of movement from resampling generations alone,
    which the cluster bootstrap over snippets **does not capture** (it resamples snippets, not seeds).
    **Every interval in this family is therefore optimistic**, and no effect below ~0.04 should be
    believed from a single run at T 0.7. Raises **H-R15**.
  - `codesteer_auto` is the only arm positive on both halves, and its 98-snippet CI [+0.001, +0.090]
    excludes zero. It is the single most credible steering effect anywhere in this programme — and it is
    **theirs, not ours**, it is ~+0.02 after conditioning on parse, and it fails the pre-registered bar.
  - The damage is too small to support a restoration ratio, so their headline metric is simply not
    computable on this model. That is a statement about the precondition, not about their result.
  - Declared and unchanged: our renamer is not theirs, our model is in none of their four rows, and only
    1 of their 4 obfuscation families is testable here. **Nothing here refutes their table.**

- **New questions / new hypotheses:**
  - **H-R15 (raised, free):** the generation-seed noise floor. Re-run 2–3 arms at a second seed on the same
    148 and report a seed-resampled interval beside the snippet-clustered one. Until it exists, treat every
    CI in this family as a lower bound on the true uncertainty. **This is now the highest-value cheap run
    in the thread**, because it sets the bar every future steering claim must clear.
  - **H-R16:** is `codesteer_auto`'s +0.029 parse gain a real compliance mechanism (Eq. 10's head subset
    stabilising the output format) or a decoding artifact? Testable by scoring format-validity separately
    from answer correctness on the banked rows — **no GPU**.
  - **H-R17:** the damage is ~2 pts on CodeLlama-7B-Instruct against their ~36 on Qwen2.5-7B. With the
    Chinese-model bar this cannot be closed by running their model. H-R3's stronger renamer is the only
    lever left, and it now has a 148-snippet baseline to move.
  - CruxEval-X packs are built (**698 snippets / 1 396 cases / 2.00 per snippet** vs their 698 / 1 378 /
    1.97, job 408482) but **no GPU arm has run there**; that needs its own pre-registration, and given
    `DAMAGE-ABSENT` here the first question is whether renaming damages this model on CruxEval at all.

- **Next Steps:** report + artifact for the user; H-R15 seed floor before any further steering claim;
  decide on CruxEval only after H-R15.
