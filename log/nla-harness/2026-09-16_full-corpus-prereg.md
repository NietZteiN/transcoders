### Target Date: 2026-09-16 (H-R14 — full HumanEval-X corpus vs the paper's own numbers)

- **Hypotheses / what we're testing:**

  Everything in the 50-snippet bake-off ([2026-09-15_chat-template-rerun-results.md](2026-09-15_chat-template-rerun-results.md))
  came back **provisional** for one reason: at n=50 the cluster-bootstrap CI on every steering contrast is
  ~±0.10 wide while the effects at stake are ~0.05. This entry spends the corpus we already have — **148
  snippets / 1,742 cases**, 4.0× the 434 cases scored so far — to resolve them, and adds the comparison the
  user asked for: **our numbers against the paper's own Table 2 / Table 4 rows.**

  This is a scale-up of a frozen design, not a new one. **H-R14a–d re-ask H-R7's questions at 4× n with the
  same frozen thresholds**; H-R14e is the new cross-setup comparison.

  - **H-R14a (damage).** Adversarial renaming costs CodeLlama-7B-Instruct accuracy on the full corpus.
    `DAMAGE-PRESENT` if `original_unsteered − unsteered` ≥ +0.03 with the 95 % CI excluding 0 ·
    `DAMAGE-WEAK` if ≥ +0.03 with the CI containing 0 · `DAMAGE-ABSENT` if < +0.03.
    *Prediction: `DAMAGE-PRESENT`.* At n=50 it was `DAMAGE-WEAK` (+0.0584 [−0.019, +0.140]); the point
    estimate is 2× the threshold and the CI should shrink ~2× with 4× the clusters, so if the point estimate
    holds this clears. **If it lands `DAMAGE-ABSENT`, H-R14b/e are unanswerable and are reported as such** —
    a restoration ratio with a ~0 denominator is not a number (see the gate below).
  - **H-R14b (their arms).** CodeSteer, in its own runtime at its own settings, changes accuracy on the
    renamed corpus. `CODESTEER-RESTORES` if `codesteer − unsteered` ≥ +0.05 with the Bonferroni-adjusted CI
    (α/2, two comparators: `codesteer`, `codesteer_auto`) excluding 0 · `CODESTEER-INERT` if |Δ| < 0.05 ·
    `CODESTEER-HARMS` if ≤ −0.05 with the adjusted CI excluding 0.
    *Prediction: `CODESTEER-INERT` again* (n=50 gave −0.0169 and −0.0023, both CIs spanning 0). A flip to
    `CODESTEER-RESTORES` at 4× n is the outcome that would overturn our headline, and is exactly why this runs.
  - **H-R14c (ours vs theirs — the user's question).** Our best latent arm beats their attention steering.
    Comparator is the BETTER of `codesteer` / `codesteer_auto` by the primary statistic.
    `NLA-BEATS-CODESTEER` if `ridge_map − best_codesteer` ≥ +0.05 with the α/3 CI (three comparisons:
    vs `codesteer`, vs `unsteered`, vs `prompt`) excluding 0 · `NLA-MATCHES-CODESTEER` if |Δ| < 0.05 ·
    `CODESTEER-BEATS-NLA` if ≤ −0.05 with that CI excluding 0.
    *Prediction: `NLA-MATCHES-CODESTEER`.* n=50 gave +0.0630 vs `codesteer` / +0.0484 vs `codesteer_auto`,
    both CIs spanning 0. **This is the pre-registered answer to "does NLA beat CodeSteer" and it is declared
    before the data exists.**
  - **H-R14d (the baseline that keeps winning).** `PROMPT-SUFFICES` if `ridge_map − prompt` < +0.05 or its
    α/3 CI contains 0 · `LATENT-BEATS-PROMPT` if ≥ +0.05 with that CI excluding 0.
    *Prediction: `PROMPT-SUFFICES`* (n=50: +0.033). CLAUDE.md §4 makes this baseline mandatory, and at n=50 it
    is the finding that blocks every steering claim we could otherwise make.
  - **H-R14e (cross-setup comparison to the paper).** Descriptive, NOT a replication test. Their
    Qwen2.5-7B identifier-renaming row is orig **76.49** → obf **40.20** → steered **78.30**, restoration
    **104.99 %** (Table 4); their aggregate HumanEval Pass@1 row is 76.49 / 64.01 / 77.38, restoration
    **107.1 %** (Table 2). We report the same five quantities for CodeLlama-7B-Instruct.
    `DAMAGE-FAR-WEAKER` if our damage is < 1/3 of their 36.3-point renaming drop ·
    `DAMAGE-COMPARABLE` if within [1/3, 3×].
    *Prediction: `DAMAGE-FAR-WEAKER`* (n=50 implies ~5.8 points vs their 36.3). **Their four models are all
    outside our permitted set and their renaming generator is not in the artifact, so a difference here is
    un-attributable between model, renamer and protocol.** It is reported as a gap to explain, never as a
    refutation of their table.

  **Restoration-ratio gate (frozen).** `R = (steered − obf) / (orig − obf)` is reported **only if** H-R14a
  returns `DAMAGE-PRESENT`. Otherwise the ratio is omitted and raw deltas are reported instead. Rationale:
  the denominator is the damage; at `DAMAGE-WEAK`/`ABSENT` the ratio's bootstrap distribution is heavy-tailed
  around a near-zero denominator and the point estimate is arbitrary. Their 104.99 % / 107.1 % rest on a
  36.3-point denominator, so the comparison is only meaningful if ours is real too.

- **Setup:**

  **Corpus (already built, no new generation).** `packs_humaneval_java.jsonl` (164 snippets / 1,930 cases)
  and `packs_renamed_humaneval_java.jsonl` (156 / 1,784) were execution-validated with the artifact's own
  `evaluation/java_counterfactual.build_case_pack`. The run set is the **148 snippets that survive residual
  alignment** — frozen to `full/snippets_148.json` from `pool_codellama7b_L7_chat.pt`, whose 8 exclusions are
  recorded verbatim (7 occurrence mismatches, 1 un-renamed method name `Java_106`). Renamed and original
  packs agree at **1,742 cases each**, satisfying the PACKS-PAIRED gate. **Every arm runs this identical
  148-snippet set** so `paired()` never silently intersects.

  **Model.** `codellama/CodeLlama-7b-Instruct-hf`, their `SteeredCausalLM`, chat template installed
  (`install_chat_template`, ids gated against `apply_chat_template`) — the single substitution carried from
  H-R7. Sampler T 0.7 / top_p 1.0 / top_k 7, 3 runs/case, `--deterministic` OFF. Chinese-model refusal active.

  **Arms (8).** `original_unsteered` (L0 = damage denominator) · `unsteered` (L1b baseline) · `codesteer` ·
  `codesteer_auto` (Eq. 10 top-4 heads) · `ridge_map` (cross-fitted, below) · `erasure` (mean-difference
  latent baseline that `ridge_map` nests) · `prompt` (mandatory text baseline) · `swap_oracle` (oracle upper
  bound on an L7 write). **Dropped vs H-R7:** `codesteer_beta0`. Its job was the identity floor; that was
  established at n=50 (−0.0376, PASS) and the two-sided effect gate re-verifies identity per call
  (`l2_effective_calls == 0`) on every run, so 8 GPU-h of re-confirmation buys nothing. Also dropped:
  `rand_prior`, `uniform_prior`, `ast_prior`, `foreign`, `role_proto`, `combined`, `prompt_types` — all
  settled at n=50 and none load-bearing for H-R14a–e.

  **`ridge_map` must be re-fitted, and this is the one real code change.** At n=50 the fit pool was the 106
  non-test snippets; with all 148 in the test set that pool is empty. Replaced by **nested grouped
  cross-fitting** in `ase_vectors.py --cross-fit`: outer 5 folds by snippet (seeded, `seed 20260724`); for
  each fold the nuisance quantities (`mu`, mean delta `c`, the RRR map `A`, and the `role_proto` prototypes)
  are fitted on the **out-of-fold snippets only**, then applied to the held-out fold. λ/rank are selected by
  the existing grouped 5-fold `cv_select` **inside each outer training fold**, so not even the two
  hyperparameters see a snippet's own data. Measured cost ~2–3 h CPU (one `rrr_fit` at n=4,799 is 20.7 s),
  **no GPU**. The frozen pre-GPU gate is unchanged: the selected map must beat mean-only held-out cosine by
  **≥ 0.05** in every outer fold, else `ridge_map` is not run. Config: `configs/ase_vectors_full.yaml`
  (new file; the H-R7 config is not edited).

  **Sharding.** Only the two CodeSteer arms are sharded (4 shards each) — at n=50 they took 1 h44–2 h20 for
  attention recording, so 148 snippets is ~7–9 h, past the 3 h wall. Shards are disjoint snippet sets and the
  arms carry no cross-snippet state, so concatenating the shard jsonls is exact. **`erasure` is deliberately
  NOT sharded:** its vector is a leave-one-ITEM-out mean over the other snippets *passed to the runner*, so a
  shard would silently redefine the arm from LOO-over-147 to LOO-over-36.

  **Scoring.** `ase_bakeoff_stats.py --scoring cn` (primary, Chen et al. `c/n` over all 3 runs, case-weighted
  `Σ p_i n_i / Σ n_i`) **and** `--scoring pass1` (their run-1 Pass@1) so the paper's columns line up; unparsed
  = wrong; chance 0.500; parse rate reported separately per arm. Cluster bootstrap over the 148 snippets,
  `N_BOOT 10 000`, `SEED 20260724`. Bonferroni α/2 for H-R14b and α/3 for H-R14c/d, as frozen at H-R7.

  **Budget.** 2 CodeSteer arms × 4 shards × ~2 h + 6 arms × ~1 h ≈ **22 GPU-h**, ≤ 3 concurrent jobs
  (`--job-name=r14_lane{1,2,3} --dependency=singleton`) ⇒ ~8 h wall. CPU-only in parallel: the cross-fit
  (~3 h) and the CruxEval-X pack build.

  **CruxEval-X (stage 2, exploratory, CPU first).** Their second dataset: 698 Java snippets / 1,378 cases.
  `evaluation/java_counterfactual` does support it (`_extract_cruxeval_seeds`, `resolve_task_profile`
  hard-switches `cruxeval` to `counterfactual_tf`), but our pack builder has never been run on it, so
  yield is unknown. Pack building is CPU (javac + java per case) and produces **no model outcome data**, so
  it starts now; any GPU arms on CruxEval are a separate pre-registration once the yield is known.
  Their comparison rows: Table 2 CruxEval 83.11 / 71.40 / 80.43 (restoration 77.1 %), Table 4 renaming
  83.11 / 74.51 / 85.48 (127.50 %).

- **Results:** pending — this entry is the pre-registration. Results land in a separate dated entry.

- **What worked / hypothesis verdict:** pending.

- **Observations (recorded before the data, so they cannot be retrofitted):**
  - The corpus was always there. H-R7 ran 50 of 156 available snippets because it was a bake-off across 15
    arms; the cost of 15 arms bought arm coverage instead of n. Having settled 11 of those arms, the same GPU
    budget now buys n on the 8 that matter. **H-R12's power question is answered by running, not by a
    calculation** — 148 clusters is inside the 150–200 band that calculation was going to name.
  - **H-R13 is absorbed.** It asked whether `ridge_map`'s +0.046 survives a fresh seed. The cross-fit refits
    the map on a different (nested) partition over 3× the fit data and evaluates it on 98 snippets it has
    never been scored on, which is a strictly stronger replication than a seed change.
  - The outcome most likely to embarrass the headline is H-R14b flipping to `CODESTEER-RESTORES`. Stating
    that now, with the threshold frozen, is the point of pre-registering.
  - Known un-attributable confounds, declared: our renamer is not theirs (their generator and equivalence
    checker are excluded from the artifact), our model is in neither of their model rows, and only 1 of their
    4 obfuscation families is testable here.

- **New questions / new hypotheses:** deferred to the results entry. H-R11 (fixation diagnostic) untouched.

- **Next Steps:** freeze the 148-snippet set and shard files → launch CruxEval pack build (CPU) → land the
  cross-fit and check its per-fold gate → submit the 6 unfitted arms → submit `ridge_map`/`erasure` once the
  gate passes → score both statistics → results entry + report update.
