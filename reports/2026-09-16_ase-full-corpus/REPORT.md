# Steering CodeLlama-7B on the full ASE-2026 corpus: NLA vs CodeSteer

*Run 2026-09-16 · 148 snippets / 1,742 cases · pre-registered in `log/nla-harness/2026-09-16_full-corpus-prereg.md` · results in `log/nla-harness/2026-09-16_full-corpus-results.md`*

> ### ⚠ Amended 2026-09-17 — the one positive effect did not replicate
>
> A pre-registered replication ([H-R15](../../log/nla-harness/2026-09-17_replication-floor-results.md))
> re-ran these arms on the same 148 snippets changing **only the sampling draw**.
> **`codesteer_auto`'s +0.0379 became −0.0031.** The single arm whose interval excluded zero in this
> report does not survive a different draw, and its parse-compliance component (§5) did not either
> (Δparse +0.0293 → −0.0019). The pipeline is *exactly* reproducible at a fixed seed (per-item agreement
> 1.0000), but **38 % of individual case labels flip** when the draw changes, and contrast drift is
> **1.08×** this report's own half-width for `codesteer_auto`.
>
> **Read every interval below as ≈1.2× wider than printed**, and treat no single-run contrast here as an
> effect. What survives unchanged: the **oracle bound** (§4) — an upper bound, not a contrast — and
> therefore the conclusion that no better L7 vector can beat CodeSteer. What is withdrawn: any reading in
> which `codesteer_auto` is "the most credible steering effect in this programme". **Nothing in this
> bake-off survives replication — ours or theirs.**

**Question asked:** does accuracy improve more under our NLA/residual steering than under CodeSteer, on the paper's own dataset, whole?

**Answer: no.** On all 148 aligned snippets the best NLA arm sits *below* the better CodeSteer arm by −0.041, and the +0.046 lead it showed on 50 snippets does not replicate. Neither method clears the pre-registered bar, and the reason is structural: there is only ~2 points of accuracy to win at this intervention site.

---

## 1. The table

Primary statistic `c/n` — accuracy over all cases × all 3 runs, case-weighted. Chance = 0.500. **NLA/latent arms in bold.**

| arm | what it is | acc | Δ vs `unsteered` (95 %) | parse | acc·given·parsed | Δ acc·g·p |
|---|---|---|---|---|---|---|
| `codesteer_auto` | theirs, Eq. 10 top-4 heads | **0.7179** | **+0.0379 [+0.0009, +0.0766]** | 0.932 | 0.7706 | +0.0170 |
| `original_unsteered` | L0, un-obfuscated ceiling | 0.7015 | *(+0.0214 = the damage)* | 0.899 | 0.7805 | +0.0269 |
| **`erasure`** | **ours — LOO mean-difference write, L7** | 0.6996 | +0.0195 [−0.0233, +0.0634] | 0.917 | 0.7633 | +0.0097 |
| **`swap_oracle`** | **ours — ORACLE: true clean state at L7** | 0.6963 | +0.0163 [−0.0312, +0.0621] | 0.899 | 0.7743 | +0.0207 |
| `prompt` | one line of text, no steering | 0.6914 | +0.0113 [−0.0346, +0.0572] | 0.924 | 0.7485 | −0.0051 |
| `unsteered` | L1b baseline (do nothing) | 0.6801 | — | 0.902 | 0.7536 | — |
| **`ridge_map`** | **ours — cross-fitted reduced-rank ridge, L7** | 0.6770 | −0.0031 [−0.0446, +0.0383] | 0.899 | 0.7528 | −0.0008 |
| `codesteer` | theirs, as published (β 0.8, L24–31) | 0.6552 | −0.0249 [−0.0765, +0.0249] | 0.853 | 0.7681 | +0.0145 |

## 2. Verdicts, against thresholds frozen before the data existed

| hypothesis | verdict | number | predicted? |
|---|---|---|---|
| **a** damage | `DAMAGE-ABSENT` | +0.0214 [−0.0274, +0.0690] | ✗ **I predicted `DAMAGE-PRESENT`** |
| **b** their arms | `CODESTEER-INERT` | −0.0249 / +0.0379, α/2 CIs span 0 | ✓ |
| **c** ours vs theirs | `NLA-MATCHES-CODESTEER` | **−0.0409**, α/3 CI [−0.0860, **+0.0049**] | ✓ as a word, **sign reversed** |
| **d** vs prompting | `PROMPT-SUFFICES` | −0.0144, α/3 CI [−0.0625, +0.0349] | ✓ |
| **e** vs the paper | `DAMAGE-FAR-WEAKER` | ours +2.14 pts vs theirs +36.29, ratio 0.059 | ✓ |
| restoration ratio | **WITHHELD** | gate: only reported if damage is real | — |

Two of these deserve to be read carefully rather than at the label:

- **H-R14c is the user's question, and the word flatters us.** `NLA-MATCHES-CODESTEER` fires because |−0.0409| < 0.05. But the sign is negative, and the *uncorrected* 95 % interval [−0.0774, −0.0046] **excludes zero**. Only the Bonferroni α/3 widening — by 0.0049 — keeps this out of `CODESTEER-BEATS-NLA`. The defensible statement is: **NLA does not beat CodeSteer here, and may lose to it.**
- **H-R14b is generous to CodeSteer in the other direction.** `codesteer_auto`'s +0.0379 *does* clear zero at the uncorrected 95 % level ([+0.0009, +0.0766]). It fails the frozen rule on magnitude (< 0.05) and on the Bonferroni interval. It looked like the most credible steering effect in this programme — and it is theirs. **It did not replicate (amended 2026-09-17): +0.0379 → −0.0031 at a different draw.**

## 3. Comparison to the paper's own numbers

| | orig | obfuscated | + steering | restoration |
|---|---|---|---|---|
| **Theirs**, Qwen2.5-7B, identifier renaming (Table 4) | 76.49 | 40.20 | 78.30 | 104.99 % |
| **Theirs**, Qwen2.5-7B, all 4 obfuscations (Table 2) | 76.49 | 64.01 | 77.38 | 107.1 % |
| **Ours**, CodeLlama-7B-Instruct, renaming | 70.15 | 68.01 | 71.79 (`codesteer_auto`) / 65.52 (`codesteer`) / 67.70 (`ridge_map`) | **withheld** |

Their renaming drop is **36.29 points**; ours is **2.14**. A restoration ratio divides by that drop, so on our model the denominator is indistinguishable from zero and the ratio is not an estimate — it is withheld by a rule frozen before the run, not by hindsight.

**This is not a refutation of their table.** Three differences are un-attributable: all four of their models are outside our permitted set, their renaming generator and equivalence checker are not in the released artifact (ours is our own), and only 1 of their 4 obfuscation families is testable here. What we can say is that *on a model we are allowed to run, the precondition their method needs — a large renaming deficit — is absent.*

## 4. Why no better vector could have won: the oracle bound

`swap_oracle` writes the **true clean-code hidden state** at layer 7 — the thing every oracle-free vector is trying to approximate. It buys **+0.0163**.

That caps the whole family. `erasure` (+0.0195) is already at the bound within noise; `ridge_map` (−0.0031) is below it despite a cross-fit whose held-out cosine to the true delta is 0.39–0.41 against a mean-only baseline of 0.18. **The vectors got better and the accuracy did not move**, because there are only ~2 points available at this site. Further vector engineering at L7 cannot produce a CodeSteer-beating number.

## 5. What CodeSteer actually does here: compliance, not reasoning

Splitting accuracy into *does the model answer* and *is the answer right*:

| arm | Δ accuracy | Δ parse rate | Δ accuracy **given parsed** |
|---|---|---|---|
| `codesteer_auto` | +0.0379 | **+0.0293** | +0.0170 |
| `codesteer` | −0.0249 | **−0.0494** | **+0.0145** |

Both variants nudge conditional accuracy by about the same small amount (+0.015 to +0.017). What separates them — and what drives the entire headline difference between +0.038 and −0.025 — is **whether the model emits a parseable answer at all**. `codesteer`'s parse rate of 0.853 is the worst of any arm. This is the same trap that voided an apparent `role_proto` win earlier in this programme, which is why parse rate now sits beside every arm.

## 6. The replication that settles the previous result

At 50 snippets `ridge_map` led by +0.0461. Splitting today's run by snippet half:

| arm | the 50 from the earlier run | the 98 **never scored before** | all 148 |
|---|---|---|---|
| **`ridge_map`** | **+0.0061** [−0.061, +0.076] | **−0.0061** [−0.056, +0.044] | −0.0031 |
| `codesteer_auto` | +0.0154 [−0.052, +0.087] | **+0.0454 [+0.001, +0.090]** | +0.0379 |
| `codesteer` | +0.0177 [−0.055, +0.095] | −0.0390 [−0.102, +0.023] | −0.0249 |
| `swap_oracle` | +0.0515 [−0.025, +0.131] | +0.0046 [−0.053, +0.059] | +0.0163 |
| damage | +0.0369 [−0.051, +0.127] | +0.0163 [−0.040, +0.074] | +0.0214 |

`ridge_map`'s +0.046 became +0.006 **on the very same snippets**. It was the best of six steered arms, selected after the fact, with a CI that already spanned zero — textbook winner's curse.

## 7. A caveat that applies to every number in this programme

Between the two runs, on the *same* 50 snippets, arms moved by ±0.02–0.04 from resampling generations alone (`ridge_map` +0.0461 → +0.0061; `codesteer` −0.0169 → +0.0177; `codesteer_auto` −0.0023 → +0.0154). The cluster bootstrap resamples **snippets**, not generation seeds, so it cannot see this. **Every interval reported here is therefore a lower bound on the true uncertainty**, and no effect below ~0.04 from a single sampled run at T 0.7 should be believed. Pinning that floor (H-R15) is the next run.

## 8. Method

Same protocol as the 50-snippet report (`reports/2026-09-15_ase-bakeoff/REPORT.md` §5), with three changes:

- **Corpus.** All 148 snippets surviving residual alignment out of 156 renamed packs; frozen to `full/snippets_148.json` before submission, 8 exclusions recorded verbatim. Renamed and original packs agree at 1,742 cases each (PACKS-PAIRED).
- **`ridge_map` is now cross-fitted.** With every snippet under test the old fit pool is empty, so vectors come from **nested grouped cross-fitting**: 5 outer folds by snippet; per fold the centring, mean delta, reduced-rank ridge map, role prototypes **and** the (λ, rank) choice are fitted on the ~118 out-of-fold snippets only. All 5 folds selected λ=100 / rank=256 and passed the frozen 0.05 margin gate at +0.209 to +0.223; all 4,799 spans covered. A regression gate first confirmed the refactored code reproduces the earlier fit exactly.
- **Both statistics reported.** `c/n` is primary; their run-1 Pass@1 is in the stats JSON for column-matching. On Pass@1 the ranking is unstable (it puts `ridge_map` last and `codesteer_auto` first) — a single sampled run is not a reliable ranking, which is why `c/n` is primary.

Effect gate two-sided and live on all 148: `codesteer` 2,595–2,866 effective level-2 attention calls/run, `codesteer_auto` 3,128–3,396. Residual arms aligned 148/148, 0 excluded. Cluster bootstrap over snippets, 10,000 resamples, seed 20260724, Bonferroni α/2 and α/3.

## 9. Provenance

14 GPU jobs 408488–408505, nodes g-04-02 / g-06-01 / g-07-08 / g-07-09 / g-07-11 / g-08-04 / g-08-05, all rc=0, three `--dependency=singleton` lanes with **peak 3 concurrent GPU jobs** (verified from scheduler start/end intervals). Smoke on 3 never-scored snippets first. Cross-fit job 408487 (CPU). Shas: runner `cad05d864ac91eaa…` (unchanged from the previous run), `ase_vectors.py` `2eb5109568691d41…`, `ase_r14_stats.py` `e16257bacd379aaa…`, config `18695d2e051b9a54…`, snippet set `9056b6b6aba9b0b6…`.

**Also built, not yet run:** CruxEval-X Java packs — 698 snippets / 1,396 cases / 2.00 per snippet, against the paper's 698 / 1,378 / 1.97. Our builder reproduces their second corpus closely. No GPU arm has run there; given `DAMAGE-ABSENT` here, the first question is whether renaming damages this model on CruxEval at all.
