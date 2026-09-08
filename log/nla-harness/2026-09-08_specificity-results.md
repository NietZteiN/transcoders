### Target Date: 2026-09-08 (H-W35 — `W35-INTERMEDIATE`: the route is graded by content, my "generic transport" prior was wrong, and a defect cost the joint sub-hypothesis)

**Thread:** nla-harness · **Job:** 383160 (h200, g-08-05, **1:49:41**, COMPLETED 0:0) ·
**Resolves:** [`2026-09-08_specificity-prereg.md`](2026-09-08_specificity-prereg.md) (a, b, c;
**d not run — see defect below**) · **Artifacts:** `data/nla/p0/w35/gemma12b/heads_rows.jsonl`
(180 rows), scored against the banked `data/nla/p0/heads/gemma12b/heads_rows.jsonl` from job 382366.

- **Hypothesis.** H-W31 found eight components carrying 60 % of the transported state, but every arm
  it ran carried the *same content*. Writing **different content at the same 1,459 positions**
  separates "a circuit that carries identifier meaning" from "where anything written here arrives".

- **Setup.** Three null arms built from the banked vectors — `N_sibling` (another span of the same
  item), `N_foreign` (another item's clean span), `N_random` (a random unit direction) — swept over
  all 255 components of layers 33–47, 60 items each. 93,660 forwards at 67 ms. Identity gate
  **exact again**: `SELF` 0.0000, `ALL` 0.0000, `ko_gap` 0.0000.

- **Results.** Whole-effect magnitudes first, as a construct check — the arms reproduce the banked
  null battery (job 378019) from an independent construction:

  | arm | dG_S here | banked |
  |---|---|---|
  | `C3pure` | +41.04 | — |
  | `P_patch` | +41.52 | — |
  | `N_sibling` | **+37.98** | +39.94 |
  | `N_foreign` | **+12.99** | +15.18 |
  | `N_random` | **−298.96** | −365.59 |

  **Routing, against `C3pure`'s 255-component sufficiency profile:**

  | comparison | Spearman | Jaccard top-16 |
  |---|---|---|
  | vs `P_patch` — same content, written raw | 0.838 | 0.78 |
  | vs **`N_sibling`** — same item, wrong span | **0.847** | 0.68 |
  | vs **`N_foreign`** — wrong item | **0.743** | 0.60 |
  | vs **`N_random`** — no content | **0.283** | 0.19 |

  Top-8 by sufficiency:

  | arm | components |
  |---|---|
  | `C3pure` | L41H4, L34M, L45H3, L47H14, L47H4, L41H5, L40M, L44H1 |
  | `N_sibling` | L41H4, L45H3, L47H14, L47H4, L41H5, L34M, L47H2, L44H1 |
  | `N_foreign` | L41H4, L47H4, L45H3, L47H1, L47H7, L47H3, L47H14, L47H2 |
  | **`N_random`** | **L46H7, L46H0**, L47H5, L47H15, L47H3, L47H1, L47H4, L47H14 |

- **Hypothesis verdicts.**
  - **H-W35a → `W35-INTERMEDIATE`.** ρ(C3pure, N_foreign) = **0.743**, below the 0.80 `W35-GENERIC`
    bar; Jaccard 0.60, above its 0.50. Not generic, not content-specific. **My pre-registered prior
    was `W35-GENERIC` and it is wrong.**
  - **H-W35b ✓ the ordering holds, and it is monotone in content distance.**
    ρ(sibling) **0.847** ≥ ρ(foreign) **0.743**, as required. The full ladder — 0.847 same item,
    0.743 wrong item, 0.283 no content — was not something the rule demanded, and it is the finding.
  - **H-W35c — descriptive, and sharper than expected.** `N_random` does **not** load the same
    components: ρ 0.283, Jaccard 0.19, and its top-8 is led by **L46H7 and L46H0**, which appear in
    no content arm's top-8. A catastrophic write travels a different path from a meaningful one.
  - **H-W35d — NOT RUN.** See the defect.

- **Defect #12 — `--arms` did not reach the joint stage's item selector, and the stage failed
  silently.** `nla_heads.py:418` read `sids = [r for r in hrows if r["arm"] == ARMS[0]]`, where
  `ARMS[0]` is the module constant `"C3pure"`. With `--arms N_sibling,N_foreign,N_random` no banked
  row matches, `sids` is empty, the loop body never executes — and the stage printed
  **`joint done: 0 forwards, 0.6 min` and exited 0**. Same family as bugs #8 and #9: *nothing
  distinguished "did no work" from "did work that produced nothing"*. The sweep was unaffected
  (its loop was correctly switched to `args.arm_list`), so H-W35a/b/c are intact and only the
  descriptive H-W35d is lost. Fixed by selecting the first arm actually present in the rows and
  **refusing with exit 2** when none is. The lesson `arm_guard` already encodes for arms now needs
  encoding for *stages*: a stage that performs zero units of work should be an error, not a success.

- **Observations.**
  1. **The honest headline is a gradient, not a dichotomy, and it dissolves the question as I posed
     it.** I designed H-W35 as generic-vs-specific and the data says *both, in proportion to how much
     the content differs*: 0.85 for the same item's other span, 0.74 for a different item, 0.28 for
     no content at all. The late global heads are neither a content-blind dump nor a semantic
     circuit — they are a **content-sensitive transport path**, and how much of it a write uses
     scales with how much like the true clean state that write is.
  2. **H-W31's localisation survives, with its description corrected.** It is still true that eight
     components carry 60 % of the transported state and that `L41H4` leads. What is no longer
     available is "these heads carry identifier *meaning*" — a wrong-item activation, carrying the
     wrong meaning entirely, still uses 60 % of the same top-16.
  3. **`N_random` is the load-bearing control, and it is the one I nearly filed as descriptive
     padding.** Without it, ρ = 0.743 could be read as "everything routes alike, 0.74 is just what
     this metric gives". The random arm shows the metric can go to 0.28 on this very corpus, so
     0.743 is a real similarity and 0.847-vs-0.743 is a real difference.
  4. **`N_sibling` correlates with `C3pure` (0.847) slightly *more* than `P_patch` does (0.838).**
     Different span, same item, routed marginally more like the NLA state than that span's own raw
     state. Consistent with H-W12 — the causal content is item-level, not span-level — and a small
     reminder that the arms differ in encoding as well as content.

- **New questions.**
  - **H-W39:** the gradient is measured on three points chosen for other reasons. A designed dose —
    interpolate between a span's own clean state and a foreign one, and watch ρ fall — would say
    whether route-similarity is smooth in content-similarity or has a threshold.
  - **H-W35d still owed**, now cheap: with the fix, select the top-8 on `C3pure`'s opposite half and
    measure the fraction of `N_foreign`'s +12.99 they recover. That is the magnitude counterpart of
    the rank result above.
  - **H-W36b's motivation is restored.** The prereg said it should probably not run if the heads
    turned out generic. They are not generic — so asking whether the reconstruction error lies along
    what `L41H4` reads is meaningful again.

- **Bounds.** Rank comparisons over item-mean profiles; no CI is attached to the Spearman values
  themselves, so "0.847 > 0.743" is an ordering, not a tested difference — H-W39 is the way to make
  it one. One host, one layer, one tier, 60 items. `N_random`'s profile is the profile of *damage*
  and should not be read as a transport path for anything.
