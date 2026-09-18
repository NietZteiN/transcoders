### Target Date: 2026-09-18 (H-R22 — a stimulus that actually damages: the identifier derangement)

- **Hypotheses / what we're testing:**

  H-R18 left the programme blocked, not on method but on **stimulus**. Under greedy decoding on 148
  snippets the renaming damage is **+0.0069** and the oracle ceiling — writing the *true* clean state at
  L7 — is **+0.0052**. There is nothing for any steering method to restore, so no steering result on this
  corpus can be informative, however good the vector (H-R21). Before spending another GPU-hour on
  steering, the deficit has to exist.

  Two diagnosed weaknesses of the renamer that produced every banked number:
  1. `len(n) > 1` skipped **every single-character identifier** — most loop counters and accumulators in
     HumanEval-X Java, i.e. exactly the names a reader uses to track state.
  2. Decoys were assigned by `rng.sample(pool)` — **at random**. A randomly-paired decoy is *unfamiliar*,
     not *misleading*. That makes the banked stimulus closer to this project's **L1 (nonsense renaming)**
     than to **L1b (adversarial renaming, which injects plausible-but-WRONG semantics)**. The ladder in
     `CLAUDE.md` §3 distinguishes these precisely because they are different failure routes.

  **The manipulation, and why it is a cleaner control than the pooled-decoy version.** Permute each
  file's own declared identifiers among themselves by a seeded **derangement** (Sattolo ⇒ one cycle ⇒ no
  fixed point). Every identifier keeps a name that is plausible *in this very program* while being
  attached to the wrong entity. Observed on `Java_000`: the method becomes `j`, a parameter becomes
  `hasCloseElements`, the loop index becomes `numbers`, the computed distance becomes `j`'s parameter.
  Crucially the **set of identifier names in the file is exactly preserved**, so any damage **cannot** be
  attributed to unfamiliar or out-of-distribution tokens — no new token is introduced. It isolates
  *wrong semantics* from *strange vocabulary*, which the pooled-decoy renamer conflates. Semantics are
  preserved by construction (a bijective rename of declarations), and that is validated rather than
  trusted.

  - **H-R22a (corpus, gate).** `CORPUS-OK` if ≥ 100 of the frozen 148 snippets survive
    derangement + parse + harness-pattern + execution-validated pack rebuild + PACKS-PAIRED (same case
    count and same `case_id → expected_bool` sequence as the original) · `CORPUS-TOO-SMALL` otherwise.
    **If this fails, H-R22b is not read** — a damage figure on a biased surviving subset is not a damage
    figure. *Prediction: `CORPUS-OK`*, with some attrition from declared names that collide with library
    members (a local named `size` alongside `list.size()`); the member-access guard holds those back for
    variables, while **method names are always renamed** because the method name is what appears inside
    every case expression, and the rare same-named library call is caught by compile validation.
  - **H-R22b (primary — is there a deficit to restore?).** Greedy decoding, `unsteered`, on the deranged
    corpus versus the **already-banked greedy L0** (`original_unsteered` = 0.8507), re-scored on the same
    surviving subset. Damage `D = acc(L0) − acc(deranged)`.
    **`DAMAGE-SUFFICIENT`** if `D ≥ +0.05` with the 95 % CI excluding 0 — the level at which a steering
    experiment is worth running, and not a new number: **+0.05 is the frozen steering-effect bar** used
    by H-R14b/c/d, so a deficit below it cannot host a detectable restoration ·
    **`DAMAGE-PRESENT-BUT-SMALL`** if `+0.01 ≤ D < +0.05` · **`DAMAGE-STILL-ABSENT`** if `D < +0.01`.
    *Prediction: **`DAMAGE-SUFFICIENT`***. Reasoning stated in advance so it can be wrong: the
    derangement attacks state-tracking (single-char counters) and the call site (the method name) at
    once, and it is the first stimulus in this thread that is adversarial by *construction* rather than
    by vocabulary.
  - **H-R22c (descriptive, no verdict).** `D` against the weak renamer's greedy damage (**+0.0069** on
    the same 148 snippets, same decoder) and against the paper's **+36.29** on Qwen2.5-7B; plus parse
    rate, and how much of `D` is single-char renaming versus method-name renaming is **not** decomposed
    here (that needs its own arms and is deferred, so no causal attribution is claimed).

  **Pre-committed consequences.** If **`DAMAGE-SUFFICIENT`**: the steering arms are worth re-running on
  this corpus and that gets its own pre-registration — and because the vectors are fitted on aligned
  L0/L1b span pairs, `ridge_map`/`erasure`/`swap_oracle` need a **new pool capture and a new cross-fit**
  on the deranged corpus; nothing from the old fit transfers. If **`DAMAGE-STILL-ABSENT`** or
  **`DAMAGE-PRESENT-BUT-SMALL`**: **renaming is abandoned as a stimulus for this model**, and the
  remaining options are control-flow flattening (L2/L3 — the generator is *not* in the artifact and would
  have to be written) or a different host. Either way **no steering arm runs on a corpus whose damage is
  below the effect bar.**

- **Setup:** new `nla/src/ase_rename_swap.py` (sha `37d3f13c5a96f6ac…`); `ase_rename_java.py` is **not
  modified** — it produced every banked H-R1/H-R7/H-R14/H-R18 number. Restricted to the frozen
  `full/snippets_148.json` so the comparison is snippet-for-snippet against the banked greedy L0. Packs
  rebuilt with the artifact's own execution-validating `build_case_pack` via `ase_casepacks.py`
  (unchanged; it already accepts an absolute `--lang-dir`), CPU only, JDK. Then **one** greedy arm
  (`--greedy --runs 1`, ~17 min GPU) — the L0 side is already banked, which is why the primary question
  costs one short job. Scored with `ase_r14_stats.py` (sha `bd2009919223743b…`). Seed 20260724 throughout;
  cluster bootstrap over snippets, `N_BOOT 10 000`.

  **Not touched:** the alignment/pool machinery. Damage needs only packs, so H-R22 is unaffected by
  whether the deranged corpus aligns token-for-token with L0 — but that alignment is a *prerequisite for
  the steering arms*, and a derangement renames far more identifiers than before, so the attrition from
  `prepare_residual` is an open question deferred to the stage-2 pre-registration.

- **Results / verdict / observations:** pending — this entry is the pre-registration.

- **Next Steps:** build the corpus → rebuild packs (CPU) → PACKS-PAIRED gate → one greedy arm → read
  H-R22a/b → either pre-register stage 2 or close renaming as a stimulus.
