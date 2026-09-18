### Target Date: 2026-09-18 (H-R28 — control-flow flattening: the other obfuscation route)

- **Hypotheses / what we're testing:**

  H-R22/H-R23 closed the **renaming** route: a vocabulary-preserving derangement of every identifier
  costs CodeLlama-7B **+0.0122**, no permitted model shows a deficit (`NO-PERMITTED-HOST`), and per-item
  agreement of 0.76–0.93 says these models **trace code rather than read names**. Flattening attacks the
  **other** route in the charter's ladder — **relational overload**: a dispatcher loop severs the
  correspondence between textual order and execution order, so a reader must simulate a state variable
  instead of reading top-to-bottom. **Nothing measured in this block speaks to it**, and the ASE
  artifact ships no generator, so one was built (`ase_flatten_java.py`).

  - **H-R28a (corpus, gate).** `CORPUS-OK` if **≥ 80** snippets survive flattening + `javac` +
    execution-validated pack rebuild + PACKS-PAIRED; `CORPUS-TOO-SMALL` otherwise.
    **Disclosure, because this bar is being set later than the others:** the generator has already been
    run and yields **103 candidates** (102 compiling) before execution validation, and I know that. The
    bar is therefore justified on **power**, not convenience: the greedy damage CI half-width at n = 146
    was 0.0254, which scales to ≈ 0.034 at n = 80 — still comfortably able to resolve the +0.05 effect
    the verdict turns on. It is **not** set at the 103 I happen to have.
  - **H-R28b (primary — does flattening damage what renaming did not?).** Damage
    `D = acc(L0) − acc(flattened)` under greedy, per model, on the surviving subset, with the **L0 side
    re-scored from the already-banked panel runs** on that same subset. **Bands carried over unchanged**
    from H-R22/H-R23: `DAMAGE-SUFFICIENT` ≥ +0.05 with CI excluding 0 · `DAMAGE-PRESENT-BUT-SMALL`
    +0.01 … +0.05 · `DAMAGE-ABSENT` < +0.01. Same **floor gate** (L0 < 0.60 ⇒ recorded, not read).
  - **H-R28c (the decision).** `FLATTENING-HOST-FOUND` if ≥ 1 readable model returns
    `DAMAGE-SUFFICIENT` — that model hosts the steering arms and the programme restarts there ·
    `NO-FLATTENING-DEFICIT` otherwise, which closes the *second* route and means the ASE line is finished
    on both.
    *Prediction: **`FLATTENING-HOST-FOUND`**, and this is the first time in this block I expect a
    positive.* Reasoning, stated so it can be wrong: renaming leaves the computation intact and only
    relabels it, which is why models that trace code were immune; flattening changes **what has to be
    traced**, turning straight-line reading into state simulation, and the charter's anchor finding —
    dispatcher complexity r = −0.196 under flattening — is about exactly this.

- **Setup:** the same 146 PACKS-PAIRED snippets as the renaming work (so all three conditions share a
  corpus), greedy (`--greedy --runs 1`, deterministic), `unsteered` only — no steering, no vectors.
  Packs rebuilt on the flattened variants with the artifact's own execution-validating `build_case_pack`,
  then PACKS-PAIRED against the originals. Four models, one arm each (**the L0 side is already banked**),
  now running up to **8 concurrent GPU jobs** per the raised allowance. `ase_flatten_java.py` sha
  recorded in the results entry; `ase_steer_run.py` `d271188e29dcf1cd…` (H-R27 reply-excerpt banking
  added, so this stimulus's parse failures are diagnosable after the fact — the gap H-R27 named).

  **The transform, and its deliberate limits.** The target method's top-level statement sequence becomes
  a `while(true) switch(__state)` dispatcher with **permuted case labels**, so printed order ≠ execution
  order (identity permutations are rejected and redrawn — 0 remain). Local declarations are **hoisted**
  above the loop with type-appropriate defaults, because Java's definite-assignment analysis rejects a
  declaration in one case and a use in another. Compound statements (`if`/`for`/`while`/`try`) move
  **whole** into their case rather than being decomposed: decomposing nested control flow needs a real
  CFG with back edges, and a subtly wrong CFG would silently change semantics — the one failure this
  corpus cannot catch cheaply, since a variant that compiles and runs but computes something else would
  be scored as a comprehension result. **This makes the stimulus weaker than a production obfuscator and
  that is a declared limitation**: a null here bounds *this* flattening, not flattening in general.

  Skips are recorded per reason, never guessed at: 29 too-few-statements, 7 last-statement-not-terminal,
  5 lambda-capture (hoisting breaks effective-finality), 2 unparseable.

- **Results / verdict / observations:** pending — this entry is the pre-registration.

- **Next Steps:** pack rebuild + PACKS-PAIRED → read H-R28a → 4 greedy arms → H-R28b/c.
