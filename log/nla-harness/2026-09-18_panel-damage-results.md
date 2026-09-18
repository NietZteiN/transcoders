### Target Date: 2026-09-18 (H-R23 RESULTS — no permitted model shows a renaming deficit, and the floor gate earned its keep)

- **Hypotheses / what we're testing:** as pre-registered in
  [`2026-09-18_panel-damage-prereg.md`](2026-09-18_panel-damage-prereg.md). Prediction on record:
  **`NO-PERMITTED-HOST`**, with `codegemma-7b-it` flagged as the model to watch (H-R1 measured +0.076 on it).

- **Setup:** 146 PACKS-PAIRED snippets / 1 722 cases, greedy (`--greedy --runs 1`, deterministic),
  `unsteered` in both conditions — **no steering, no vectors, no alignment**: the stimulus only. Feasibility
  smoke 410861 first; arms 410868–410875, nodes g-07-12 / g-05-01, all rc=0, 146/146 rows each, 4–11 min
  per arm, ≤ 3 concurrent GPU jobs. `ase_steer_run.py` sha `a73a3fd93ad8b4c5…`.

  **All four models passed the chat-template ids gate with their own native format** — Gemma
  `<bos><start_of_turn>`, Llama-3.1 `<|begin_of_text|><|start_header_id|>`, Phi `<|user|>`, CodeLlama-13b
  `<s>[INST]`. `chat_wrap` turned out to be genuinely generic rather than Llama-specific, and it verifies
  `tokenizer(text) == apply_chat_template(tokenize=True)` per prompt, so no `PROTOCOL-BROKEN` exclusions
  were needed.

- **Results:**

  | model | L0 | deranged | damage | 95 % CI | parse (L0) | per-item agreement | verdict |
  |---|---|---|---|---|---|---|---|
  | `codegemma-7b-it` | 0.8124 | 0.8055 | +0.0070 | [−0.0226, +0.0365] | 0.999 | 0.8966 | **`DAMAGE-ABSENT`** |
  | `Llama-3.1-8B-Instruct` | 0.8200 | 0.8031 | +0.0168 | [−0.0170, +0.0517] | 0.985 | 0.8751 | `DAMAGE-PRESENT-BUT-SMALL` |
  | `Phi-3.5-mini-instruct` | **0.5639** | 0.5093 | +0.0546 | [+0.0076, +0.1059] | 0.768 | 0.7578 | **`FLOOR-TOO-LOW` — not read** |
  | `CodeLlama-13b-Instruct` | 0.7631 | 0.7875 | **−0.0244** | [−0.0827, +0.0317] | 0.948 | 0.8031 | **`DAMAGE-ABSENT`** |
  | *(reference)* `CodeLlama-7b-Instruct` | 0.8531 | 0.8409 | +0.0122 | [−0.0125, +0.0382] | 1.000 | 0.9286 | `DAMAGE-PRESENT-BUT-SMALL` |
  | *(reference)* their Qwen2.5-7B | 76.49 | 40.20 | **+0.3629** | — | — | — | — |

  **H-R23b `NO-PERMITTED-HOST`**, read on 3 of 4 models (the floor gate excluded one).

- **What worked / hypothesis verdict:**
  - **H-R23b SUPPORTED as predicted.** Across four models spanning **3.8B to 13B**, two code-tuned and two
    general, **no readable model shows a renaming deficit**. The largest readable damage is
    Llama-3.1-8B-Instruct's **+0.0168** with a CI containing zero; `codegemma-7b-it` is **+0.0070**; and
    `CodeLlama-13b-Instruct` is **−0.0244** — the deranged code scores *higher*. Against the paper's
    **+0.3629** the whole panel sits between −2.4 and +1.7 points.
  - **The floor gate was the most valuable line in the pre-registration.** `Phi-3.5-mini-instruct`
    returned damage **+0.0546 with a CI excluding zero** — formally the *only* `DAMAGE-SUFFICIENT` cell in
    the entire block, and it would have been reported as `HOST-FOUND` and sent the steering programme to
    that model. But its **L0 accuracy is 0.5639 against a chance floor of 0.500**, and it fails to emit a
    parseable answer on **23 %** of cases. It cannot do output prediction on unobfuscated code, so it
    cannot lose accuracy to obfuscation; its "damage" is the model degrading from near-chance to chance.
    **That gate was written before any of these numbers existed**, precisely so this could not be
    harvested as a positive result.
  - **`codegemma-7b-it`'s H-R1 damage does not survive: +0.076 → +0.0070**, a ~10× reduction. H-R1 used
    the weak renamer, sampled decoding and a different harness — and this block has separately shown all
    three move numbers by more than the effect (H-R15 sampling ±0.02–0.04; H-R18 decoder ~16 points).
    The model flagged as most likely to be a host is in fact the *least* affected readable model.
  - **The null is not a scale artefact.** CodeLlama at 13B shows *negative* damage, and its L0 accuracy
    (0.7631) is **lower** than the 7B's (0.8531) — doubling parameters within the same family neither
    creates a deficit nor improves the task.

- **Observations:**
  - **Per-item agreement is 0.76–0.90 across the panel** (CodeLlama-7B 0.9286). Every model gives mostly
    the same answers after every identifier in the program has been rewired. The H-R22 conclusion — these
    models trace code rather than read names — is a **panel-level** property, not a quirk of one checkpoint.
  - **This closes the ASE replication at the honest boundary.** Combining the block: their protocol
    reproduces (case packs 11.73 vs 11.72 per snippet), their steering is inert-to-harmful on the model we
    could test, the oracle ceiling is +0.0052, the strongest possible renaming costs ~1 point, and **no
    permitted model shows the deficit their method exists to repair**. The defensible claim is *"their
    effect does not reproduce on any model we are allowed to run, and on these models the mechanism it is
    attributed to — reliance on lexical identifier cues — is absent."* **Their result is not
    contradicted**: their four models are all barred, their generator is not in the artifact, and only one
    of their four obfuscation families was tested.
  - **What is still untested, and is now the only route left:** control-flow flattening (L2/L3) attacks
    **relational overload / hidden-state simulation**, a different failure route from atom-level
    interference, and nothing here speaks to it. The generator is not in the artifact and would have to be
    written and equivalence-checked.
  - Reusing the frozen corpus and the banked L0 made the whole panel cost **~45 min of GPU**, not the
    2.3 h budgeted — the arms ran 4–11 min each.

- **New questions / new hypotheses:**
  - **H-R24 (the decision, not an experiment):** the renaming route is exhausted across every model
    available to us. Either invest in an **L2/L3 flattening generator** (a real build: generator +
    semantic-equivalence checker + pack rebuild, and it tests the route the charter's E3 actually cares
    about), or **write up the ASE line as a bounded negative replication** and return the GPU budget to
    Instrument 3's own experiments (E1/E2/E3 on the SAE side). **This is a scope call for the human**, and
    it should not be made by defaulting into the larger build.
  - **H-R25 (cheap, falls out of this):** `Phi-3.5-mini-instruct` sits at 0.5639 with parse 0.768 on
    output prediction that three other models do at 0.76–0.85. Is that a capability floor or a formatting
    failure? Its acc-given-parsed would separate them on banked rows, no GPU — and it matters because the
    charter's panel includes Phi-3.5-mini for the main study.
  - **H-R20 unchanged** and now the largest unexplained effect in the block: the ~16-point greedy gain.

- **Next Steps:** H-R24 is a scope decision for the human. No further ASE GPU work without it.
