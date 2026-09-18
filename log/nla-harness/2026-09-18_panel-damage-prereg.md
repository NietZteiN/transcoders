### Target Date: 2026-09-18 (H-R23 — does ANY permitted model show a renaming deficit?)

- **Hypotheses / what we're testing:**

  H-R22 closed renaming as a stimulus **for CodeLlama-7B-Instruct**: deranging all 837 identifiers
  (365 single-char, method name included, vocabulary exactly preserved) costs **+0.0122 [−0.013, +0.038]**,
  and the model answers identically on **92.9 %** of cases. That is one model. This entry asks whether the
  paper's premise — that obfuscation removes lexical cues the model depends on — holds for **any model we
  are permitted to run**. It is the precondition for everything downstream: without a host that shows a
  deficit, no steering method can be evaluated, and building a control-flow-flattening generator (the
  other route, absent from the artifact) would be spending a large build on an untested premise.

  - **H-R23a (per model).** Damage `D = acc(L0) − acc(deranged)` under greedy decoding on the 146
    PACKS-PAIRED snippets. **Thresholds are carried over unchanged from H-R22, not re-chosen:**
    **`DAMAGE-SUFFICIENT`** if `D ≥ +0.05` with the 95 % CI excluding 0 ·
    **`DAMAGE-PRESENT-BUT-SMALL`** if `+0.01 ≤ D < +0.05` · **`DAMAGE-ABSENT`** if `D < +0.01`.
  - **H-R23b (panel, the decision).** **`HOST-FOUND`** if ≥ 1 model returns `DAMAGE-SUFFICIENT` — that
    model becomes the host and the steering arms move there · **`NO-PERMITTED-HOST`** otherwise, in which
    case the ASE replication is finished as far as we can take it and the honest write-up is *"their
    effect does not reproduce on any model we are allowed to run"*.
    *Prediction: **`NO-PERMITTED-HOST`***, held with moderate confidence. The model to watch is
    **codegemma-7b-it**: H-R1 measured the largest permitted damage on it (**+0.076**) — but with the weak
    renamer, sampled decoding and a different harness, all three of which this block has since shown to
    move numbers by more than the effect.
  - **Floor gate (pre-registered, so a null cannot be manufactured by a broken model).** A model whose
    **L0 accuracy < 0.60** (chance = 0.500) has its damage recorded but **not read into H-R23a/b** — a
    model that cannot do the task ungobfuscated cannot show a loss from obfuscation.
  - **Exclusion, with precedent.** A model that fails the chat-template ids gate or their runtime is
    recorded **`PROTOCOL-BROKEN`** and excluded, never worked around — exactly as StarCoder2 was in H-R1.
    `chat_wrap` raises rather than silently mis-tokenising, which is why this is safe to attempt.
  - **Descriptive, no verdict:** per-model **per-item agreement** between L0 and the derangement (the
    H-R22 measure, exact because greedy is deterministic). This is the direct read on *does this model use
    identifier names at all*, and it is more informative than the damage point estimate.

- **Setup:** the frozen `full/snippets_swap.json` (146 snippets / 1 722 cases), the same original and
  deranged packs, greedy (`--greedy --runs 1`), their `SteeredCausalLM` with the chat template installed.
  Arm = `unsteered` in both conditions, so **no steering, no vectors, no alignment** is involved — this
  measures the stimulus only. Two arms per model.

  **Panel (every permitted instruct model in the offline cache; no Chinese models, per the standing rule):**

  | model | why |
  |---|---|
  | `google/codegemma-7b-it` | code-instruct; **largest permitted damage in H-R1 (+0.076)** — the one to watch |
  | `meta-llama/Llama-3.1-8B-Instruct` | general instruct at 8B; H-R1 measured +0.030 on the base model |
  | `microsoft/Phi-3.5-mini-instruct` | the charter's panel; smallest, tests whether capacity matters |
  | `codellama/CodeLlama-13b-Instruct-hf` | **same family at ~2× scale** — tests whether H-R22's null is a scale artefact rather than a family one |

  *Not run:* SmolLM3-3B and Qwen/DeepSeek-family models are unavailable — the former is not in the offline
  cache and `HF_HUB_OFFLINE=1`, the latter are barred. StarCoder2-15B is cached but was `PROTOCOL-BROKEN`
  in H-R1 and is not retried here.

  A **2-snippet smoke per model first**, because their runtime is Llama-derived and Gemma/Phi BOS
  conventions differ; models that fail it are excluded as `PROTOCOL-BROKEN` before any real GPU is spent.
  ~2.3 GPU-h for 8 arms (13B ≈ 30 min, the rest ≈ 15 min), three singleton lanes, ≤ 3 concurrent jobs.

- **Results / verdict / observations:** pending — this entry is the pre-registration.

- **Next Steps:** smoke → surviving arms → per-model damage + agreement → H-R23a/b, and either name the
  host or close the ASE replication.
