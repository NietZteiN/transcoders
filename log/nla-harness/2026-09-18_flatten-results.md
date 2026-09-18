### Target Date: 2026-09-18 (H-R28 RESULTS — flattening does not bite either, and both obfuscation routes are now closed)

- **Hypotheses / what we're testing:** as pre-registered in
  [`2026-09-18_flatten-prereg.md`](2026-09-18_flatten-prereg.md). Predictions: **a** `CORPUS-OK`;
  **c `FLATTENING-HOST-FOUND`** — the first positive I expected in this block.

- **Setup:** `ase_flatten_java.py` (sha `ebcf2815066fc9e2…`), the 146
  PACKS-PAIRED snippets as parent set, greedy, `unsteered` only. Packs rebuilt with the artifact's own
  execution-validating `build_case_pack` (job 411560). Five models, one arm each (L0 already banked),
  jobs 411566–411570 on g-07-01/03/10, all rc=0, 102/102 rows, **3–9 min per arm**.

- **Results:**

  **H-R28a `CORPUS-OK`** — 103 of 146 flattened (skips itemised: 29 too-few-statements, 7
  last-statement-not-terminal, 5 lambda-capture, 2 unparseable), **0 compile or execution failures**
  across all 103 under their builder, **102** survive PACKS-PAIRED. 1 188 cases, 11.55 per snippet
  against their 11.72. Mean 4.1 dispatcher states per method, 176 declarations hoisted, **0 identity
  permutations** (printed order never equals execution order).

  **H-R28b / c `NO-FLATTENING-DEFICIT` — prediction REFUTED.**

  | model | L0 | flattened | damage | 95 % CI | parse | agreement | verdict |
  |---|---|---|---|---|---|---|---|
  | `codegemma-7b-it` | 0.8392 | 0.8519 | **−0.0126** | [−0.0347, +0.0077] | 0.997 | 0.9562 | `DAMAGE-ABSENT` |
  | `Llama-3.1-8B-Instruct` | 0.8350 | 0.8106 | +0.0244 | [−0.0008, +0.0563] | 0.993 | 0.9453 | small |
  | `Phi-3.5-mini-instruct` | 0.5581 | 0.6221 | −0.0640 | [−0.1362, +0.0105] | 0.818 | 0.7399 | **floor — not read** |
  | `CodeLlama-13b-Instruct` | 0.7946 | 0.7593 | +0.0354 | [−0.0194, +0.0936] | 0.906 | 0.8510 | small |
  | `CodeLlama-7b-Instruct` | 0.8737 | 0.8527 | +0.0210 | [−0.0120, +0.0551] | 1.000 | 0.9453 | small |

  Read on 4 of 5; no readable model reaches +0.05 and every CI contains zero.

  **The two routes, on the SAME 102 programs** (the comparison that matters, and the one I nearly got
  wrong):

  | model | renaming D | flattening D | flattening − renaming |
  |---|---|---|---|
  | `codegemma-7b-it` | +0.0067 | −0.0126 | −0.0194 [−0.0544, +0.0171] |
  | `Llama-3.1-8B-Instruct` | +0.0244 | +0.0244 | **−0.0000** [−0.0475, +0.0488] |
  | `CodeLlama-13b-Instruct` | −0.0067 | +0.0354 | +0.0421 [−0.0318, +0.1189] |
  | `CodeLlama-7b-Instruct` | +0.0295 | +0.0210 | −0.0084 [−0.0441, +0.0277] |

- **What worked / hypothesis verdict:**
  - **H-R28a SUPPORTED.** The generator works and is validated by execution, which is the claim that
    mattered: a dispatcher rewrite has many ways to change behaviour silently.
  - **H-R28c REFUTED my prediction.** I argued flattening would bite because it changes *what has to be
    traced* rather than merely relabelling it. The largest readable damage is **+0.0354** with a CI from
    −0.019 to +0.094; `codegemma-7b-it` is **negative** — the flattened code scores *higher*.
  - **A comparison I nearly reported wrongly.** Across the raw tables, flattening damage (+0.021…+0.035)
    looks about **2×** renaming damage (+0.007…+0.017), and I was about to say so. On the **matched 102
    snippets** the difference is **−0.019, −0.000, +0.042, −0.008 — every CI spanning zero.** The
    apparent doubling was an artefact of the two stimuli having different surviving subsets. **On the same
    programs the two obfuscation routes are indistinguishable.**
  - **Both routes are now closed on every permitted model**, which is a materially stronger statement
    than either alone: it is not that these models are insensitive to *identifier* semantics
    specifically, but that neither atom-level interference nor this dispatcher indirection moves their
    output prediction.

- **Observations:**
  - **Per-item agreement is 0.85–0.96 on readable models — HIGHER than under the derangement (0.76–0.93).**
    Rewriting the control flow into a permuted dispatcher changes these models' answers *less* than
    renaming the identifiers did. Whatever they are doing, it survives having the textual order of the
    program destroyed.
  - **The limitation declared in the pre-registration is load-bearing and I will not soften it.**
    Compound statements move whole into their case rather than being decomposed, so the dispatcher
    averages **4.1 states** over top-level statements only — far weaker than a production obfuscator,
    which would flatten nested control flow too and produce dozens of states. **This result bounds *this*
    flattening, not flattening in general**, and the honest summary is "a top-level dispatcher rewrite
    with permuted labels does not damage these models", not "control-flow obfuscation does not work".
  - `Phi-3.5-mini` again fails the floor gate (L0 0.5581) and again shows a large negative — flattened
    code scoring *higher* — which is what a near-chance model looks like when it moves at all. Consistent
    with H-R25/H-R26; not read, as pre-registered.
  - **Scheduling, recorded because it cost wall-clock and was self-inflicted:** the arms sat `PENDING
    (Priority)` with a 12-hour estimate. Measured peak RSS is **15.4 GB (7B) / 26.7 GB (13B)** while every
    sbatch asked for **200 GB**, and the busy h200 nodes had 1.5–2 GB free — the request fit nowhere.
    Dropping to **48 G** and the wall from 4 h to 40 min (arms take 3–9 min) moved the estimate ~9 hours
    earlier and they ran immediately. **All 45 sbatch scripts were corrected** with the measured figures
    in a comment.

- **New questions / new hypotheses:**
  - **H-R29 (the only remaining way to rescue the flattening route):** decompose nested control flow into
    its own dispatcher states (a real CFG with back edges), taking the mean state count from ~4 to
    ~20–40. That is the difference between this generator and a production obfuscator, and it is the only
    version of the flattening claim this block has *not* tested. It is also a substantially harder build
    whose failure mode is a silent semantics change, so it needs a differential-testing harness
    (run original and variant on the pack inputs, compare outputs) before any model sees it.
  - **H-R24 is now decidable on evidence rather than intuition.** Both routes closed, five models, two
    stimuli, a deterministic decoder and an oracle ceiling of +0.0052. The ASE line's honest write-up is a
    **bounded negative replication**; the alternative is H-R29's deeper generator. **Still a scope call —
    but the prior has moved substantially against spending more here.**

- **Next Steps:** H-R24/H-R29 is a scope decision for the human. No further ASE GPU work without it.
