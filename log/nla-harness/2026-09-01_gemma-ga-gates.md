### Target Date: 2026-09-01 (Gemma port — both G-A gates pass)

Phase G-A from [`../../docs/CHECKLIST.md`](../../docs/CHECKLIST.md), the step the whole port was
declared conditional on. Context: the model constraint (no Chinese-origin models) removed
`Qwen2.5-7B-Instruct`, and Gemma-3-12B-IT is the recommended successor because it is the only host
with **both** a released NLA pair and a dictionary suite.

- **Hypotheses / what we tested:**
  - **G-A1.** Does the extractor read the block the steerer writes, on a host where `_layers()`
    had to learn a new path? Gemma-3 loads as `Gemma3ForConditionalGeneration`, so decoder blocks
    sit at `model.language_model.layers` and **both** `extract.py` and `steer.py` were edited. A
    silent drift between them would mislabel every depth result.
  - **G-A2 (G0 stage A).** Do our per-token `||v||` reproduce the shipped worked example? Pure
    forward passes, no NLA — so a failure isolates layer indexing and chat templating from
    anything about the autoencoder.

- **Setup:**
  ```
  jobs    360879 then 361192 (clean re-run) · h200 g-07-02 · 1m21s / 1m27s
  host    google/gemma-3-12b-it (23 GB) · AV 22 GB · AR 17 GB — 62 GB fetched 2026-08-31
  ref     nla/vendor/nla-repo/examples/gemma12b_layer32_step4000.txt
  code    nla/src/p04_gate.py --host gemma12b · nla/src/replicate_example.py --stage A --model gemma12b
  ```
  Both scripts were parameterised for this: the gate takes `--host`, and the G0 script now parses
  its constants **out of** the reference file rather than from hardcoded Qwen values.

- **Results:**

  **G-A1 — layer indexing: PASS.**

  | layer | fp32 residual | positions written | other positions moved |
  |---|---|---|---|
  | 8 | 1.33e-07 | 1 | 0.0 |
  | 20 | 1.15e-08 | 1 | 0.0 |
  | **32** | **9.39e-08** | 1 | 0.0 |

  bf16 sits at 3.53–3.82e-03, the same 8-bit-mantissa floor Qwen showed, which is why fp32 is the
  gate.

  **G-A2 — G0 stage A: PASS.** `tokens: ours=164 reference=164` · **median rel err 0.00005** ·
  max 0.01443 · **0/164 over the 0.02 tolerance**. `<bos>` reproduces at 712,708.5 exactly.

- **What worked / hypothesis verdict:**
  - **Both gates ✓.** The layer index is right, the chat template is right, and extraction
    reproduces the reference on a host with a different architecture, a different tokenizer and
    activation norms ~1000× larger than Qwen's (80k–700k against ~235).
  - **The riskiest item in the port plan is settled for stage A.** Stage B — the full AV→AR round
    trip, which needs the server up — remains the open half.

- **Observations:**
  - **Deriving the end-of-turn token from the reference's last row was load-bearing.** Qwen ends on
    `<|im_end|>` and Llama-70B on `<|eot_id|>`, but **both Gemma references stop mid-sentence with
    no end-of-turn at all**. Since `extract_chat` appends the reply verbatim, assuming
    `<end_of_turn>` by analogy would have produced 165 tokens against 164 — a length mismatch that
    reads as a broken chat template rather than as a wrong constant.
  - **Parsing the reference beat transcribing it.** The Gemma reply is 639 characters of escaped
    text; hand-copying it is precisely the silent-error class this gate exists to catch. Two more
    footguns surfaced only because all four files are now parsed uniformly: the reply literal is
    single-quoted in two files and double-quoted in the other two, and a `[\d.]+` capture for
    `Var(v_nrm)` swallowed the sentence's closing full stop.
  - **A cosmetic failure on the first run is worth recording.** Job 360879 printed both verdicts
    correctly and then died with `NameError: TARGET_MODEL` in the manifest-writing code — a stale
    reference my own parameterisation left behind. The science was complete before the crash; the
    fix also moved the manifest to a **per-host filename** so a Gemma run cannot overwrite the
    banked Qwen result.

- **New questions / new hypotheses:**
  - **Stage B is where the low-variance issue will actually show.** With Var(v_nrm) = 0.0302,
    expect `cos` near 0.99 across the board and `fve_nrm` carrying the signal. The reference's own
    rows confirm `mse_nrm = 2(1 − cos)` and `fve_nrm = 1 − mse_nrm/Var`, so no re-capture is needed
    — but the **reported** metric must switch, and the `rt_cos` validated band (0.70–0.96) does not
    transfer to this host.
  - G-B can start on the strength of stage A alone: the behavioural baseline is generations, not
    reads, and does not depend on the AV.

- **Next Steps:** G0 stage B once the AV server is up; G-B behavioural baseline (5 tiers × 60 items
  × 10 draws) in parallel, since it needs no NLA.
