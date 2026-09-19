### Target Date: 2026-09-19 (E4 — dispatcher state-binding probe)

- **Hypotheses / what we're testing:**

  E4 is one of the charter's four committed experiments (`CLAUDE.md` §3) and the only one that needs **no
  pretrained dictionary**, which is why it can run while the Instrument-3 env is being restored (T0.1).
  It also finally has a stimulus: H-R28 built a flattened corpus of **102 execution-validated snippets**
  whose `switch` case labels are **permuted**, so a case's printed position carries no information about
  when it runs and the execution order can only be had by following the `__state = N;` chain.

  **The question.** At the token where a case label appears, is that case's **execution rank** linearly
  decodable from the residual stream? If yes, the model binds each case to its position in the simulated
  sequence — the relational/state-tracking capacity the charter's flattening route is about. If no, it is
  reading the text without building the dispatcher's order.

  - **E4a (positive control).** `successor` — the next state, written **literally** in the case body
    (`__state = 7;`). **`PIPELINE-OK`** if decodable well above the majority baseline at any layer;
    **`PIPELINE-BROKEN`** otherwise, in which case **nothing else in this entry is read** — a null on the
    real target would be uninterpretable if the probe cannot recover information that is sitting in plain
    text at the probed position.
  - **E4b (positional control).** `printed_rank` — the case's ordinal position in the text. Expected to
    be easy (transformers encode position). **It is not evidence of state binding**, and it exists so
    that a high `exec_rank` score cannot be claimed as such without clearing it.
  - **E4c (the question).** `exec_rank`. **`STATE-BOUND`** if accuracy exceeds both the majority baseline
    and the shuffled control by ≥ 0.10 at some layer, **and** Spearman ρ > 0 ·
    **`STATE-NOT-BOUND`** if it sits within 0.05 of the shuffled control everywhere ·
    **`E4-INCONCLUSIVE`** in between.
    *Prediction: **`STATE-NOT-BOUND`** at the early layers and **`E4-INCONCLUSIVE`** overall.* Reasoning
    stated so it can be wrong: H-R28 found flattening costs these models almost nothing and that per-item
    agreement *rose* under it, which is what one would expect if the dispatcher is never simulated at
    read time. If E4 nonetheless returns `STATE-BOUND`, that is the more interesting outcome — it would
    mean the order **is** represented and the behavioural null is about something else.
  - **Shuffled control (mandatory).** Execution ranks permuted **within** each snippet. Must land at
    chance. Without it a probe that has memorised per-snippet idiosyncrasies reads as a finding.

- **Setup:** `ase_e4_state_probe.py` (sha `626b55903f0a3e73…`), the 102 flattened snippets and their
  PACKS-PAIRED packs, prompts built through the **same** `SteeredCausalLM` + chat template as every run
  in this block, so probed states are the states the behavioural arms actually saw. Residual stream at
  layers **4, 10, 16, 22, 28**; one forward pass per snippet, no generation. Probe = multinomial logistic
  regression on standardised activations, **GroupKFold(5) grouped by snippet** so no case from a program
  appears in both train and test. Baseline reported per fold is the **majority class of the training
  fold**, not 1/n — with 3–9 states per method the uniform figure would flatter the probe. Seed 20260724.
  Primary model `codellama/CodeLlama-7b-Instruct-hf`; the script is model-agnostic and the panel can be
  added without change. Cost ~0.3 GPU-h.

  **Ground truth is recovered from the generated source, not from the generator**, so a bug in the
  manifest cannot silently define the labels: printed order, entry state and the successor map are parsed
  back out of each variant and the chain must cover every case or the snippet is dropped.

  **Declared limit.** The dispatcher averages **4.1 states** over top-level statements only (H-R28's
  declared scope). A `STATE-NOT-BOUND` result therefore bounds *this* dispatcher depth; a deeper
  generator (H-R29) could change it, and that is the same limit the behavioural result carries.

- **Results / verdict / observations:** pending — this entry is the pre-registration.

- **Next Steps:** run on CodeLlama-7B; extend to the panel only if E4a passes.
