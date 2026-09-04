### Target Date: 2026-09-03 (T's alignment cannot be fixed in code — adversarial renaming is not length-preserving)

Follow-up to [`2026-09-03_site-alive-results.md`](2026-09-03_site-alive-results.md), which recorded
T refusing at its smoke gate (`all @ L0` recovered **0.784** against a frozen **0.80**) and listed
"fix `patch_map.py`'s alignment" as the remaining GPU item. **It is not a code defect.** No GPU used.

- **Hypotheses / what we're testing:** **H-T2** (from the results entry) — *"with 1:1 alignment on
  `replace` blocks, `all @ L0` reaches ≥ 0.80 and `L32|all` lands at ~1.00."* CONFIRM if a corrected
  alignment makes the sanity cell pass; REFUTE if the information required for a 1:1 map is not
  present in the corpus.

- **Setup:** CPU only, env `nla-mi`, no model loaded. `difflib.SequenceMatcher(autojunk=False)` over
  the **banked prompt token ids** (`l0_prompt_ids` / `l1b_prompt_ids` in
  `data/nla/p0/trace_llr/gemma12b/traces.jsonl`, 60 items — the exact ids R scored, not a
  re-tokenisation). Second pass over `rename_map` from `load_pairs(None, Random(20260724))` with the
  Gemma-3-12B-it tokenizer.

- **Results:**
  - **Block-level alignability, 60 items:** 1,289 `replace` blocks, of which **355 (27.5 %) are
    length-matched**. Of 3,628 replace tokens, **3,225 (88.9 %) sit in blocks whose L0 and L1b sides
    have different token counts**. Plus 141 insert tokens (L1b-only) and 91 delete tokens (L0-only).
  - **Items admitting a complete 1:1 map: 0 / 60.** Items where merely every `replace` block is
    length-matched: **0 / 60**. Median 19 replace blocks per item; median 43 mismatched tokens.
  - Unpatchable share of the L1b prompt: **median 19.4 %**, mean 19.5 %, max 38.8 %. Prompt length
    delta (L1b − L0): median **+27** tokens, range **[−1, +149]**.
  - **Cause, at the stimulus level:** across 328 rename pairs (median 5.5 per item), the decoy
    identifier matches the original's token length in only **60 / 328 (18.3 %)** of cases — 181 decoys
    are longer, 87 shorter. **180 of 328 originals are single tokens** (`f`, `ls`, `name`), replaced by
    multi-token descriptive decoys (`_get_byte_parser`, `custom_type`, `child_iterator`).
  - **The machinery is correct — the smoke output proves it.** `L0|code` = **−0.003** and
    `L0|instr` = **0.000** are exactly right: `equal` blocks are *identical token ids* by definition,
    so patching them at layer 0 is a no-op, and the harness returns a no-op. A patching bug would not
    produce a clean zero here.
  - **`L32|all` = 1.177 (over-recovery) has a separate cause** and is not an alignment artifact: by
    layer 32 the clean run's states already encode the answer, so patching them transplants the
    *answer* rather than restoring the *computation*, overshooting a reference that still contains the
    corrupt run's early layers. Late-layer "recovery" above 1.0 is the signature of that, not of a
    better circuit.

- **What worked / hypothesis verdict:**
  - **H-T2 ✗ REFUTED — and the refutation is structural, not fixable.** A 1:1 token map does not exist
    for **any** of the 60 items, so the pre-registered sanity cell `all @ L0` **cannot reach 1.0 on
    this corpus at any effort**. The 0.784 the smoke returned is close to the arithmetic ceiling of
    mean-broadcasting ~19 % of the prompt, not a threshold that better code clears.
  - **T as pre-registered is not executable here.** Recording that plainly instead of tuning the gate
    down to 0.75 — which would have "passed" a map that cannot represent 19 % of the positions.
  - The banked steering results are unaffected: they never required cross-prompt alignment.

- **Observations:**
  - This is a **stimulus-design constraint that token-level activation patching imposes and this
    corpus was never built to satisfy** — Papers 2–3 needed L0/L1b pairs that are *semantically*
    matched and execution-equivalent, and nobody had reason to also constrain tokenisation. It is
    worth stating for the write-up: adversarial renaming inflates prompts by a median of 27 tokens,
    so **any** L0-vs-L1b comparison that is sensitive to sequence length (attention-mass fractions in
    Instrument 1 included) carries a token-count confound that has not been controlled.
  - The `code` and `instr` cells at layer 0 are **vacuous by construction**, not informative nulls.
    Anything downstream must not read them as evidence that non-identifier positions carry no
    corruption; at L0 they *cannot*.

- **New questions / new hypotheses:**
  - **H-T3:** a **length-matched L1b tier** — decoys chosen so `len(tokenize(decoy)) ==
    len(tokenize(orig))` — makes `all @ L0` exactly 1.0 by construction and makes T executable.
    Feasibility is measured, not assumed: 180 of 328 originals are single tokens, so ~55 % of the
    renames need a *single-token* misleading identifier; the remaining 148 need 2–5 tokens, which is
    roomier. No GPU to build; requires re-running baselines on the new tier and re-validating
    functional equivalence.
  - **H-T4:** with such a tier, does the L1b−L0 accuracy penalty survive at matched token count? If
    the penalty shrinks, part of the documented L1b effect is length/dilution rather than semantic
    displacement — which would matter to Papers 2–3, not just to this instrument.
  - **H-T5 (cheap, no new stimuli):** restrict the `id` cell to the **355 length-matched replace
    blocks** (403 tokens, 11 % of identifier tokens) where patching *is* exact, and ask whether those
    positions alone move the readout. A positive there is interpretable; a null is underpowered and
    must be labelled so.

- **Next Steps:**
  1. **Do not re-submit T against the current corpus.** The gate stays at 0.80; the corpus fails it.
  2. Decision required before any build: a length-matched L1b tier is a **corpus fork** — new
     stimuli, new baselines, and results not directly comparable to the banked 60. Worth it only if
     T (or H-T4) is wanted; not worth it to rescue a steering question that R2 has already closed.
  3. H-T5 is the only version of T runnable on the banked corpus and it is a bounded sub-question,
     so it should be pre-registered as its own experiment rather than filed under T.
