### Target Date: 2026-09-03 (Experiment W — the NLA's own read→edit→reconstruct→write-back protocol. Frozen before running.)

Raised in session after [`2026-09-03_nla-and-coverage-results.md`](2026-09-03_nla-and-coverage-results.md)
and [`2026-09-03_clean-half-results.md`](2026-09-03_clean-half-results.md). **Committed before any
run.** Host `gemma12b` (google/gemma-3-12b-it, L32/48) — the only permitted host with a released NLA
pair. Standing constraint: no Chinese models.

#### Why this family exists: the NLA's protocol has never been executed here

Every steering arm to date fed the AR a **template string**, not a verbalized read:

| banked arm | text handed to the AR | what it therefore is |
|---|---|---|
| `V1_gloss` | `"code whose identifiers are about <ids>"`, true vs decoy | a bag-of-identifier lexical contrast |
| `V2_wordedit` | the same template, one term swapped | the same, smaller |

`steer_run.py:480` says so explicitly ("An AV-read-and-edit variant would additionally measure the
verbalizer's paraphrase, which is a different question"). So **`V1 ≡ V3` exactly (CI [0,0]) is close
to guaranteed by construction** — AR(template with true ids) − AR(template with decoy ids) *is* a mean
lexical-contrast direction, which is what V3 is. That result bounds the *template* method; it does not
test the NLA paper's protocol, which is **read → minimally edit the read → reconstruct → write back**.

Three properties of the released pair, none of which the banked design was built around
(`nla_meta.yaml`, both checkpoints, `extraction_layer_index: 32`, `mse_scale 61.9677`,
AV `injection_scale 80000`):

1. **The AR emits a whole state at one position** (`text → residual at the final token`), not a delta.
   The native operation is therefore **replace**, not add. Adding AR *differences* was our invention.
2. **The AR is direction-only** — trained on `MSE = 2(1−cos)`, so its output norm carries no
   information. Every α we swept was a hand-set magnitude on a direction-only object. Norm-matched
   replacement removes α from the design entirely.
3. **Position is the untried lever.** `V5_replace` — the exact clean state, all 33 layers, at
   `last_prompt` — moves the clean half by **+0.0014 of G** (CI [−0.0003, +0.0012]): nothing at that
   position matters. The only arms with a CI excluding zero on the positive side were at **identifier
   spans** (`V1_gloss|0.149|id_spans` **+0.048**, `V3|0.149|id_spans` +0.023, `R_random|0.149|id_spans`
   +0.056). Stage 1 writes at spans, not at `last_prompt`.

- **Hypotheses / what we're testing.**

  - **H-W0a (representability).** The NLA round trip preserves item identity at L1b decoy-identifier
    span positions. CONFIRM if matched `cos(AR(AV(h_i)), h_i)` exceeds **cross-item** mismatched
    `cos(AR(AV(h_i)), h_j)`, `i ≠ j`, with **non-overlapping 95 % bootstrap CIs**. REFUTE → **`NLA-VOID`**:
    the instrument cannot represent these positions, steering through it is impossible in principle,
    the family stops here and is written up as a bounded negative about the instrument.
  - **H-W0b (span specificity).** The round trip distinguishes *this* identifier from *another
    identifier in the same snippet*. CONFIRM if matched cos exceeds **within-item other-span**
    mismatched cos, non-overlapping CIs. REFUTE → **`NLA-COARSE`**: the NLA carries "this is obfuscated
    code" but not "this identifier", so a per-span edit has no referent; **stage 1 does not run** and
    the family routes to stage 2 alone.
  - **H-W1 (the protocol steers).** The edited write-back recovers clean-trace likelihood. CONFIRM if
    ΔG_sum ≥ **+12.11 nats** (= 10 % of mean G_sum = 121.127, the R2 support convention carried over)
    with a bootstrap CI excluding 0, **and** it clears **both** controls (below), **and** it survives
    the behavioural veto. REFUTE → **`W-NULL`**.
  - **H-W2 (the write registers).** Re-verbalizing the written position shows the read moved toward the
    edit. CONFIRM if `cos(AR(AV(h_after)), AR(text_edited)) > cos(AR(AV(h_after)), AR(text_read))` on
    **> 50 %** of written spans with a CI excluding 0.50. REFUTE → the channel did not deliver.
  - **H-W3 (descriptive, no gate — E1 by another instrument).** In AV reads of L1b decoy spans, the
    **decoy** semantic field is mentioned more often than the **true** one. Reported as two rates with
    CIs; **pre-registered as descriptive**, and it is exactly the E1 semantic-capture question asked
    through the NLA rather than through an SAE.

- **Setup (frozen).**
  - **Stimuli:** the banked 60 L0/L1b pairs from `load_pairs(None, Random(20260724))` — same corpus,
    same order, so results sit beside R2/L/L3 without re-baselining.
  - **Sites:** identifier-span token positions from `nla/src/span_positions.py` (offset-mapping
    verified against the rendered prompt; the same mapping the `id_spans` coverage arms used).
  - **AV:** `kitft/nla-gemma3-12b-L32-av`, sglang, `injection_scale 80000`, injection char `㈜`
    (token 246566). **AR:** `kitft/nla-gemma3-12b-L32-ar`, `mse_scale 61.9677`, BOS **on** (Gemma
    critic was trained with it; omitting it degrades every reconstruction).
  - **The edit is deterministic — no judge in the loop.** Inside the AV read text, string-substitute
    the decoy identifier term with the matched true term from `rename_map`
    (`substitute_terms(..., require_single=True)`). A span whose read does not contain its decoy term
    is **unedited and recorded as such** — that count is H-W3's rate and it bounds how many spans stage 1
    can act on at all.
  - **The write:** `h ← ‖h‖ · unit(AR(edited))` at every token position of the span. Norm-matched
    per position, so there is **no α** — this is the point of using the AR natively.
  - **Primary readout: `G_sum`** (total nats, all 60 items) per the 2026-09-03 method fix; **accuracy is
    secondary and declared underpowered in advance** (6/60 flippable → a perfect rescue is +0.10, the
    support threshold itself).
  - **Controls, all three required for H-W1 to be interpretable:**
    | id | arm | must show |
    |---|---|---|
    | **C1** | write back AR(**unedited** read) | ≈ 0. If the round trip alone moves G_sum, every stage-1 number is round-trip artifact, and the family is **UNINFORMATIVE**, not null. |
    | **C2** | substitute a **foreign** item's true term (same edit shape, wrong content) | must not match the edited arm — else the effect is generic perturbation, the `R_random` failure again |
    | **C3** | write back AR(AV(h at the **matched L0 span**)) — the NLA round trip of the *clean* state | the **ceiling for this channel**. If C3 is null, no NLA write at these positions can work and H-W1's null is about the channel, not the edit. |
  - **Behavioural veto (inherited verbatim from R2):** an arm is disqualified if greedy accuracy or
    parse rate drops by **> 0.05** vs `baseline.jsonl`. This is the clause whose absence produced the
    `S-LIVE` artifact; it is not optional and it is not re-tunable after seeing data.
  - **Stats:** percentile bootstrap over items, **10,000** resamples, **seed 20260724**; two-sided
    permutation p; **BH-FDR across the family**. `--deterministic` stays **OFF** (2026-08-29: it
    changes answers and forks the corpus).

- **Decision table (frozen). The 2×2 is the point of the design — it is what makes a null mean something.**

  | H-W1 | H-W2 | conclusion |
  |---|---|---|
  | moves | registers | the NLA protocol steers — the first positive in this programme |
  | **null** | **registers** | **readout ≠ mechanism**: the model's own verbalized state changed and behaviour did not. The strongest form of the Phase-1b thesis, measured rather than inferred |
  | null | does not register | the channel did not deliver → **UNINFORMATIVE about the NLA claim**, reported as such, never as a null |
  | moves | does not register | generic perturbation → adjudicate against C2 |

- **Power, stated in advance.** n = 60 items on G_sum (not 6 flippable), CV 0.353. A +12.11-nat shift is
  ~0.28 SD of the item distribution; paired bootstrap at n = 60 detects it comfortably. The binding
  limit is **multiplicity across arms**, not sample size — hence BH across the family and no post-hoc
  arm additions.

- **Results:** *not yet run — this file is the pre-registration.*
- **What worked / hypothesis verdict:** *pending.*
- **Observations:** *pending.*
- **New questions / new hypotheses:** *pending.*
- **Next Steps:** implement stage 0 (`nla/src/nla_cycle.py`), run it, and **stop at the gate** — stages 1
  and 2 do not launch unless H-W0a and H-W0b both pass.
