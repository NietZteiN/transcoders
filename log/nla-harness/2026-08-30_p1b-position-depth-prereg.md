### Target Date: 2026-08-30 (PRE-REGISTRATION — position × depth: when does correctness become readable?)

**Frozen before any generation or extraction.** Follows
[`2026-08-30_p1b-read-depth-results.md`](2026-08-30_p1b-read-depth-results.md) (correctness rises
with depth to AUC 0.7612 at L27) and
[`2026-08-29_reliability-floor-and-length.md`](2026-08-29_reliability-floor-and-length.md)
(`rt_cos` adds nothing pooled, but **does** at the `answer_line`, q = 8e-05).

- **Hypotheses / what we're testing:**

  Two independent results this week point the same way, and both are read-side — the half proven
  bit-exact and free of the reproducibility floor:

  | | signal | where |
  |---|---|---|
  | 2026-08-29 | `rt_cos` beats length by ΔCV-AUC +0.0267, q = 8e-05 | **`answer_line` only** — nothing across `reasoning` (p = 0.59) |
  | 2026-08-30 | dense correctness probe 0.6205 → **0.7612** | **late layers only** — monotone L12 → L27 |

  **Every item-level read in this programme has been taken at `last_prompt`** — B4's vectors, P0.1,
  P0.2, P0.4, and yesterday's probe. If correctness only becomes decodable late in *both* senses —
  late layer and late position — then the ledger of item-level nulls was not measuring absence, it
  was measuring **too early**. That is the readout paper's candidate positive and it is testable
  in under an hour.

  - **H-PD1 (primary — position, at fixed depth).** Is correctness more decodable at the
    `answer_line` than at `last_prompt`, with the layer held at 20?
  - **H-PD2 (the branch that decides what may be claimed).** If position matters, does the signal
    appear **mid-reasoning** or only once the answer is effectively decided?
  - **H-PD3.** The full 28 × 6 grid, reported as a surface. Exploratory by construction.

- **Setup:**
  ```
  items    the 60 L1b (adversarially renamed) stimuli — the failure mode of interest.
           Correctness is the unsteered greedy grade on that item.
  generate max_new_gen 2048 (NOT 1100): a 0.8833 parse rate would censor 12% of items and
           `answer_line` is undefined for a reply that never emits one.
  extract  ONE forward pass over prompt+reply, output_hidden_states, all 28 layers at 6 positions
  probe    logistic, L2, C = 1.0 frozen · GroupKFold(5) on snippet_id · bootstrap over snippets
  ```

  **Positions, defined now so none can be added later:**

  | tag | definition |
  |---|---|
  | `last_prompt` | final prompt token — the incumbent site every prior read used |
  | `reply_q25` / `reply_q50` / `reply_q75` | 25 / 50 / 75 % through the generated span |
  | `answer_line` | the token immediately preceding the answer value after the `Output:` marker |
  | `last_token` | final generated token |

  Token indices for `answer_line` come from cumulative per-token decoding of the generated span,
  so the char→token map is exact for the ids actually generated rather than for a re-tokenisation
  of the decoded string. Items with no `Output:` line have no `answer_line` position and are
  excluded **from that cell only**, with the cell's n reported.

  **Three mandatory controls.**
  1. **Length.** Reply length alone predicts correctness at AUC 0.693 (2026-08-29, inverted), and
     position indices are a function of length. Every cell is therefore also fitted with a
     length-only probe, and the primary requires the activation probe to beat it by ≥ +0.05.
     Without this a "late position is better" result could be a length result.
  2. **Permutation.** A null **distribution** of 200 draws per tested cell, not one draw. The
     2026-08-30 probe froze a single-draw threshold against a max-over-28 statistic and misfired
     on its own control; that error is not repeated.
  3. **Grouping.** GroupKFold on `snippet_id`, as before.

- **Decision rules — frozen:**

  Statistic: grouped-CV AUC, bootstrap CI over snippets. n = 60 with ~28 positives is
  **underpowered and stated as such in advance**; CIs will be wide and a null is not evidence of
  absence.

  **H-PD1 primary — `answer_line` vs `last_prompt`, both at layer 20.**
  - **POSITION MATTERS** iff AUC(`answer_line`) − AUC(`last_prompt`) **≥ +0.10** with a bootstrap
    CI excluding zero, **and** the `answer_line` probe beats its length-only baseline by ≥ +0.05.
  - **POSITION DOES NOT MATTER** otherwise. Then the item-level nulls stand as measured, reading
    earlier was not the mistake, and the readout paper has no positive from this direction.

  **H-PD2 — only evaluated if H-PD1 returns POSITION MATTERS.** The identical rule applied to
  `reply_q50` vs `last_prompt`:
  - **EARLY KNOWLEDGE** — `reply_q50` also clears it. The model represents its eventual
    correctness *before* committing to an answer. This is a genuine predictive claim and the
    strongest available outcome.
  - **LATE READOUT ONLY** — only `answer_line` / `last_token` clear it. **Then the probe is reading
    a decided answer, not predicting one, and no predictive claim is licensed.** L27 is one block
    from the logits and the answer line is one token from the value; a signal confined there is
    close to reading the output. This is the sceptical branch and it is written down first
    precisely because the alternative is more attractive.

  **H-PD3.** The 28 × 6 surface is reported whatever the primary says, explicitly as exploratory.
  BH-FDR across the 6 positions at layer 20; the grid itself carries no inferential claim.

  **Standing constraints:** full sample only; the length and permutation controls are reported
  beside every headline number, not in an appendix; and this cannot revisit P0.4 or the depth
  result — a position effect would not mean those were wrong, it would mean they were measured at
  the wrong *place*, which is a different and compatible statement.

- **Results:** *(none — this is a pre-registration)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** `nla/src/p1b_position_depth.py`, one GPU, ~1 h.
