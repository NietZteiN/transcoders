# 2026-09-03 — Pre-registration: NLA steering on a permitted host, and token coverage

**Thread:** nla-harness · **Status:** frozen before either run · **Host:** `gemma12b` L32/48

Filed while the two gates (`374534` AR alignment, `374550` coverage energy match) are still
running, so no rule here can be shaped by what they report.

---

## Why these two, and why now

Today's KV-bypass work varied **depth** (1 layer → 33) and returned UNINFORMATIVE. It did not touch
the two things that actually matter to the programme's central question:

1. **NLA steering has never run on a permitted host.** Every arm today was V3/V4/V5/random — pure
   activation arithmetic, no autoencoder. V1/V2 were excluded because the AR on disk is the Qwen
   pair. The banked B4 refutation is Qwen and cannot be re-run under the 2026-09-02 constraint, so
   **the NLA's causal claim currently rests on zero permitted evidence.**
2. **Every steering result ever produced here wrote at ONE token.** `last_prompt` — 1 of ~240
   prompt tokens, 0.4%. The stimuli have carried 11 annotated `identifier_spans` per item since
   Paper 2 and **not one run has used them**, despite the adversarial-rename hypothesis being
   specifically about identifiers.

## Experiment N — NLA steering, Gemma L32, `last_prompt`

Reproduces B4's question on a host the project may run. Conditions: **V1_gloss** (AR-derived, the
NLA's own direction), **V2_wordedit** (one-word minimal edit), **V3_taskvec**, **V4_oracle**,
**R_random**, **F_foreign**, **A_antipodal**. Primary α = 1.0; secondary α ∈ {0.25, 4.0} for V1 and
V3 only, to keep the family small.

**Gate, inherited verbatim from B4 so the two are comparable:** V1 must beat V3, paired on the
programs both scored, at the primary α. The reasoning is unchanged — if V1 does not beat a plain
contrastive difference that uses no autoencoder, the NLA is not doing causal work.

- **N-1 ✓ supported** iff V1 − V3 ≥ **+0.10**, bootstrap CI excluding zero, at α = 1.0.
- **N-0 ✗** iff |V1 − V3| is inside the reproducibility floor (±0.075). **This replicates B4's
  refutation on a second architecture** and is the outcome that makes the negative durable rather
  than host-specific.
- **UNINFORMATIVE** iff neither V1 nor V3 moves off the unsteered baseline by more than the floor.
  Today's run already showed V3 flat on this host, so **this outcome is expected**, and it is
  declared in advance as the likely one. It is not a failure: N is worth running because V1 has
  never been tested here at all, and a V1 that moves where V3 does not would be the single most
  important result in the programme.
- **Sensitivity clause — new, and added because today's rule lacked one.** `R_random` at the
  largest α is the positive control. If it moves the metric by more than the floor while V1 and V3
  do not, the measurement is demonstrably sensitive and the null is reported as **N-0 refuted-not-
  untestable**. Today's UNINFORMATIVE verdict was over-conservative for exactly this reason; the
  fix belongs in the rule, pre-registered, not in the reading of a result already seen.

**AR validity is a precondition, not a finding.** If gate `374534` reports `separation` failed —
`AR(gloss_true) == AR(gloss_decoy)` — then V1 is identically zero and the whole experiment is void;
it is reported as void, not as a null. `alignment` (cos(V1,V3)) is **reported and never gating**: a
low value plus a null means *"the AR is not addressing this space"*, which is a different claim
from *"there is no belief to edit"* — and distinguishing those two is the thing B4 could not do.

## Experiment C — token coverage at matched delivered perturbation

Coverage ∈ {**last_prompt** (1 token), **id_spans** (median 44 tokens, 18.4% of prompt)}.
Conditions: V1_gloss, V3_taskvec, R_random. One α per coverage, set by the frozen rule in
`nla/src/coverage_calibrate.py` and computed **before** any generation.

- **C-1 ✓ site matters** iff a condition at `id_spans` beats the same condition at `last_prompt`
  by ≥ **+0.10**, CI excluding zero, with `R_random` at `id_spans` not matching the gain.
  → the banked single-token nulls were testing the wrong site, and **every write-side result in
  this programme is scoped to a position choice nobody validated.**
- **C-0 ✗** iff the difference is inside the floor → the single-token choice was not the
  limitation, and the banked nulls keep their scope.
- **C-2 energy, not site** iff parse rate at `id_spans` collapses toward P0.2's 0.100, or the
  gain tracks perturbation magnitude rather than placement → reported as a negative for C.

**The matching choice is a judgment call and is stated, not hidden.** Delivered perturbation is
equalised **at the last prompt token** — the position every coverage setting shares and where the
answer is produced. Matching total injected energy across all edited positions instead would leave
the wide arm much weaker at the read-out site. Both are defensible; this one follows the question
being asked. A reader who prefers the other convention should read C as answering *"does moving the
edit to the identifier sites, at equal delivered effect on the answer position, change the
outcome"*.

## Common to both

- **Reproducibility floor ±0.075 binding** (0.85–0.90 per-item greedy agreement, 2026-08-29).
  The unsteered baseline is re-run in the same job; `--deterministic` stays OFF because it changes
  answers and would fork the corpus.
- **Prompting baseline runs automatically** and every steering claim is reported against it, per
  charter §4. A steering gain that does not beat prompting is a mechanism claim, not a method.
- **Reply-level identity is recorded**, not just accuracy. Today's run showed the intervention can
  rewrite every word while leaving correctness inside the floor; that distinction must survive
  into these tables.
- Seed 20260724. Both outcomes of every rule above are declared publishable in advance.

## Provenance

`nla/src/{steer_run.py, ar_gate.py, span_positions.py, coverage_calibrate.py}`; AR resolved per
host from `AR_CHECKPOINTS`, layer cross-checked against the checkpoint's own `nla_meta.yaml`
sidecar (Gemma: layer 32, d_model 3840, mse_scale 61.968).
