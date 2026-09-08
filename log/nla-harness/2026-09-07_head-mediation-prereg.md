### Target Date: 2026-09-07 (H-W31 — which downstream heads and MLPs carry the NLA-transported clean state? Frozen before running.)

Follows [`2026-09-05_nla-fidelity-results.md`](2026-09-05_nla-fidelity-results.md) (C3pure **+44.94**
vs P_patch **+45.71** G_sum nats, 98.3 % fidelity) and the meaning decomposition in
[`2026-09-06_tier-source-results.md`](2026-09-06_tier-source-results.md). The W family has shown
*that* writing AR(AV(h_L0)) at the L1b identifier spans of layer 32 repairs the reply, and *what*
in the written state does it (≈ 51 % identifier meaning / 49 % decoy removal / 1 % structure).
This entry asks *through which components* of layers 33–47 the repair travels. **Written before
any patched forward has run.** Config: [`../../nla/configs/nla_heads.yaml`](../../nla/configs/nla_heads.yaml).

- **The structural fact that makes the question well-posed.** The write lands on *prompt* positions
  (the spans); G_sum is scored on *reply* tokens (`l0_reply_ids` teacher-forced after
  `l1b_prompt_ids`). A reply position's residual stream never contains a span position's residual
  directly, so **every nat of the +45 crosses positions through at least one attention head in
  layers 33–47**. Two consequences are fixed in advance: (i) "all 255 components of 33–47 patched
  from the steered run into the unsteered run" must reproduce dG(S) exactly — a pre-registrable
  identity, not a result; (ii) Gemma-3-12B's sliding window is 1024 with every sixth layer global
  (downstream: **35, 41, 47**), so a reply token > 1024 positions past the last span can reach it in
  one hop only through a global head (17/60 items have T ≥ 1024).

- **Method — activation patching between two runs of the same sequence.** Per item, per arm:
  **U** = unsteered L1b run; **S** = `PositionReplacer` writing the arm's vector at the spans at
  L32. Both runs cache every head's pre-`o_proj` slice (`[T,16,256]`) and every MLP output
  (`[T,3840]`) at layers 33–47. Per component c (240 heads + 15 MLPs), patched at all positions
  (positions before the first span are identical in U and S):
  - `suf_c = dG(U, c←S)` — sufficiency;
  - `nec_c = dG(S) − dG(S, c←U)` — necessity;
  - `read_{L,h} = logp_mref[L] − logp(S, head h at L blind to the span keys)` — a read-side
    knockout (bool sdpa mask, `m[0,h,:,span_keys]=False`), defined against a **mask-only
    reference** `logp_mref[L]` because a materialised mask moves that layer off the flash kernel.
  Then a joint pass: greedy top-k by `suf` for k ∈ {1,2,4,8,16,32} **selected on the opposite
  half** (`crc32(snippet_id) & 1`), against a uniform random-k null (3 seeded draws), a
  layer-matched random-k null, and `ALL`.
  Arms: **C3pure** (NLA-transported; the object of study) and **P_patch** (raw h_L0; comparator).
  Vectors are computed once on the **repaired** anchoring (`--repair`, 471 spans / 60 items) and
  persisted, so the sweep runs with the host alone and the repaired-corpus fidelity ratio
  (**H-W25**, still owed) falls out as a secondary.

- **Hypotheses (frozen).**
  - **H-W31-ID (gate).** Every `SELF_c` (S with c←S) within **0.05 nats** of dG(S); `ALL` within
    **5 %** of dG(S); every `ko_gap[L] = logp_mref[L] − logp_S` within 0.05 nats. Any failure →
    exit 3, verdict `W31-HARNESS-FAULT`, nothing below is reportable. 0.05 is a determinism
    tolerance, deliberately tighter than the banked SELF arm's 1.0 (which absorbs bf16
    re-quantisation of a norm-matched vector); the hooks here copy the module's own bf16 tensor.
  - **H-W31a (localisation).** `CONCENTRATED` if TOP_16 (6.7 % of heads) recovers ≥ **50 %** of
    dG(S) *and* exceeds RAND_16 by ≥ **30 points** of dG(S) with the paired CI excluding 0.
    `DISTRIBUTED` if no k ≤ 32 reaches 50 %. Else `INTERMEDIATE`, reporting the first k that
    crosses 50 %.
  - **H-W31b (heads vs MLPs).** Best single head vs best single MLP `suf`, and their shares of the
    TOP_16 set. Descriptive.
  - **H-W31c (same route?).** Spearman ρ between the C3pure and P_patch per-component `suf`
    profiles (255 item-means) and Jaccard of the two TOP_16 sets. `SAME-CIRCUIT` if ρ ≥ **0.80**
    and Jaccard ≥ **0.50**; `DIFFERENT-CIRCUIT` if ρ < **0.50**; else `INDETERMINATE`. **Prior:
    SAME-CIRCUIT** — 98.3 % fidelity leaves little room for a different route. The informative
    residue is any component where C3pure `suf` < P_patch `suf` by a CI-clearing margin.
  - **H-W31d (global heads).** Global-layer heads (35/41/47; 48 of 240) carry larger mean `nec`
    than local-layer heads, and the gap is larger on items whose reply extends > 1024 tokens past
    the last span. Paired contrast on `nec` (primary) and `read` (secondary); clears if the CI
    excludes 0.
  - **H-W31e (direct readers).** Spearman ρ between per-head `read` and `nec`, and the share of
    the TOP_16-by-`nec` heads whose `read` CI excludes 0. Descriptive.
  - Reported without a rule: the per-layer depth profile of summed `nec` (33→47); the sum of
    single-component `suf` vs dG(S). **Singles need not add**: `post_attention_layernorm` acts on
    the *sum* over heads, so the `o_proj` input is the only well-defined per-head site and the
    joint pass, not the sum, is the localisation evidence.

- **Setup (frozen).** `nla/src/nla_heads.py`, stages `vectors` (host + AV + AR; `--repair`) →
  `sweep --read-knockout` → `joint`; host `gemma12b` L32/48; AV/AR `kitft/nla-gemma3-12b-L32-{av,ar}`
  (snapshots recorded in `vectors_manifest.json`); G_sum on `l1b_prompt_ids + l0_reply_ids`;
  `use_cache=False`, `logits_to_keep=len(reply)+1` (identical numbers, less memory); paired
  cluster bootstrap N_BOOT = 10,000, seed 20260724; `--deterministic` OFF; `arm_guard` position
  stamps on every row (`n_pos_S`, `n_pos_ALL`, `n_pos_TOP_k`) so a joint arm that patched fewer
  components than it claims is refused rather than averaged. Smoke: 3 items, single-component
  list restricted to layers 45–47 (hooks and `ALL` still span 33–47 — `ALL` is an identity only
  when every layer above 32 is covered); all identities must pass before the full run starts.
  Cost: ≈ 62.5 k sweep + 30.6 k knockout + 5.2 k joint forwards; estimated 2–3.5 h on one H200.

- **Rejected in advance.** Attribution patching (gradient approximation; unnecessary at this cost
  and poor on large-effect heads). Switching the model to eager attention for the knockout
  (changes the kernel for all ~98 k forwards; the per-layer bool mask plus mask-only reference
  gives the same measurement on sdpa). Zero/mean ablation (measures task necessity, not mediation
  of *this* write). Reusing obtune's `attention_knockout` verbatim (it adds a float bias to what
  is a **bool** sdpa mask under transformers 5.12.1 — this would destroy the causal structure and
  mismatch the query dtype).

- **Stated in advance.** I expect `INTERMEDIATE` on H-W31a (a handful of late global heads plus a
  long tail), `SAME-CIRCUIT` on H-W31c, and H-W31d to clear on `nec`. I record these so that a
  `DISTRIBUTED` verdict cannot be re-read afterwards as "expected". A `DIFFERENT-CIRCUIT` verdict
  would be the more interesting outcome for Phase B (multi-layer NLA): it would say the NLA state
  is *read* differently from the raw state despite being scored the same.
- **Bounds.** Patching at all positions conflates "carries the span state" with "carries anything
  that differs between U and S downstream of the spans" — by construction those are the same set
  here (U and S differ only at the spans at L32), but the interpretation is mediation of the
  *write*, not of the identifier meaning per se. Split-half top-k selection removes the
  selection-evaluation circularity; it does not remove the 60-item ceiling on power.
- **Results / verdict:** *not yet run — this file is the pre-registration.*
