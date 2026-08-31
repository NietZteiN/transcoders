# 2026-08-17 — B1 ✗ REFUTED: the released NLA does not transfer to Qwen2.5-Coder-7B-Instruct, and it is not a scale artifact

**HB1 refuted.** Pre-registration: `2026-08-15_b1-transfer-prereg.md` (thresholds frozen before
the run; position-defect remedy pre-specified before the subject arm produced a read).

## Setup

```
env     nla-mi · seed 20260724 · layer 20 · GPU 0 · 3.9 h wall
corpus  164 HumanEval Java sources (allocation_replication/data/obf/humaneval)
reads   5 evenly-spaced code positions/snippet, 820/arm, 1,640 total, 0 errors
driver  nla/scripts/b1_transfer.sh · config nla/configs/b1_transfer_gate.yaml
```

The two models ship the **same `tokenizer.json` blob** and a byte-identical chat template, so
identical inputs give identical token grids: both arms were read at **the same 820 positions**.
The only variable is the weights.

## Result — the gate, template-token reads excluded as pre-specified

656 code reads per arm (164 reads on the `assistant` token dropped from each, identically).

| check | verdict | detail |
|---|---|---|
| median `rt_cos` ≥ 0.70 | **FAIL** | **0.694** (control 0.864) |
| within 0.05 of control | **FAIL** | control − subject = **+0.170** |
| `act_norm` within 2× band | PASS | 154.0 (control 111.7) |
| own > foreign, 95% CI | PASS | Δ **+0.355** [+0.344, +0.366] |

**GATE FAIL.** Unfiltered (all 820 reads/arm) the verdict is identical — 0.672 vs 0.846, gap
0.174 — so the conclusion is **robust to the filter**, which moved both medians up by ~0.02 and
the gap by 0.004.

Independently, the 3-snippet plumbing smoke had given 0.703 vs 0.852 (gap 0.149). Three samples,
same direction, same magnitude.

## The failure is representational, not a scale artifact — and that is the interesting part

The obvious boring explanation is that the Coder's activations sit outside the band
`injection_scale = 150` was tuned for. Its median `act_norm` **is** higher (154.0 vs 111.7,
+38%), so this deserved checking rather than asserting.

**It cannot be the cause.** `nla_inference.normalize_activation` rescales every incoming vector
to a fixed L2 norm before injection:

```python
def normalize_activation(v, target_scale):   # target_scale = injection_scale = 150
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)
```

Injection is **direction-only; magnitude is discarded**. A 38% norm difference is normalized
away before the AV ever sees it.

Two consequences:

1. **The "re-derive `injection_scale` from the subject's norm distribution" branch of plan
   v0.3's decision rule is a no-op for this failure mode.** It would renormalize to a different
   constant and change nothing about direction. That branch should be struck from the plan, not
   attempted.
2. **The remaining explanation is that the Coder's layer-20 *directions* have diverged** from
   its own base model's far enough that a frozen verbalizer reads them measurably worse — even
   though the two models share a tokenizer, a chat template, and every architectural dimension.

## The instrument is degraded, not broken

Own-vs-foreign on the subject is **+0.355 [+0.344, +0.366]** — a large margin, CI nowhere near
zero. The Coder reads still carry substantial item-specific information; they are simply less
faithful than the host's (+0.456). And 0/820 reads were mostly-CJK, so injection never failed.

So this is a graded transfer loss, not a collapse. Stated as a number: **cross-model transfer
costs 0.170 of round-trip cosine, about 20% of the host's margin**, on the most favourable
possible transfer target.

## Verdict and consequence

**Branch B.** The paper's NLA subject stays **Qwen2.5-7B-Instruct**, the model the checkpoints
were trained on and on which all banked reads (18,610) already sit. No downstream re-work: B3,
B4 and N13 are all already on that model. What Branch B costs is the *option* of putting the NLA
arm on CodeSteer's own primary model.

**This negative is a contribution, as pre-registered.** The NLA release is silent on cross-model
transfer, and the natural assumption — same architecture, same tokenizer, therefore the
checkpoint transfers — is now measured and false, with the scale explanation ruled out by the
injection maths rather than by hand-waving.

## Limitations

- **One transfer target.** A continued-pretraining sibling (Coder) is the *closest* plausible
  target; this says nothing about how transfer degrades with fine-tune distance. A second target
  (e.g. `DeepSeek-R1-Distill-Qwen-7B`, distilled off Qwen2.5-Math) would turn one point into a
  gradient, at ~2 GPU-h.
- **One layer.** The AR is trained at L20 only, so this cannot separate "L20 diverged" from "the
  model diverged".
- 20% of the raw reads sat on the `assistant` chat-template token because `code_end` was passed
  as `len(align.full)`; excluded here, and the caller is to be corrected before any future run.

## Next

1. Strike the `injection_scale` re-derivation branch from `docs/PLAN_believe_the_lie.md`.
2. Fix `pick_positions`' caller (`code_end` should end at the code, not the template tail).
3. Optional, cheap, and it upgrades the finding: run the same gate on a second, more distant
   fine-tune to get a transfer-vs-distance gradient.
