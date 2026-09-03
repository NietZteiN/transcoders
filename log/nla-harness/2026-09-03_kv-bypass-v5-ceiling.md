# 2026-09-03 — KV bypass: V5 state-replacement ceiling (pre-registered addition)

**Thread:** nla-harness · **Extends:** [`2026-09-03_kv-bypass-prereg.md`](2026-09-03_kv-bypass-prereg.md) · **Status:** frozen — **no arm has produced a result yet**

Filed before any steering generation exists, for the reason the project's own discipline demands:
adding an arm after seeing the primary is a forking path, adding one before is just design.

---

## The gap this closes

Every arm in the original pre-registration writes a **direction derived at unsteered
activations**. Once layer 0 is edited, layer 1 no longer sees the state its own vector was
measured at, so multi-layer steering is a *compounded perturbation*, not a transport of the item
to its clean state.

That leaves the null outcome (H-C0) open to one obvious objection: **"the vectors were too weak."**
Nothing in the design as filed answers it, and the objection is not cheap to dismiss after the
fact.

## V5_replace — an exact ceiling

Assign, at the steered position and at **every layer 0..L**,

```
h_l  :=  h_clean,l
```

so the position's state simply **is** the clean run's state everywhere the KV cache will be read
from. If the model still cannot answer under that, no vector-quality argument survives — and the
bypass is decisively not the explanation.

Labelled a **ceiling, not a deployable intervention**: it requires the clean program, exactly as
V4 does. Every table must carry that label, per the same rule the banked V4 rows carry.

## The implementation error this exposed, and why it mattered

My first attempt added the raw difference `h_clean,l − h_obf,l` at every layer, on the reasoning
that `h_obf + (h_clean − h_obf) = h_clean`. **That is wrong once there is more than one layer.**
Layer *l+1* receives the state layer *l* has already corrected and adds its own difference on top,
so the error accumulates linearly with depth.

Caught by a CPU test with no model, which is the only reason it was caught at all — the arm would
have run, produced numbers, and been reported as an exact ceiling while being a roughly-3×
overshoot at the top layer:

| layer | max |h − h_clean| under additive "replacement" |
|---|---|
| 0 | 1.19e-07 |
| 1 | 2.63343 |
| 2 | 5.26687 |
| 3 | 7.90030 |

Only an **absolute assignment** is idempotent with respect to what propagated up from below.
Re-verified after the fix: **error exactly 0.0 at every layer**, no other position touched, the
additive path still additive, `absolute + energy_matched` and `deltas + absolute` both refused
rather than silently resolved, and α = 0 still byte-identical on the additive path.

Consequences recorded because they are the kind of thing that rots:
- The bank now stores **absolute clean activations** alongside the differences.
- Banks written before this carry no `clean` array. V5 is then **dropped with a printed warning**,
  never silently falling back to the additive oracle — a fallback would report a ceiling that is
  not one.
- V5 returns its targets under a sentinel key so it **cannot** be routed through the additive path
  by accident. A silent mix-up would look like a weak ceiling rather than a bug, which is the
  failure mode that is hardest to notice.

## Decision rule — frozen

V5 is **diagnostic, not a hypothesis test**, and cannot on its own support or refute H-C0/H-C1:

- **V5 recovers accuracy (Δ vs single-layer V3 ≥ +0.10, CI excluding zero)** → a clean state at
  the read site is sufficient. If matched multi-V3 did *not* recover, the correct reading is
  **"the channel can deliver, but contrastive directions are not what it needs"** — which is a
  statement about the vectors, not about beliefs, and must be written that way.
- **V5 does not recover** → the strongest possible write cannot move the task. H-C0 stops being
  "no effect found" and becomes **bounded**: not a null of the instrument, but of the intervention
  class. This is the outcome that makes the causal negative worth publishing.
- **V5 destroys generation** (parse rate collapse toward P0.2's 0.100) → the read position is
  load-bearing for fluency and the arm is **UNINFORMATIVE**, reported as such, not as a null.

## Provenance

`nla/src/steer_multilayer.py` (`MultiLayerSpec.absolute`), `nla/src/steer_run.py`
(`V5_replace`), `nla/src/multilayer_vectors.py` (clean-activation bank).
Exactness proof is CPU-only and reproducible without a GPU. Seed 20260724.
