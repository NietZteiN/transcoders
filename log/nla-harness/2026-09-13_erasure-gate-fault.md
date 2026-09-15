### Target Date: 2026-09-13 (E-HARNESS-FAULT — the erasure identity gate fired, and the fault is in the GATE, not the science)

- **Hypotheses / what we're testing:** nothing. This is a **fault and deviation entry**, written **before** the
  re-run and before any erasure number is read as a result. Job **392723** ran the pre-registered erasure-vector
  experiment ([`2026-09-12_erasure-vector-prereg.md`](2026-09-12_erasure-vector-prereg.md)) and its identity gate
  refused the run, exactly as designed: *"Identity gates: `erase_own` == banked `swap`, `edit`/`foreign`/`swap`
  reproduce banked rows (≤ 0.05 nats), `self` == 0 → else exit 3, `E-HARNESS-FAULT`."*
  **No H-E1/H-E2/H-E3 verdict is assigned and no erasure arm mean is quoted anywhere in this entry.** The prereg
  says a failed gate means nothing is reportable, and that is honoured rather than relitigated now that the
  numbers exist on disk.

- **Setup:** job **392723**, `sbatch nla/scripts/nla_erasure_sbatch.sh`, node **g-06-01**,
  `2026-09-13T08:1xZ`, **14:37 elapsed, exit 1** (the runner returned 3; the sbatch `|| exit 1` maps it).
  Pytest and the 3-item smoke both passed in-job. All 60 items completed (1 020 forwards) before the gate was
  applied. Output `data/nla/ml/gemma4b/gate/erasure/` — **quarantined**.

- **Results.** The gate's report:

  ```
  [ERASE] IDENTITY FAILED: {'erase_own': 1.201, 'edit': 0.0, 'foreign': 0.0, 'swap': 0.0, 'self': 0.0}
  ```

  **Three of the four score-level checks reproduced the banked rows at exactly 0.0, and `self` is exactly 0.0.**
  Only `erase_own` deviates, at **1.201 nats** worst-case against a 0.05 tolerance.

  Diagnosis, computed offline from `data/nla/ml/gemma4b/gate/vectors/L7.npz` with no GPU (471 spans × 2 560 dims,
  all arrays **float32** — no dtype mismatch):

  | quantity | measured |
  |---|---|
  | `max abs(erase_own_vec − h0)` in fp32 | **3.05 × 10⁻⁵** |
  | spans where the two vectors are bit-equal in fp32 | 2 / 471 |
  | components differing after the bf16 cast the write performs | **111 of 1 205 760 (0.0092 %)** |
  | spans with ≥ 1 differing bf16 component | 97 / 471 |
  | max abs bf16 component difference | **3.81 × 10⁻⁶** |
  | for scale: ‖h1b‖ mean | 5 427 |

- **What worked / hypothesis verdict:** **the gate worked and the harness is sound; the gate's *specification* was
  wrong.** `erase_own` is built by arithmetic — `h1b + ‖Δ‖ · unit(Δ)` — which equals `h0` to **3 × 10⁻⁵** on a
  vector of norm 5 427, i.e. the erasure code path is correct (and the unit test
  `test_erase_own_reproduces_h0_exactly_in_float32` passes, because in fp32 it *is* correct). But it is not
  **bit-identical** to `h0` for 469 of 471 spans, and 111 tiny-magnitude components land on a different bf16 value
  after the cast. **A perturbation of 4 × 10⁻⁶ on 0.009 % of components moves the summed teacher-forced logp by up
  to 1.2 nats** — the bf16 forward is chaotic at that scale, and a per-token shift of ~0.002 over several hundred
  reply tokens is all it takes.
  **The assumption the 0.05-nat tolerance encoded — "mathematically identical input ⇒ identical score" — is false
  in bf16.** It held for every previous gate in this programme only because those arms wrote **banked bytes**
  (`arr["h0"]`, `arr["edit"]`, `arr["foreign"]`), which is why they still report exactly 0.0 here. `erase_own` is
  the first arm ever built by arithmetic, and it is the first to trip the gate.
  **Defect #14 (mine), specification-class, caught by the mechanism designed to catch it.**

- **Observations:**
  - **This is not a tolerance to loosen.** Raising `IDENT_TOL` after seeing the number would be exactly the frozen-
    threshold retuning the project forbids. The fix instead **tests the claim where the claim lives**:
    `erase_own`'s purpose is to prove *the erasure arithmetic reconstructs `h0`*, which is a statement about
    vectors, not about scores. So the vector check becomes the gate and the score-level `erase_own` comparison
    becomes a **reported diagnostic** — the measured bf16-chaos magnitude — rather than a pass/fail.
    `IDENT_TOL = 0.05` and `SELF_TOL` are **unchanged**, and they keep their teeth on the three banked-byte arms
    where they demonstrably work (0.0 on all three today). No scientific rule (H-E1/H-E2/H-E3, `DIRECTIONAL_NATS`,
    `SHARE_BAR`, α ladder) is touched.
  - **A programme-wide consequence worth more than this experiment:** every identity gate here has been written at
    0.05 nats, and today establishes that this is only valid for **bit-identical** writes. Any future arm
    constructed by arithmetic — interpolations, projections, the length-matched controls proposed in H-A7, the
    β-interpolated writes if ever rebuilt from parts — inherits a **~1-nat floor** that has nothing to do with the
    intervention. The floor should be *measured* rather than assumed, which is H-E4 below.
  - The run's own diagnostics were otherwise healthy: 1 020 forwards, all 60 items, `self` exactly 0.0, the three
    banked-byte arms exact, and the per-item `swap`/`own` pairs agree to ≤ 1.2 nats throughout — consistent with
    chaos, not with a wiring error (a wiring error would not reproduce `swap` at 0.0 while `own` drifts).

- **New questions / new hypotheses:**
  - **H-E4 (measure the floor, ~0.1 GPU-h):** write the `swap` arm twice — once with the banked `h0` bytes, once
    with `h0` perturbed by 1 bf16 ULP on exactly the 111 components that differ above — and report the score
    spread. That converts today's fault into a **measured constant** ("the bf16 write-chaos floor at L7 is X nats")
    that every future gate in this programme can be specified against. Cheap, and it makes the next
    arithmetic-built arm safe by construction.
  - Should the `erase_*` arms be scored against a **bit-perturbed `swap` baseline** rather than the banked `swap`?
    If the chaos floor is ~1 nat, contrasts of 3+ nats (H-E3's `DIRECTIONAL_NATS`) are unaffected, but the H-E2
    share test compares against 0.55 × `swap` ≈ 40 nats and is likewise safe. Worth stating explicitly in the
    re-run's entry so the floor is visible next to the effects it does and does not threaten.

- **Next Steps:** patch `nla/src/nla_erasure.py` (split the gate; add the vector check; keep every frozen number),
  extend `nla/tests/test_erasure.py` with a test that a mathematically-equal-but-not-bit-identical vector **passes**
  the vector gate while a genuinely drifted one still **fails**, re-run, and report H-E1/H-E2/H-E3 from the re-run
  in a new dated entry. Then H-E4.
