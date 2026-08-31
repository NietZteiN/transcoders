### Target Date: 2026-08-06 (S1 — async AV client: 6.3× faster, but FAILS the equivalence gate)
- **Hypotheses / what we're testing:** Pre-registered gate: does a concurrent AV client produce the same reads as the sequential one? Bar: **≥95% byte-identical text, max |Δrt_cos| < 0.01, ≥3× throughput.** Pass → N5 runs in one ~3 h night; fail → Option B (sequential, stride 20, ~8.8 h).
- **Setup:** New [`../../nla/src/nla_client_async.py`](../../nla/src/nla_client_async.py) (httpx + semaphore; `_build_embeds` and the request body reused **verbatim** from the vendored client — injection is never reimplemented). Gate harness [`../../nla/src/async_gate.py`](../../nla/src/async_gate.py). 120 banked vectors from 40 cases, AV server on GPU 2, temp 0, max_new 180. Seed 20260724.
- **Results:**
  - **Throughput: PASS, emphatically — 6.31× (6,066 reads/hour vs ~940).** 449 s sequential → 71 s async at concurrency 8.
  - **Equivalence: FAIL — only 18/120 (15%) byte-identical**, max |Δrt_cos| **0.053** (bar 0.01), mean 0.0049.
  - **The decisive control: the AV is perfectly deterministic sequentially.** Running the same 60 vectors twice through the sequential client: **60/60 identical, max |Δrt| = 0.0000.** So the divergence is *caused by concurrency*, not by inherent sampling noise — the 95% bar was achievable, and async misses it by a mile.
  - **It is not a concurrency-level knob:** at concurrency **2** identity is **18%** (1.9× speedup); at **4**, also **18%** (3.6×). Any batch >1 changes the decode. **The AV's greedy decode is not batch-invariant** — sglang's kernel reductions differ with batch composition, and that flips the argmax on ~82% of reads.
  - **Verdict: Option B. N5 runs sequential.** At the measured 3.77 s/read (consistent with the 3.83 s/read regression from the overnight log), stride 20 → ~7,900 reads → **~8.4 h, one night**.
- **What worked / hypothesis verdict:** Gate **FAILED** on equivalence, **PASSED** on throughput. Honoring the pre-registered rule rather than moving the bar, because the divergence is not cosmetic for this project: the primary data *is* the read text (N7 judges text; N8 matches answer mentions in text), and the entire banked corpus was produced sequentially — adopting async would make new reads non-comparable with the 5,090 already collected.
- **Observations:** (1) mean |Δrt| 0.0049 is ~9% of one SD (0.054), so async reads are *semantically* near-equivalent even when textually different — that is why this needed a decisive control rather than a judgement call. (2) The async client is kept in the tree, unused, with the gate result recorded next to it: if a future analysis needs bulk reads where exact text does not matter (e.g. aggregate statistics only), it is a 6× lever with a documented caveat. (3) Three integration bugs found while wiring it: `_build_embeds` takes the prompt **positionally** and returns a `(embeds, prompt_len)` **tuple**; the response is a list-or-dict; and it requires a **torch tensor**, not ndarray. All three would have produced silently wrong or crashed reads.
- **New questions / new hypotheses:** Would sglang's `--enable-deterministic-inference` (present in this build's server args) make batched decode batch-invariant? If so the 6× lever becomes usable without sacrificing comparability — worth one gate re-run before N5 if the flag works on this version.
- **Next Steps:** N5 sequential at stride 20 (~8.4 h, one night). Optionally test the deterministic-inference flag first — it is a cheap re-run of the same gate and would halve the remaining compute for this whole program.

---

### Addendum (same day): `--enable-deterministic-inference` tested — it works, but it is an either/or

| server mode | async == sequential | fresh sequential == **banked corpus** |
|---|---|---|
| default (flashinfer + flashinfer sampling) | 18/120 = **15%** ❌ | **60/60 = 100%** ✓ |
| `--enable-deterministic-inference` (fa3 + pytorch sampling) | **60/60 = 100%** ✓, **7.1×** | **0/60 = 0%** ❌ |

The flag does exactly what it advertises: batched decode becomes batch-invariant, so the async client
becomes *exact* and the 7× lever is real. But enabling it switches the attention and sampling
backends, and that changes every read — **not one of 60 matches the banked corpus**. Conversely,
default mode reproduces the banked reads **perfectly** (60/60), so the 2026-08-04 corpus is fully
reproducible; it simply cannot be parallelised.

**So the choice is between two goods, not a bug:**
- **Default + sequential** — preserves the corpus of record (5,090 reads, the published faithfulness
  analysis, the gallery and the artifact all cite it); N5 costs ~8.4 h, unattended, which we have.
- **Deterministic + async** — one homogeneous, batch-invariant, exactly-reproducible corpus at ~7×;
  but it is a *new decode regime*, so it cannot be mixed with the banked reads. Doing it properly
  means re-reading all 5,090 banked positions in the new regime (~45 min at 7×) and re-running the
  faithfulness analysis, the gallery and the artifact on the new corpus.

**Decision (confirmed by the user, 2026-08-06):** keep **default mode for N5**, sequential. The 7× saving buys
little on a run that executes unattended overnight, and switching would invalidate already-published
numbers for no scientific gain on this experiment. **Deterministic mode is, however, the right
default for any future fresh corpus** — batch-invariance plus 7× is pure win where there is no
legacy corpus to match. Recorded here so the choice is explicit rather than inherited.
