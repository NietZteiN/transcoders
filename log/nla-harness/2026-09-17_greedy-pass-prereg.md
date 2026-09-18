### Target Date: 2026-09-17 (H-R18 — the greedy pass: make every contrast a deterministic function of the corpus)

- **Hypotheses / what we're testing:**

  H-R15 established two things that together make this run cheap and decisive.
  (1) **`SEED-DETERMINISTIC`** — at a fixed seed the pipeline reproduces *exactly* (per-item agreement
  1.0000, |Δacc| = 0.000000, across machines). (2) **`CI-OPTIMISTIC`** — a different sampling draw moves
  contrasts by up to **1.08×** their published half-width, so every interval this thread has reported on
  a single sampled run is ≈1.2× too narrow, and **`codesteer_auto`'s +0.0379 evaporated to −0.0031**.

  Greedy decoding removes that entire noise source: with `do_sample=False` there is no draw, so a
  contrast becomes a **deterministic function of the corpus** and the cluster bootstrap over snippets
  becomes the *complete* uncertainty rather than a lower bound on it. One run per case then suffices,
  which is why re-running all 8 arms costs ~3 GPU-h instead of ~18.

  **This is a declared deviation from the paper's protocol, and it does not replace the sampled runs.**
  Their README sets T 0.7 / top_p 1.0 / top_k 7 and their metric is Pass@k — "passed at k if at least one
  of k runs matches" — which presupposes sampling; under greedy `pass@1 == pass@2 == pass@3` and the
  metric degenerates. So the greedy pass is a **complement**: it answers "what does this intervention do
  when the decoder is not a lottery", while the sampled runs remain the protocol-faithful comparison to
  their tables. Neither supersedes the other, and the H-R14 sampled numbers are not withdrawn.

  - **H-R18a (the enabling check).** Two greedy runs of `unsteered` on the same snippets.
    **`GREEDY-EXACT`** if per-item label agreement is 1.0000 and |Δacc| ≤ 1e-9 ·
    **`GREEDY-NOT-EXACT`** otherwise. *Prediction: `GREEDY-EXACT`* (implied by H-R15b, but it is the
    premise the whole run rests on, and it costs two snippets to check). **If it fails, H-R18b/c are not
    read** — a non-deterministic greedy decoder would mean the H-R15b result does not transfer and the
    noise model is wrong again.
  - **H-R18b (do the verdicts hold without the lottery?).** Re-read **H-R14a–e under the *same frozen
    thresholds*** — `DAMAGE-PRESENT/WEAK/ABSENT` at ±0.03; `CODESTEER-RESTORES/INERT/HARMS` at ±0.05,
    α/2; `NLA-BEATS-CODESTEER / MATCHES / CODESTEER-BEATS-NLA` at ±0.05, α/3; `PROMPT-SUFFICES`;
    `DAMAGE-FAR-WEAKER`; restoration ratio gated on `DAMAGE-PRESENT`. **Nothing is retuned** — these were
    frozen on 2026-09-16 before any full-corpus data existed and are simply applied to a second decoder.
    *Predictions: `DAMAGE-ABSENT` · `CODESTEER-INERT` · `NLA-MATCHES-CODESTEER` · `PROMPT-SUFFICES` ·
    `DAMAGE-FAR-WEAKER`* — i.e. **the same five words**, on the reasoning that H-R15 showed the one
    apparent exception was a draw artefact. The interesting free parameter is the *sign* of
    `codesteer_auto`, which was +0.0379 at one seed and −0.0031 at another; I decline to predict it.
  - **H-R18c (descriptive, no verdict).** The level shift from sampling to greedy: accuracy, parse rate
    and accuracy-given-parsed per arm. Expected direction is greedy ≥ sampled on a binary task (argmax
    beats a T 0.7 lottery), so **the damage and the contrasts, not the levels, are the comparable
    quantities across decoders.**

  **Pre-committed:** whatever H-R18b returns is reported beside the sampled numbers in the same table,
  including if greedy makes an arm of ours look better. A result that appears only under a decoder we
  chose, and not under the paper's, is labelled as such.

- **Setup:** the frozen 148-snippet set, same packs, same runtime, same chat template, **same cross-fitted
  vectors** — only the decoder changes. All 8 H-R14 arms: `original_unsteered`, `unsteered`, `codesteer`,
  `codesteer_auto`, `ridge_map`, `erasure`, `prompt`, `swap_oracle`. `--runs 1` (the runner **refuses**
  `--greedy` with `--runs != 1`, since greedy runs are byte-identical and would triple the cost for no
  information). `--greedy` added to `ase_steer_run.py`, new sha `a73a3fd93ad8b4c5…` (was
  `9fc1389f2075acf5…`); it flips `do_sample` on **their** `run_llama`, so argmax is taken by their own
  `_sample_next_token`, and it stamps `decode` into each run's provenance. Seed still passed (irrelevant
  under argmax, recorded anyway).

  ~3 GPU-h: the six non-attention arms at ~12–15 min each and the two CodeSteer arms at ~40 min each,
  **unsharded** — at one run per case they fit the wall comfortably, which also removes the shard-concat
  step. Three `--dependency=singleton` lanes, ≤ 3 concurrent GPU jobs. Scored with `ase_r14_stats.py`
  (sha `e16257bacd379aaa…`, unchanged) so the decision code is identical to the sampled pass.

- **Results / verdict / observations:** pending — this entry is the pre-registration.

- **Next Steps:** smoke H-R18a → 8 arms → score under the frozen H-R14 rules → results entry, and a
  side-by-side sampled-vs-greedy table in the report.
