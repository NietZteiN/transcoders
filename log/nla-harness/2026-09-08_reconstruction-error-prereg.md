### Target Date: 2026-09-08 (H-W36 — is the 1.7 % causal shortfall just reconstruction error? Plus a correction to how I filed this yesterday.)

**Thread:** nla-harness · **Job:** none for H-W36a (CPU) · **Raised by:**
[`2026-09-07_head-mediation-results.md`](2026-09-07_head-mediation-results.md) observation 3 ·
**Corrects:** that entry's own statement of H-W36.

- **Correction, first.** Yesterday I filed H-W36 as *"the residue predicts that the AR's
  reconstruction error is **anisotropic** — larger along the directions `L41H4` and `L46H1` read.
  Measurable from the banked `vectors.npz` with **no GPU**."* **The "no GPU" is wrong**, and the
  reasoning behind it was sloppy in a specific way worth recording. The vectors are the residual
  stream at **layer 32**. The heads in question read at layers **41 and 46**, after eight to
  fourteen more blocks have transformed that residual. To ask whether the error lies along what
  `L41H4` reads, the error has to be *propagated* to layer 41 first, and that is a forward pass per
  item — a GPU job, not an arithmetic one over a `.npz`. I conflated "the vectors are banked" with
  "the question is answerable from the vectors". Same class as the 2026-09-04 `H-W2′` defect, where
  I reasoned about which *position* to read and never about which *layer*.

- **So H-W36 splits.** The cheap half is still worth running and is a real test; the head-direction
  half is deferred and correctly priced.

  - **H-W36a (this entry, CPU, no GPU).** Is the causal shortfall explained by reconstruction
    quality? Per item, the shortfall is `gap_i = dG_S(P_patch)_i − dG_S(C3pure)_i` from the banked
    120 rows of job 382366; reconstruction quality is `cos_i`, the mean over that item's spans of
    `cos(h0, c3)` from `vectors.npz` (corpus mean 0.992). **Frozen rule:** Spearman over the 60
    items, cluster bootstrap 10,000, seed 20260724.
      `W36-RECONSTRUCTION` if ρ(gap, cos) ≤ −0.30 with a CI excluding zero — worse reconstruction,
      bigger shortfall, and the 1.7 % is simply the round trip's error showing up causally.
      `W36-NOT-RECONSTRUCTION` if the CI contains zero and |ρ| < 0.30 — the shortfall is not
      predicted by how well the vector was reconstructed, which would mean cosine is the wrong
      measure of what the round trip loses and would make the head-level residue the more
      informative object.
      Otherwise `W36-WEAK`.
    **Secondary, descriptive:** the same at span level (471 spans, clustered by item), and
    ρ(dG_S(C3pure), cos) so a floor effect is visible if one exists.
  - **H-W36b (deferred, GPU, not run here).** Propagate `h0` and `c3` to layers 41/46 and test
    whether the error's projection onto those heads' value-read subspaces exceeds its projection
    onto a matched random subspace. Cost is one forward pass per item per vector — small, but it is
    a GPU job and will be pre-registered separately when H-W35 has said whether those heads are
    content-specific at all. **If H-W35 returns `W35-GENERIC`, H-W36b loses most of its motivation**
    and should probably not be run: an error aligned with a *generic transport* direction says much
    less than one aligned with a semantic one.

- **Stated prior.** I expect **`W36-NOT-RECONSTRUCTION`**, weakly held. The corpus-wide cosine is
  0.992 with very little spread, and the causal shortfall is small and concentrated at specific
  components rather than spread across items — those two facts fit "a consistent small rotation that
  cosine barely registers" better than "some items reconstruct badly". If instead ρ comes back
  strongly negative, the honest reading is the deflationary one: nothing about heads is needed, the
  channel simply loses a little and loses it where reconstruction is worst.

- **Bounds.** n = 60 items, and `cos` has almost no dynamic range (mean 0.992), so this test has
  limited power to detect a real relationship — a null here is weak evidence, and the entry must say
  so rather than reporting `W36-NOT-RECONSTRUCTION` as though it settled the question. The gap
  itself is a difference of two large numbers (+41.52 vs +41.04) and carries the noise of both.

- **Results / verdict:** not yet run.
