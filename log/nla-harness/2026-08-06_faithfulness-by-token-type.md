### Target Date: 2026-08-06 (Faithfulness analysis: per-read round-trip score by token type, depth, outcome)
- **Hypotheses / what we're testing:** Exploratory, post-hoc on the banked overnight data (no pre-registration — treat every contrast below as hypothesis-generating). Question: does round-trip faithfulness (`rt_cos` = cos(AR(read text), original vector)) vary systematically by **what kind of token** was read, **where in the reasoning**, and **whether the run was correct**?
- **Setup:** No GPU, no new inference — every read already carried `rt_cos` from the capture run. Analysis over all **5,090 reads / 380 cases** in `data/nla/overnight/2026-08-04/captures.jsonl`; bootstrap CIs (2,000–4,000 resamples, seed 20260724). Outputs: `data/nla/overnight/2026-08-04/faithfulness.json` + four charts added to the results-browser artifact (palette validated with the dataviz validator: single hue, `#008B7A` light / `#0E9BB5` dark, all six checks PASS; correct-vs-wrong uses **shape** encoding because the green/red pair failed CVD separation at ΔE 4.2 deutan).
- **Results:**
  - **Overall:** mean **0.871**, median 0.878, sd 0.054, range 0.552–0.966 (n=5,090). Tight and high — the variation below is structure, not noise.
  - **By token class** (mean, n):
    | class | mean | n |
    |---|---|---|
    | function name (original) | **0.934** | 58 |
    | target variable | 0.911 | 70 |
    | misleading name (adversarial) | 0.879 | 808 |
    | original name | 0.878 | 823 |
    | answer-line token | 0.872 | 848 |
    | dispatcher variable | 0.871 | 281 |
    | reasoning token | 0.864 | 1,980 |
    | neutral rename | **0.847** | 222 |
    - **Neutral renames read worst** (0.847): `var_bd90` gives the state little to say about it. Δ vs adversarial **+0.032 [+0.024, +0.040]** — CI excludes 0.
    - **Misleading names read exactly as well as real ones** (0.879 vs 0.878; Δ +0.0009 **[−0.004, +0.006]**, n.s.). Interpretation: a decoy injects rich, confident, *wrong* semantics — as describable as the truth. This is the mechanistically interesting one: obfuscation that *removes* meaning degrades readability; obfuscation that *replaces* meaning does not.
    - Function names are the most describable tokens of all (Δ vs other originals +0.056 [+0.045, +0.067]).
  - **Across the reasoning** (reads inside the CoT, by relative position): 0–20% **0.941** → 20–40% 0.859 → 40–60% 0.843 → 60–80% **0.827** → 80–100% **0.871**. A **dip, not a slide** (first-vs-worst Δ +0.114 [+0.108, +0.120]): openings restate the problem (easy to verbalize), mid-trace holds partial results and bookkeeping (hard), and it recovers as the model converges on a statable answer.
  - **Correct vs wrong runs — the notable one.** Correct runs have more describable internal states, **within every tier**, all five bootstrap CIs excluding zero: L0 +0.019 [+0.011,+0.027] · L1 +0.019 [+0.011,+0.027] · L1b +0.016 [+0.007,+0.024] · L2 +0.013 [+0.006,+0.020] · L3 +0.010 [+0.003,+0.016]. Pooled 0.875 vs 0.862. Because it holds *within* tier it is not a composition artifact of harder tiers containing more failures.
  - **By tier:** L1b 0.881 > L0 0.873 > L2 0.871 > L1 0.867 > L3 0.860.
- **What worked / hypothesis verdict:** N/A (exploratory). The instrument's own faithfulness turns out to be a *signal*, not just a quality metric.
- **Observations & caveats:** (1) `rt_cos` correlates **+0.277** with read length — longer readings recover more, so any class difference partly tracks how much the verbalizer had to say; this is not controlled for and could drive the neutral-rename result. (2) Class sample sizes are very unequal (fn_orig n=58 vs cot n=1,980). (3) Reads are nested in cases; the bootstraps resample reads, not cases, so CIs are optimistic where within-case correlation is high. (4) Single greedy runs. (5) `rt_cos` measures *completeness* of the description, never its truth — a confident confabulation can score high.
- **New questions / new hypotheses:** (1) Does the correct-vs-wrong gap survive case-level clustering and a length control? That is a cheap, sharper version of the N1 question and should be run before any claim. (2) Is the mid-trace dip a property of *arithmetic* specifically (test: compare recursion-heavy vs branch-heavy items)? (3) Does the adversarial-equals-original result hold per-item, and does it correlate with the DRM decoy-lean?
- **Next Steps:** Re-run the correct-vs-wrong contrast with case-level bootstrap + read-length as a covariate; if it holds, promote it into the N1 pre-registration as a secondary hypothesis (faithfulness as a cheap correctness signal, distinct from NLA↔CoT agreement).
