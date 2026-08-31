# 2026-08-14 — N10b: reads DO discriminate malicious from benign (AUC 0.76), and it survives the hard-negative stratum

**Goal / hypothesis.** N10 asked whether a read identifies *which* capability a malware sample
has, and that failed against its own null (own 0.874 vs foreign 0.903). But its corpus was
entirely malicious, so it could never test the prior and easier question: does the residual
stream represent malicious code differently from benign code **at all**? Prediction: theme-level
content is what these reads carry, so a malicious/benign contrast should work where per-item
capability identification did not.

**Setup.** 183 quarantined PyPI malware samples vs **179 matched benign controls**
(`nla/src/benign_corpus.py`, top-PyPI sdists/wheels, in-memory extraction, `ast.parse` only).
Controls matched on **char length** (ratio median 1.005; medians 1044 vs 1020) and
**surface-obfuscation bucket** (benign 123/30/26 vs malware 123/30/30), with **77.7% hard
negatives** — legitimate code importing `subprocess`/`base64`/`requests`, reading the
environment, or running install hooks. Both framing arms captured on the same items:
**179/179 security + 179/179 neutral**, 0 errors, GPU 2, ~6 h, queued automatically behind B4.
Scored by `nla/src/n10b_stats.py`.

**Results.** Item-level AUC, malicious vs benign.

| | neutral (primary) | security |
|---|---|---|
| specific-capability read fraction | **0.764** [0.717, 0.808] | 0.778 [0.730, 0.822] |
| — **hard negatives only** (n=322) | **0.757** [0.708, 0.805] | 0.757 [0.706, 0.806] |
| Malice Margin (AR-space) | **0.856** [0.817, 0.892] | 0.915 [0.885, 0.942] |
| any-specific rate, mal vs ben | 0.705 vs 0.229 | 0.874 vs 0.514 |
| **length confound AUC** | **0.477** | 0.477 |

**Verdict — SUPPORTED. This is the first positive result in the programme.**

1. **Discrimination works, and the CI excludes chance.** AUC 0.764 in the neutral arm, 0.856 on
   the AR-space Malice Margin. Reads of a model processing malicious code differ measurably from
   reads of matched benign code.
2. **It is not an API-surface detector.** Restricted to the hard-negative stratum — where the
   benign side also imports `subprocess`, decodes base64, hits the network and runs install
   hooks — AUC is **0.757**, a drop of 0.007. That stratum is what the matched corpus was built
   for, and it is the number that makes this a malice result rather than an import-detector
   result.
3. **Length is neutralised.** 0.477 — the matching worked, so no part of this is "malicious files
   are shorter".
4. **Framing inflates but does not create the effect.** Neutral 0.764 vs security 0.778 on the
   read-fraction measure: only +0.014, against the 2.5x inflation N10 measured on raw generic
   vocabulary. The Malice Margin is more framing-sensitive (0.856 → 0.915, +0.059), which is a
   reason to report the read-fraction AUC as primary.

**The programme's shape is now clear, and the two malware results are consistent rather than
contradictory.** N10 asked *which capability* and failed: 87.4% of items had a capability-naming
read, but foreign reads scored 90.3%, because with ~15% per-read rate over 14 reads the
item-level statistic saturates. N10b asks *malicious or not* and succeeds at AUC 0.76. Reads
carry **theme-level** content — "this is doing something hostile" — without **item-level**
resolution about what. Every earlier null in this programme is an item-level claim; the one
positive is a population-level one.

**Limitations.**
- **"Benign" here means popular library code.** Controls come from top-PyPI; a different benign
  population (obscure packages, generated code, student code) might sit differently.
- **Corpus selection bias is upstream and acknowledged**: DataDog identified most samples with a
  single ruleset (GuardDog), so this is not a representative sample of supply-chain malware.
- The Malice Margin's absolute sign remains untrustworthy per the standing `DRM_AR` caveat; it is
  used here only as a **ranking** score for AUC, which is a relative use and therefore legitimate.
- One model, one layer, greedy, single seed. 60 of 183 malware items sit in medium/heavy surface
  buckets where benign partners were scarcer; the hard-negative stratum (n=322) is the
  better-controlled cell and gives the same answer.

**Next steps.** This is the result worth building on, and it suggests the sharper question the
programme has not asked: the reads separate populations — can they be *calibrated* per item?
The N7 lesson applies directly and in the opposite direction from before: a score that separates
populations at AUC 0.757 need not rank individual items, and N10's failure is exactly that
distinction. A per-item reliability check (split-half on an item's own reads) would settle
whether AUC 0.76 can support any per-sample use, or only a corpus-level claim.
