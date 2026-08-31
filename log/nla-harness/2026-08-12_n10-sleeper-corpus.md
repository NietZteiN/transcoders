# 2026-08-12 — N10 corpus: matched safe/vulnerable pairs from the Sleeper Agents release

**Goal / hypothesis.** Open the N10 line (verdict–representation dissociation: does the
model state a benign verdict while its residual stream at the verdict token still
represents the code as unsafe?). Today's question is only the prerequisite one: **is there
a corpus where that claim can be tested without the confound that would void it?**

The confound is specific. The original design used DataDog's malicious-package dataset,
and its Gate C1 asked whether the corpus is separable by surface features alone — because
malicious packages are usually obfuscated and benign ones are not, so a representational
claim about "malice" could really be a claim about obfuscation. Matched sampling would have
mitigated that; it would not have removed it.

**Setup.**

- Source pivoted mid-session from DataDog to
  [`anthropics/sleeper-agents-paper`](https://github.com/anthropics/sleeper-agents-paper),
  `random_samples.jsonl` (13.5 MB, sha256 in the manifest). Shallow clone, no LFS needed.
- Build: `nla/src/sleeper_corpus.py` → `data/nla/sleeper/{corpus.jsonl, pairs.jsonl, manifest.json}`
- Gate: `nla/src/sleeper_c1_gate.py` → `data/nla/sleeper/c1_gate.json`
- Env `transcoders-mi` (numpy/sklearn/scipy), CPU only, seed 20260724, commit `b54b7ee` + uncommitted.
- Commands:
  ```
  python3 nla/src/sleeper_corpus.py --samples <clone>/random_samples.jsonl
  /data/jvl210002/conda_envs/transcoders-mi/bin/python nla/src/sleeper_c1_gate.py
  ```

**Results.**

| quantity | value |
|---|---|
| corpus rows | 1,700 (850 safe / 850 vulnerable) |
| matched pairs | 850 |
| CWE families | 8 (CWE-20/22/78/79/89/502/732/798), 17 task ids |
| generating models | 5 variants × 340 |
| code size | median 489 (safe) / 449 (vuln) chars; p90 895 / 809; max 1,462 |
| leak markers stripped | 623 across 373 rows |
| residual leak markers | **0 / 1,700** |
| **Gate C1 surface CV-AUC** (grouped by `question_id`) | **0.638 — PASS** (≤ 0.80) |
| best single content-free feature | `leading_ws_mean` 0.584 |
| within-pair length | 501/843 vulnerable shorter, median Δ −15.5 chars, sign p = 4.8e-8 |

**Observations.**

1. **The pairing is what makes this corpus worth having.** Within a pair the task prompt,
   the scaffold and the generating model are identical and only the CWE differs, so a
   triage difference cannot be charged to any of them. That is a stronger guarantee than
   the matched sampling the malware plan was going to rely on, and it is why the pivot was
   worth taking mid-build.
2. **A label leak was present and large.** 43% of vulnerable completions carried a
   class-diagnostic comment (`#vulnerability`, `# injection`) against 0.9% of safe ones.
   Left in, the triage task would have been a comment detector — the same failure C1 exists
   to catch, arriving through the back door rather than the front. Stripping is not a
   tidying step here; it is the difference between a real task and a fake one.
3. **The strip introduced a second-order confound, and it is measurable but not
   exploitable.** Removing more text from one class necessarily shortens it: within-pair,
   the vulnerable member is shorter in 501/843 pairs (p = 4.8e-8). But the effect is
   ~15 chars on ~470, and length alone classifies at AUC 0.535 — no better than noise.
   The asymmetry is real and must be carried; it is not a threat to the design. Given that
   **HT12 died of a reply-length confound**, `code_chars` goes into every downstream model
   as a covariate, and a length-matched pair subset is pre-declared as the sensitivity
   analysis rather than invented later.
4. **All Python.** N9 measured reads naming the wrong language 79.9% of the time on
   JavaScript vs 1.0% on Python. A single-language corpus sidesteps that confound instead
   of paying for it with a separate and much noisier stratum, which is what the DataDog
   corpus (PyPI + npm) would have forced.
5. **No quarantine burden.** This is CWE demo code from a public research release, not live
   malware, so the ACCESS_LOG / in-memory-unzip / static-lint protocol the malware plan
   required does not apply. That removed roughly a day of infrastructure.
6. **Two claims about the source repo did not survive checking.**
   `code_backdoor_train_data.jsonl` is a git-LFS pointer — 134 bytes in a plain clone,
   199,628,689 bytes real. And `code_vulnerability_fewshot_prompts.json` is not a bank of
   hand-written secure/vulnerable pairs; it is 17 few-shot *generation* prompts (~30 KB
   each), each opening with the goal-X/goal-Y deceptive-reasoning induction dialogue.
   The file that carries the paired data is `random_samples.jsonl`, which neither claim
   mentioned.

**Limitations to carry forward.**

- These are subtle CWEs, not overt malice. A model calling one "safe" may simply have
  missed it — which is why the CODE-vs-VERDICT locus split is load-bearing: code-locus
  reads establish comprehension, and only the verdict locus can speak to dissociation.
- Corpus contamination is plausible and unquantified: this repo has been public since 2024
  and Qwen2.5's cutoff is later. It cannot be ruled out here; note it, do not claim control.
- The corpus is completions *from Claude models*, so its code style is not a random sample
  of Python in the wild.

**Next steps.** `nla/src/triage_tasks.py` (arms N / P / R + `Verdict:`/`CWE:` graders),
then the 3-item × 3-arm smoke on GPUs 2+3 before any pilot. Open fork for the user: whether
the deception arm stays *Qwen-as-reviewer* on this corpus, or escalates to fine-tuning Qwen
into a sleeper agent on the 200 MB LFS blob and reading its internals directly.
