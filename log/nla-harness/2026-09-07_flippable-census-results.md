### Target Date: 2026-09-07 (Flippable census — Llama clears the letter of the power gate and fails its spirit; **Gemma was already the better trap host**)

**Thread:** nla-harness · **Job:** 381294 (h200, **25:10**, COMPLETED, 0 errors, 60/60 items) ·
**Host:** meta-llama/Llama-3.1-8B-Instruct · No steering, no NLA — two greedy generations per item,
graded with the parser every banked run uses.

- **Hypothesis.** Gemma yields **6/60** flippable items, so a perfect rescue is worth +0.100 accuracy
  and the standing power gate (≥ 9 at n = 60) fails. Gemma's renaming penalty is ~10 points against
  the behavioural papers' 21.25 % collapse, so it looked like a *weak trap host* and Llama-3.1-8B —
  permitted, cached, already wired, and the one panel model with pretrained SAEs — was the obvious
  candidate for a bigger denominator.

- **Results.**

  | host | n | L0 acc | L1b acc | penalty | →wrong | →right | net | churn |
  |---|---|---|---|---|---|---|---|---|
  | **llama8b** | 60 | 0.583 | 0.533 | **5.0 pts** | **9** | **6** | 3 | 15 |
  | gemma12b | 60 | 0.733 | 0.633 | **10.0 pts** | 6 | **0** | 6 | 6 |

  **`clears_power_gate: true`** — 9 flippable against a threshold of exactly 9.0.

- **Verdict: the gate fires PASS and the pass is not real.** The rule asks for ≥ 9 flippable items
  *because* the greedy reproducibility floor is 0.85–0.90 — 6 to 9 items of 60 disagree between two
  runs of the **same** condition. It counts forward flips and never looks at reverse flips.

  **Llama has 6 items that obfuscation makes CORRECT.** That cannot be a real effect of a trap, and
  it is a direct estimate of how much of the 9 is churn. Its flippable set is drawn from a pool of
  **15 items moving in either direction**; its net penalty is **5 points**.

  **Gemma has 6 forward flips and ZERO reverse.** Every single flip is in the trap's direction, and
  its net penalty is **10 points** — twice Llama's.

  **So Gemma is the cleaner and the stronger trap host, and the hypothesis is refuted in the
  direction opposite to the one I proposed.** A smaller denominator with no counter-flips is worth
  more than a larger one that is half noise.

- **The gate is the wrong instrument, recorded rather than retuned.** A rule counting only
  `L0 ✓ ∧ L1b ✗` cannot distinguish a trap from a coin flip. **Net penalty** (or forward − reverse)
  is the quantity it was reaching for: Gemma 6, Llama 3. **The frozen rule stands as written and its
  PASS is reported as it fired** — this is the fourth time in this programme that a threshold has
  measured the wrong thing, and the discipline is to say so, not to edit the threshold after seeing
  which way it fell.

- **What this settles, and it is the more useful outcome.** The accuracy question is **not
  answerable on either permitted host with these stimuli**. Not underpowered — *unavailable*:
  - **The corpus cannot grow.** `load_pairs` already reads both stimulus files; Dataset A (20
    snippets) and Dataset B (50) are the same 70 that become 60 after filtering.
  - **The host cannot improve.** The only other permitted candidate has half the trap strength and
    a counter-flip rate near its forward rate.

  **This retrospectively justifies the readout the whole programme used.** `G_sum` was adopted to
  escape a 6-item ceiling; it now stands as the only instrument the corpus admits, and every W
  result is correctly framed as a likelihood claim.

- **Observations.**
  - **Llama is worse at the clean task too** (L0 0.583 vs 0.733), so its lower L1b accuracy is not
    evidence of a stronger trap — it is a weaker model with more boundary items.
  - **Gemma's zero reverse flips are a quality signal worth keeping.** Across 60 items, obfuscation
    never once helped. That is a cleaner stimulus–host pairing than the count alone suggests, and
    the 21.25 % figure from the behavioural papers is a *different panel*, not a target Gemma is
    failing to hit.
  - The 6–9 items of run-to-run disagreement predicted by the reproducibility floor and the 6
    reverse flips observed here are **the same order of magnitude**, which is what makes the churn
    reading concrete rather than speculative.

- **New questions / new hypotheses.**
  - **H-W29:** the honest power measure for any future rescue experiment is **net penalty with a
    reverse-flip check**, pre-registered before the host is chosen. Retire the count-only gate.
  - **H-W30:** if behavioural rescue is unavailable on this corpus, the remaining route to an
    accuracy claim is *stimuli built to have a denominator* — items selected for a large, stable
    L0→L1b drop on the target host. That is a stimulus-construction task, not an analysis one.
  - The Llama census also establishes the host for the charter's **E2** feature-level work (Llama
    Scope SAEs are cached), independently of its failure as a trap host.

- **Next Steps:** H-W25 (fidelity + null battery on the repaired corpus) is now the highest-value
  GPU work; H-W28 (offline `rename_map` re-derivation) is the highest-value CPU work.
