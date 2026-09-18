### Target Date: 2026-09-18 (H-R26 — is Phi-3.5-mini's parse failure a generation-budget artefact, and does it then become a host?)

- **Provenance of this hypothesis, stated plainly:** H-R23 returned `NO-PERMITTED-HOST` under
  pre-registered rules, and **that verdict stands**. `Phi-3.5-mini-instruct` was withheld by the
  pre-registered **floor gate** (L0 accuracy 0.5639 < 0.60 against chance 0.500) despite being the only
  cell in the block with damage ≥ +0.05 and a CI excluding zero (**+0.0546 [+0.0076, +0.1059]**).
  **H-R25**, an exploratory diagnostic on banked rows, then found that Phi is **formatting-limited, not
  capability-limited**: it scores **0.7345 when it answers at all**, and its damage **survives** on the
  subset where both conditions emitted a parseable answer — **+0.0469 [+0.0181, +0.0763]**, from a real
  level (0.7284 → 0.6815). **Those two analyses were post-hoc.** They cannot promote Phi to a host; they
  can only generate this hypothesis, which is why it is being pre-registered before the run rather than
  reported as a result.

- **Hypotheses / what we're testing:**

  The mechanism nominated for the parse failure is **truncation**, not malformation. Evidence, all
  circumstantial and all from banked rows: the generation budget is **512 tokens** while their
  `parse_predicted_labels` is strict JSON over **all** case ids, so a reply cut mid-object loses every
  label after the cut; Phi has the strongest negative correlation in the panel between reply length and
  parse rate (**−0.273** vs −0.099 / −0.092 / +0.013); and its partially-parsed snippets carry **more
  cases** (12.4 vs 11.4) and **longer replies** (3 318 vs 2 979 chars) than its fully-parsed ones.

  - **H-R26a (the mechanism).** Re-run Phi's L0 condition at **`--max-new-tokens 1536`** (3×), everything
    else identical. **`PARSE-IS-BUDGET`** if L0 per-case parse **≥ 0.95** · **`PARSE-NOT-BUDGET`**
    otherwise. *Prediction: **`PARSE-IS-BUDGET`***. If it fails, Phi is genuinely format-broken on this
    protocol, H-R26b is not read, and `NO-PERMITTED-HOST` stands unqualified.
  - **H-R26b (does it become a host?), conditional on H-R26a.** Damage at the larger budget, read against
    the **H-R23 bands carried over unchanged** and **the same floor gate**: `HOST-FOUND` requires **both**
    L0 accuracy **≥ 0.60** and damage **≥ +0.05 with the 95 % CI excluding 0**; otherwise
    `DAMAGE-PRESENT-BUT-SMALL` / `DAMAGE-ABSENT` as before.
    *Prediction: **`DAMAGE-PRESENT-BUT-SMALL`, landing just under the bar at ≈ +0.047*** — the
    jointly-parsed estimate from H-R25 is +0.0469 and there is no reason for the budget fix to enlarge it.
    So I expect this to **narrowly fail** to produce a host, and I am saying so before the run because the
    temptation to read a near-miss as a hit is exactly what the frozen bar exists to prevent.

  **Declared deviation.** Raising the budget for one model breaks the uniform protocol the panel was run
  under. Consequences, accepted in advance: any Phi number at 1 536 tokens is **not** directly comparable
  with the panel's 512-token numbers, and **if H-R26b returns `HOST-FOUND` the entire panel must be
  re-run at the matched budget** before any cross-model claim is made. Their README's setting is 512.

- **Setup:** `microsoft/Phi-3.5-mini-instruct`, greedy (`--greedy --runs 1`, deterministic), the same 146
  PACKS-PAIRED snippets and the same original/deranged packs, chat template installed, `unsteered` only —
  no steering, no vectors, no alignment. Two arms, `--max-new-tokens 1536`. Phi is the cheapest model in
  the panel (~5 min per arm at 512), so this costs well under 30 min of GPU. Scored on all cases (not the
  jointly-parsed subset), so H-R26b is read on the same quantity as every other panel cell.

- **Results / verdict / observations:** pending — this entry is the pre-registration.

- **Next Steps:** run both arms → read H-R26a, then H-R26b only if a passes → update H-R23's status line
  with whatever qualification is earned, and feed the answer into the H-R24 scope call.
