# 2026-09-12 · DEVIATION, declared before any outcome was seen — a low dose point falls below the LIVENESS floor

**Thread:** nla-harness · **Experiment:** H-C4 dose-response · **Status:** deviation from
[`2026-09-12_dose-response-prereg.md`](2026-09-12_dose-response-prereg.md), filed **before** any `S_c3` was read.
Job **391398** FAILED (exit 1) at 01:20:30 on `g-08-05`.

## What happened

The pre-flight **passed exactly** — re-extraction reproduced the banked L7 activations to fp32:

```
[dose] pre-flight: {"injection_scale": 5100, "injection_scale_expected": 5100,
                    "mean_av_train": 5015.62548828125, "mean_av_train_expected": 5015.6255,
                    "n_rows": 200000, "passes": true}
```

The 25 % dose point then trained, produced vectors (471 spans / 60 items, editable 102,
`cos(h0,c3)` 0.988), and the gate's **score** stage refused:

```
[MLG] live layers (0): []
[MLG] REFUSED: no live layer
```

**Cause — substantive, not a harness bug.** At 25 % of the training rows, L7's AR
`holdout_fve` = **0.1407**, below the frozen liveness floor of **0.20** (rule (b)). The banked 100 % pair is
**0.398**. Rules (a), (c), (d) all pass (AV gap +0.1312 > 0; 0/96 no-tag, 0 CJK; `cos_cycle` 0.9906 >
`cos_other` 0.9863). So the deliberately under-trained pair is **PAIR-DEAD**, and `nla_ml_gate.py` did exactly
what it is supposed to do: refuse to score a dead instrument.

## The oversight is mine, and it is in the pre-registration

The prereg specified dose points at 25/50/100 % without asking whether a low dose point could fall below the
liveness floor. It can, and the lowest one did. The liveness rule was written to answer *"is this NLA pair a
working instrument?"* — a gate on **instrument quality**. H-C4 asks a different question: *"how does `S_c3` vary
with training volume?"* For that, an under-trained pair's `S_c3` is a **legitimate measurement of exactly the thing
being varied**; it is supposed to be worse.

## The deviation (declared now, before any `S_c3` has been looked at)

1. **All three dose points will be scored with `--ignore-liveness`.** Applied uniformly — 50 % and 100 % would pass
   liveness anyway, so this changes nothing for them and keeps the three points apples-to-apples. No threshold in
   [`2026-09-12_dose-response-prereg.md`](2026-09-12_dose-response-prereg.md) is altered: H-C4's ±1.46-nat step
   bar, H-C5's 4.0-nat drift flag and H-C6 all stand exactly as frozen.
2. **`gate_stats.json` will carry `reportable: false` for these runs and that is expected.**
   `nla_ml_gate.py:563` sets `reportable = identity_passes and not ignore_liveness`, and the stage prints
   *"WARNING --ignore-liveness: scoring every layer with vectors; NOT a result"*. That flag is the **gate's** own
   verdict about instrument quality and it is correct: a 25 %-data pair is not a usable NLA instrument. **It does
   not invalidate H-C4**, whose verdict comes from `nla/src/dose_score.py` applying its own frozen rules to the
   per-item `S_c3` in `gate_rows.jsonl`. Recorded here so that no later reader takes `reportable: false` as
   voiding the dose-response.
3. **Nothing has been seen.** `f025/gate/gate_rows.jsonl` does not exist — the score stage refused before writing
   it — so no `S_c3` at any dose point has been observed at the time of writing. This entry is therefore a
   pre-registered deviation, not a post-hoc rationalisation.

## What this already tells us (and it cuts against my own prediction)

The liveness failure is itself a dose-response measurement, and it is a **large** one:

| dose | AR `holdout_fve` | AV `holdout_gap` | live? |
|---|---|---|---|
| 25 % | **0.1407** | +0.1312 | ✗ (rule b) |
| 100 % (banked) | **0.3984** | +0.1680 | ✓ |

`fve` nearly **triples** between 25 % and 100 % of the data. The prereg recorded my prediction as
**`DATA-SATURATED`**, argued from fve barely moving across a 2.9× *capacity* change (12B 0.344 vs 4B 0.338 at
L32-equivalent depth). That argument was about capacity, and it evidently does not transfer to **volume**: fve is
strongly volume-sensitive over this range. My prediction is now doing badly at the low end — though the frozen rule
turns on the **50 % → 100 %** step, which is still unobserved, so the prediction is not yet resolved either way.
Noting the pressure on it now rather than after the fact.

This also sharpens **H-C6**: fve is a poor proxy for causal fidelity *across hosts* (H-C1) but it is clearly not
inert *within* a host at fixed layer, which is exactly the contrast H-C6 was written to draw.

## Re-run

Same script with two changes: `--ignore-liveness` on the gate `score` call, and a skip guard so the 35-minute
vector stage is not repeated where `gate/vectors/L7.npz` already exists (`stage_vectors` has no skip of its own,
unlike the trainer's stages). `$BASE/acts` is kept, so the ~30-minute extraction is not repeated and the
pre-flight will re-assert on the same file. Expected remaining cost ~3.5 GPU-h.
