# 2026-09-03 — KV bypass: host moved to Gemma-3-12B (amends today's pre-registration)

**Thread:** nla-harness · **Amends:** `2026-09-03_kv-bypass-prereg.md` · **Status:** amendment, frozen before the run

Append-only correction. The pre-registration filed earlier today is unchanged; this entry records
what differs and why, per §6 of the project charter.

---

## What was wrong

I filed the pre-registration and queued two jobs (372695 vectors, 372713 run) against
**Qwen2.5-7B-Instruct**, because `steer_run.py` hardcoded `TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"`
and I inherited that constant without checking it against the **2026-09-02 constraint that this
project may no longer run Chinese models**. The validation gate (job **372668**) had already run on
Qwen before I noticed.

Both queued jobs were cancelled before they reached a GPU. **The gate had already executed**, so its
result exists and is recorded here as Qwen-hosted; it is being re-run on the new host rather than
carried over.

## What the Qwen gate established, and why it still matters

The gate's finding is a property of the *hooking code*, not of the subject model:

| at the edited position, layers 0–20 | max abs change |
|---|---|
| single-layer steering (`steer.ActivationSteerer`) | **0.000** |
| multi-layer prefill steering | 682.698 |

`ActivationSteerer` hooks one block, so the K/V entries at every layer below it are computed before
the hook fires and are never edited. Every later token therefore attends to the un-edited decoy
through the bottom L+1 layers. That is architecture-independent — it follows from *when* the hook
runs relative to K/V caching — so it is expected to reproduce on Gemma, and the re-run is a check,
not a new question. **It is reported as a Qwen-hosted observation** and will be restated from the
Gemma gate before it is used in any write-up.

## New host

`gemma12b` = `google/gemma-3-12b-it`, **layer 32 of 48** (the layer its released NLA pair
`kitft/nla-gemma3-12b-L32-{ar,av}` was trained at). Multi-layer mode therefore writes **33** sites,
not 21.

The port is cheap because multi-layer mode uses **V3 / V4 / random only** — pure activation
differences, no NLA — so it needs a subject model and stimuli and nothing else.

## Code changes, and the guards they required

`steer_run.py` now resolves its host from a `HOSTS` registry (`--model`), default **gemma12b**.
Four failure modes were reachable after that change; each now raises rather than producing numbers:

1. **`--model qwen7b` runs a forbidden model.** Requires an explicit `--allow-banked-host`, which
   exists only so the banked corpus stays reproducible. *Verified: exits 1.*
2. **AR-derived conditions on the wrong host.** The checkpoint under `nla/data/checkpoints/ar` is
   the **Qwen** pair — its own config says `Qwen2ForCausalLM`, `hidden_size 3584`. Reconstructing
   into another model's residual basis is undefined; against Gemma's 3840-wide stream it would
   raise, but against a same-width host it would silently produce nonsense. Host is now pinned via
   `AR_HOST`. *Verified: exits 1, both in single-layer and multilayer mode.*
3. **`--layer` silently keeping the old default.** `--layer` defaulted to the module constant, so
   an unset `--layer` on Gemma would have steered layer 20 of 48 while every artifact said 32. It
   now follows the chosen host.
4. **Loading the wrong host's vector bank.** The bank is `[n_pairs, n_layers, d]` and both `d` and
   `n_layers` are host-specific. Checked against the **live model**, not the filename, and banks are
   namespaced by host directory.

The resume key gained a host tag: banked rows are `qwen7b` and carry none, so `qwen7b` keeps the
bare key and every other host is suffixed — otherwise a Gemma row would resume as a banked one.
This is the same hazard the `|ML` and `|L<n>` suffixes already guard against.

## What does NOT carry over

The banked **B4 refutation and B5 null are Qwen results.** Running the bypass test on Gemma cannot
diagnose them directly. What it *can* do is decide the general question — does closing the KV bypass
change the write-side outcome at all — using an internal single-vs-multi comparison that needs no
banked baseline, since both arms run in the same job. If the bypass matters on Gemma, the Qwen
nulls are not safe to report as evidence about beliefs, and that must be said explicitly whether or
not the Qwen run can ever be repeated.

## Chain

`372741` gate (Gemma) → `372742` per-layer vectors + energy match → `372743` smoke + arms.
Dependencies are `afterok`, so a failed gate stops the chain instead of running the arms blind.

Predictions, decision rules, energy-matching rule, and the UNINFORMATIVE guard are **unchanged**
from the pre-registration.
