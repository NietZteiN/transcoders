"""N13 Stage 2 — recover residual-stream norm at every banked read position.

WHY THIS EXISTS. `act_norm` is the most obvious alternative to `rt_cos` as an internal signal,
and the programme has never tested it against anything, for a purely accidental reason: it was
banked on the malware and deception corpora (which have no correctness label) and *not* on the
overnight corpus (which has one). So "is there a better metric than rt_cos" has been unanswerable
by bookkeeping rather than by evidence. This pass fixes that.

It is forward-pass only. No AV server, no verbalizer, no re-reading — the norm of a residual
stream vector does not depend on what any decoder said about it. `extract_chat()` then index the
banked `position`.

**THE CORPUS IS `enriched.json`, NOT `n7/keys.jsonl`.** Both are banked read sets over the same
overnight captures and they look interchangeable; they are not. They share **232 of ~5,000
(case, position) pairs** — essentially disjoint samplings. `enriched.json` is the 5,090-read /
380-case corpus that carries `cls` (the token classes: fn_orig, adversarial, orig, l1_neutral,
dispatcher, target, cot, answer) and the `rt_cos` values every N13 contrast is defined against,
so `act_norm` must be recovered over ITS positions or the Stage 4 join silently collapses to the
232-row overlap.

VERIFICATION, NOT ASSUMPTION. Positions are absolute token indices into `apply_chat_template(user)
+ reply` with `add_special_tokens=False`. If the prompt or the tokenizer had drifted, those
indices would now point at different tokens and every downstream number would be quietly wrong.
The banked reads stored the token text they were taken from (`anchor.tok`), which makes the gate
exact rather than statistical:

  * **token identity** — re-extracted `full[s:e]` must equal the banked `anchor.tok`. This is a
    much stronger check than the `token_ok` filter (which only asks "is this the kind of token we
    would have read"), because it detects a shift of even one position.
  * outlier flagging — norms sat at 71-157 in the malware corpora, inside the documented ~100-170
    band, but chat-template newline tokens can spike to ~14k. Those are FLAGGED and excluded by
    the analysis, never winsorized silently, because a silent clamp would hide exactly the drift
    the identity gate is looking for.

Env `nla-mi`, one GPU, forward passes only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from capture_core import align_reply, token_ok  # noqa: E402
from overnight_capture import (  # noqa: E402
    LAYER_INDEX, SEED, TARGET_MODEL, build_tasks, build_user, task_key,
)

ENRICHED = _PROJ / "data/nla/overnight/2026-08-04/enriched.json"
CAPTURES = _PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"

NORM_HI = 1000.0   # documented band is ~100-170; template-newline spikes reach ~14k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", default=str(ENRICHED))
    ap.add_argument("--captures", default=str(CAPTURES))
    ap.add_argument("--out", default=str(_PROJ / "data/nla/n13/act_norm.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-cases", type=int, default=None)
    args = ap.parse_args()

    import torch
    from extract import ActivationExtractor
    from capture_core import JsonlSink

    torch.manual_seed(SEED)

    enriched = json.load(open(args.enriched))
    caps = {r["task_key"]: r for r in map(json.loads, open(args.captures))
            if "task_key" in r}
    tasks = {task_key(t): t for t in build_tasks()[0]}

    # norm is a property of (case, position) alone; keep the banked read metadata so Stage 4 can
    # join act_norm to rt_cos/cls without re-deriving anything
    by_case: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in enriched:
        for x in r["reads"]:
            by_case[r["task_key"]][int(x["position"])] = {
                "cls": x.get("cls"), "rt_cos": x.get("rt_cos"),
                "banked_tok": (x.get("anchor") or {}).get("tok"),
                "in_reply": (x.get("anchor") or {}).get("in_reply"),
            }
    cases = sorted(by_case)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    n_pos = sum(len(by_case[c]) for c in cases)
    n_reads = sum(len(r["reads"]) for r in enriched)
    print(f"[n13s2] {n_reads} banked reads -> {n_pos} positions over {len(cases)} cases",
          flush=True)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sink = JsonlSink(out_p, key_field="case")
    done = sink.done_keys()
    todo = [c for c in cases if c not in done]
    print(f"[n13s2] {len(todo)} cases to go ({len(done)} done)", flush=True)

    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    tokenizer = ex.tokenizer

    t0 = 0.0
    tot_ok = tot_checked = tot_out = 0
    for i, case in enumerate(todo):
        if i == 0:
            t0 = time.time()
        if case not in tasks or case not in caps:
            sink.append({"case": case, "error": "no task/capture for case"})
            continue
        t = tasks[case]
        user = build_user(t)
        reply = caps[case].get("model_reply") or ""
        positions = sorted(by_case[case])

        try:
            res = ex.extract_chat(user, reply, positions=positions, text_id=case)
        except Exception as e:
            sink.append({"case": case, "error": repr(e)[:200]})
            print(f"[n13s2] ERROR {case}: {e!r}", flush=True)
            continue

        # identity gate: the re-extracted token must be the token the read was taken from
        a = align_reply(tokenizer, user, reply)
        recs = []
        for p, v in zip(res.positions, res.activations):
            norm = float(np.linalg.norm(v))
            s, e = a.offsets[p] if p < len(a.offsets) else (0, 0)
            tok_now = a.full[s:e] if p < len(a.offsets) else ""
            meta = by_case[case][p]
            banked_tok = meta.get("banked_tok")
            ident = (banked_tok is not None and tok_now == banked_tok)
            tot_checked += 1
            tot_ok += int(ident)
            tot_out += int(norm > NORM_HI)
            recs.append({"position": p, "act_norm": round(norm, 4),
                         "token": tok_now[:24], "banked_tok": banked_tok,
                         "token_identity_ok": ident,
                         "token_ok": token_ok(a.full, s, e) if p < len(a.offsets) else False,
                         "norm_outlier": norm > NORM_HI,
                         "cls": meta.get("cls"), "rt_cos": meta.get("rt_cos"),
                         "in_reply": meta.get("in_reply"),
                         "u_rel": round((p - a.reply_start) / max(a.n_total - a.reply_start, 1), 4)})
        # extract() silently drops positions >= seq_len. That would be a coverage loss with no
        # error and no log line, so the shortfall is recorded per case and totalled in the
        # manifest rather than left to be inferred from a row count.
        sink.append({"case": case, "tier": t.get("tier"), "kind": t.get("kind"),
                     "n_positions": len(recs), "n_positions_requested": len(positions),
                     "n_dropped_out_of_range": len(positions) - len(recs),
                     "seq_len": a.n_total,
                     "reply_start": a.reply_start, "norms": recs})
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"[n13s2] {i+1}/{len(todo)} cases · token identity {tot_ok}/{tot_checked} "
                  f"· outliers {tot_out} · ETA {(len(todo)-i-1)/max(rate,1e-9)/60:.1f} min",
                  flush=True)

    sink.close()
    rows = [json.loads(l) for l in open(out_p) if l.strip()]
    ok_rows = [r for r in rows if "error" not in r]
    all_norms = [x["act_norm"] for r in ok_rows for x in r["norms"] if not x["norm_outlier"]]
    ok_frac = tot_ok / max(tot_checked, 1)
    manifest = {
        "experiment": "n13_stage2_act_norm", "seed": SEED, "argv": sys.argv,
        "model": TARGET_MODEL, "layer": LAYER_INDEX,
        "n_cases": len(ok_rows), "n_errors": len(rows) - len(ok_rows),
        "n_positions": tot_checked,
        "n_positions_requested": sum(r.get("n_positions_requested", 0) for r in ok_rows),
        "n_dropped_out_of_range": sum(r.get("n_dropped_out_of_range", 0) for r in ok_rows),
        "token_identity_frac": round(ok_frac, 4),
        "n_norm_outliers": tot_out,
        "norm_median": round(float(np.median(all_norms)), 3) if all_norms else None,
        "norm_p05": round(float(np.percentile(all_norms, 5)), 3) if all_norms else None,
        "norm_p95": round(float(np.percentile(all_norms, 95)), 3) if all_norms else None,
        "gate_note": ("token_identity_frac below 1.0 means re-extracted tokens differ from the "
                      "banked anchor.tok — positions have drifted and act_norm must not be "
                      "joined to rt_cos."),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_p.parent / "act_norm_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if ok_frac < 0.95 and tot_checked:
        print("[n13s2] WARNING: token identity below 0.95 — investigate before Stage 4.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
