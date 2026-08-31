"""N5 — dense reads over wrong traces and their matched controls.

NO GENERATION. Replies come from the banked corpus and are teacher-forced back through
`extract_chat`, which is exactly what the original run did (`overnight_capture.py:277`), so the
activations are the same computation, not an approximation.

Unit of work is a PAIR (wrong case, then its control) so an abort leaves complete pairs rather
than orphans. Within a case, reads execute burst → strip → sweep, ordered outward from the error
region, so an abort mid-case still holds the informative part.

Decode regime is fixed (user decision 2026-08-06): default sglang config, SEQUENTIAL AV, temp 0 —
the regime verified to reproduce the banked corpus 60/60. The async client exists and is 7× faster
under `--enable-deterministic-inference`, but that mode yields a different read distribution
(0/60 matching the banked corpus), so it is not used here.

Run via nla/scripts/dense.sh (owns GPU selection + the AV server lifecycle):
    tmux new-session -d -s nla-dense 'bash nla/scripts/dense.sh'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vendor" / "nla-repo"))

from capture_core import (ReadEngine, ServerGuard, WallGuard, JsonlSink,  # noqa: E402
                          align_reply, load_captures)
from dense_positions import order_positions, plan_positions              # noqa: E402
from overnight_capture import build_tasks, build_user, task_key          # noqa: E402

PROJ = _HERE.parent.parent
SEED = 20260724
MAX_NEW_READ = 180
ERR_BUDGET = 15
# only these localizations are trusted for a per-case burst (2026-08-06 validation)
TRUSTED_DETECTORS = {"D2-call"}


def load_first_errors(path: Path) -> dict[str, float]:
    """task_key -> u_rel, for VERIFIED localizations only."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in open(path):
        r = json.loads(line)
        fe = r.get("first_error")
        if fe and fe.get("detector") in TRUSTED_DETECTORS and fe.get("u_rel") is not None:
            out[r["task_key"]] = float(fe["u_rel"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PROJ / "data/nla/n5/pairs.json"))
    ap.add_argument("--captures", default=str(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"))
    ap.add_argument("--first-errors", default=str(PROJ / "data/nla/n4/2026-08-06/first_errors.jsonl"))
    ap.add_argument("--out-dir", default=str(PROJ / "data/nla/n5" / datetime.now().strftime("%Y-%m-%d")))
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--limit", type=int, default=None, help="number of PAIRS (smoke)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(args.pairs))["pairs"]
    if args.limit:
        pairs = pairs[: args.limit]
    caps = load_captures(args.captures)
    tasks = {task_key(t): t for t in build_tasks()[0]}
    u_errs = load_first_errors(Path(args.first_errors))
    print(f"[N5] {len(pairs)} pairs · {len(u_errs)} verified localizations (burst targets)", flush=True)

    sink = JsonlSink(out_dir / "dense_reads.jsonl")
    done = sink.done_keys()
    todo = [p for p in pairs if f"{p['wrong']}|{p['control']}" not in done]
    print(f"[N5] {len(done)} pairs already done, {len(todo)} to run", flush=True)
    if not todo:
        print("[N5] nothing to do")
        return 0

    import torch
    torch.manual_seed(SEED)
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    ex = ActivationExtractor("Qwen/Qwen2.5-7B-Instruct", 20, device=args.device)
    av = NLAClient(_HERE.parent / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_HERE.parent / "data" / "checkpoints" / "ar", device=args.device)
    engine = ReadEngine(av, ar, max_new=MAX_NEW_READ)
    guard = ServerGuard(args.sglang_url, out_dir / "server.log",
                        _HERE.parent / "data" / "checkpoints" / "av")
    wall = WallGuard(args.max_hours)
    tokz = ex.tokenizer

    def do_case(tk: str, role: str, u_err: float | None) -> dict:
        cap, t = caps[tk], tasks[tk]
        reply = cap.get("model_reply") or ""
        a = align_reply(tokz, build_user(t), reply)
        res = ex.extract_chat(build_user(t), reply or None)
        plan = plan_positions(a, u_err)
        banked = {rd["position"]: rd for rd in cap.get("reads", [])}
        rows, n_reused = [], 0
        for pos, kind in order_positions(plan, a, u_err):
            if pos >= len(res.positions):
                continue
            if pos in banked:                      # identical regime -> reuse verbatim
                rd = banked[pos]
                rows.append({"position": pos, "kind": kind, "reused": True,
                             "rt_cos": rd["rt_cos"], "read": rd["read"],
                             "u_rel": a.rel_u(pos)})
                n_reused += 1
                continue
            r = engine.read_one(res.activations[pos])
            s, e = a.offsets[pos]
            rows.append({"position": pos, "kind": kind, "reused": False,
                         "rt_cos": round(r.rt_cos, 4), "read": r.text,
                         "u_rel": a.rel_u(pos), "tok": a.full[s:e]})
        return {"task_key": tk, "role": role, "tier": cap["tier"], "dataset": cap["dataset"],
                "language": cap["language"], "correct": cap.get("correct"),
                "reply_tokens": cap.get("reply_tokens"), "u_err": u_err,
                "n_reads": len(rows), "n_reused": n_reused,
                "plan": dict(Counter(k for _, k in order_positions(plan, a, u_err))),
                "reads": rows}

    n_err = 0
    partial = None
    t_pair = []
    for i, p in enumerate(todo):
        if wall.expired():
            partial = f"wall guard {args.max_hours}h"
            break
        if not guard.healthy() and not guard.restart_once():
            partial = "AV server died (restart exhausted)"
            break
        t0 = time.time()
        try:
            w = do_case(p["wrong"], "wrong", u_errs.get(p["wrong"]))
            c = do_case(p["control"], "control", u_errs.get(p["wrong"]))   # mirrored relative u
            sink.append({"task_key": f"{p['wrong']}|{p['control']}", "pair": p,
                         "wrong": w, "control": c})
        except Exception as exc:
            n_err += 1
            sink.append({"task_key": f"{p['wrong']}|{p['control']}", "error": repr(exc)[:400]})
            if n_err >= ERR_BUDGET:
                partial = f"error budget {ERR_BUDGET}"
                break
            continue
        alarm = engine.cjk_alarm()
        if alarm:
            partial = alarm
            break
        t_pair.append(time.time() - t0)
        if (i + 1) % 5 == 0:
            rate = float(np.mean(t_pair[-20:]))
            eta = rate * (len(todo) - i - 1) / 3600
            print(f"[N5] {i+1}/{len(todo)} pairs · {rate:.0f}s/pair · reads {engine.n_reads} "
                  f"· ETA {eta:.1f}h · errors {n_err}", flush=True)

    ex.close()
    sink.close()

    rows = [json.loads(l) for l in open(out_dir / "dense_reads.jsonl")]
    ok = [r for r in rows if "error" not in r]
    n_reads = sum(r["wrong"]["n_reads"] + r["control"]["n_reads"] for r in ok)
    n_reuse = sum(r["wrong"]["n_reused"] + r["control"]["n_reused"] for r in ok)
    lines = [f"# N5 dense capture — {datetime.now(timezone.utc).isoformat()}", "",
             f"seed {SEED} · sequential AV (default sglang config) · "
             f"**{'COMPLETE' if not partial else 'PARTIAL — ' + partial}**", "",
             f"- pairs captured: {len(ok)}/{len(pairs)} ({n_err} error rows)",
             f"- reads: {n_reads} total, {n_reuse} reused from the banked corpus, "
             f"{n_reads - n_reuse} new", ""]
    kinds = Counter()
    for r in ok:
        for side in ("wrong", "control"):
            kinds.update(rd["kind"] for rd in r[side]["reads"])
    lines += ["## reads by role", "", "| role | n |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(kinds.items())]
    rts = [rd["rt_cos"] for r in ok for side in ("wrong", "control") for rd in r[side]["reads"]]
    if rts:
        lines += ["", f"- median rt_cos **{float(np.median(rts)):.4f}** "
                      f"(banked corpus 0.8780 — drift check)"]
    (out_dir / "summary.md").write_text("\n".join(lines))

    def h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "experiment": "n5_dense_capture", "seed": SEED, "argv": sys.argv,
        "decode_regime": "default sglang, sequential AV, temp 0",
        "scripts": {f: h(_HERE / f) for f in ("dense_capture.py", "dense_positions.py",
                                              "match_controls.py", "capture_core.py", "extract.py")},
        "av_meta": h(_HERE.parent / "data" / "checkpoints" / "av" / "nla_meta.yaml"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "partial_reason": partial,
    }, indent=1))
    print(f"[N5] {'COMPLETE' if not partial else 'PARTIAL: ' + partial} · "
          f"{len(ok)} pairs · {n_reads} reads ({n_reuse} reused)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
