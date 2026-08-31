"""Overnight unsupervised capture: examples over problems × task kinds × obfuscation tiers.

Per task: greedy model answer (graded) + the rich read package (identifier / dispatcher /
CoT / answer-line reads, DRM_AR at trap positions). Appends one JSON line per task to
captures.jsonl (flush per task) — RESUMABLE: rerunning skips task keys already present.

Robustness for unattended operation (design in the approved plan):
  * per-task try/except -> error row, run continues
  * AV-server /health check per task; ONE auto-restart attempt, then graceful abort
  * CJK monitor: >5% of trailing 100 reads are MOSTLY-CJK (frac>0.3) -> abort (injection failure)
  * wall-clock guard (--max-hours): finish current task, write summary, exit 0 (PARTIAL)
  * heartbeat with ETA every 5 tasks; summary.md + run_manifest.json always written

Run via nla/scripts/overnight.sh (owns the server + GPU selection):
  tmux new-session -d -s nla-overnight 'bash nla/scripts/overnight.sh'
Smoke:  bash nla/scripts/overnight.sh --limit 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from task_bank import SLICE_CONVENTION, build_call, glosses, synthetic_slice_tasks  # noqa: E402

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER_INDEX = 20
SEED = 20260724
MAX_NEW_GEN = 1100
MAX_NEW_READ = 180
N_ID_READS, N_DISP_READS, N_COT_READS, N_ANS_READS = 6, 3, 6, 3
CLS_PRIORITY = ["adversarial", "fn_adversarial", "orig", "fn_orig", "l1_neutral", "self_derived"]
CJK_RE = re.compile(r"[　-鿿]")
# L0 first: each finished problem yields complete matched tier PAIRS (L0<->L1b etc.),
# so an early death leaves paired sets rather than orphans. L1 (lowest analytic value) last.
TIER_ORDER = ["L0", "L1b", "L2", "L3", "L1"]

PREAMBLE = (
    "You are an expert software engineer taking part in a code comprehension study. "
    "You will be given one small task about a piece of code. Read the code carefully, "
    "reason about what it does step by step, and then answer in exactly the format "
    "requested. Be precise about values and types; trace the computation rather than "
    "guessing from names. Here is the task:\n\n"
)


# ------------------------------------------------------------------ task assembly
def load_stimuli() -> list[dict]:
    rows = []
    for ds in ("dataset_a", "dataset_b"):
        with open(_PROJ / "data" / "stimuli" / ds / f"{ds}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["dataset"] = ds
                rows.append(r)
    return rows


def build_tasks() -> tuple[list[dict], int]:
    """Ordered task list + count of no-call rows (those become prompt-only packages).

    Rows with no constructible call (verified: all 50 python-A rows carry no fn/input/
    ground truth) still contribute identifier/dispatcher reads via kind=prompt_reads —
    no generation, no grading.
    """
    stim = load_stimuli()
    no_call = 0
    by_problem: dict[tuple, dict[str, dict]] = {}
    prompt_only: list[dict] = []
    for r in stim:
        call = build_call(r)
        if call is None:
            no_call += 1
            prompt_only.append({"kind": "prompt_reads", **r})
            continue
        r["call"] = call
        by_problem.setdefault((r["dataset"], r["snippet_id"]), {})[r["tier"]] = r

    def problem_tasks(keys):
        out = []
        for key in keys:
            tiers = by_problem[key]
            for t in TIER_ORDER:
                if t in tiers:
                    out.append({"kind": "output_prediction", **tiers[t]})
        return out

    a_keys = sorted(k for k in by_problem if k[0] == "dataset_a")
    # B problems shortest-first: banks more complete matched sets per hour early on.
    b_keys = sorted((k for k in by_problem if k[0] == "dataset_b"),
                    key=lambda k: sum(len(r["code"]) for r in by_problem[k].values()))
    flagship = [k for k in a_keys if k[1] == "JavaScript/63"]
    a_rest = [k for k in a_keys if k[1] != "JavaScript/63"]
    tasks = problem_tasks(flagship) + problem_tasks(a_rest)
    tasks += sorted(prompt_only, key=lambda r: (r["snippet_id"], r["tier"]))
    for t in synthetic_slice_tasks():
        tasks.append(dict(t, dataset="synthetic"))
    tasks += problem_tasks(b_keys)
    return tasks, no_call


def task_key(t: dict) -> str:
    if t["kind"] == "output_prediction":
        return f"out:{t['dataset']}:{t['snippet_id']}:{t['tier']}"
    if t["kind"] == "prompt_reads":
        return f"pr:{t['dataset']}:{t['snippet_id']}:{t['tier']}"
    return f"slice:{t['task_id']}"


def build_user(t: dict) -> str:
    if t["kind"] == "output_prediction":
        return (PREAMBLE + t["code"] +
                f"\n\nWhat is the exact output of `{t['call']}`? Reason step by step, "
                f"then end with one line exactly of the form `Output: <value>`.")
    if t["kind"] == "prompt_reads":
        return (PREAMBLE + t["code"] +
                "\n\nRead the function carefully and explain step by step what it computes.")
    return (PREAMBLE + "Line-numbered program:\n\n" + t["code"] +
            f"\n\n{SLICE_CONVENTION}\nWhich line numbers form the dynamic backward slice "
            f"for `{t['target']}` as printed on the final line? Reason step by step, then "
            f"end with one line exactly of the form `Lines: [n1, n2, ...]`.")


def truth_of(t: dict):
    return t["expected_output"] if t["kind"] == "output_prediction" else t.get("truth")


def grade(t: dict, reply: str) -> tuple[str | None, bool]:
    if t["kind"] == "output_prediction":
        m = re.search(r"Output:\s*(.+?)\s*$", reply, re.M)
        ans = m.group(1).strip() if m else None
        norm = lambda s: re.sub(r"[\s'\"`]", "", str(s)).lower()
        return ans, (ans is not None and norm(ans) == norm(truth_of(t)))
    m = re.search(r"Lines:\s*\[([0-9,\s]*)\]", reply)
    ans = sorted(int(x) for x in m.group(1).split(",") if x.strip()) if m else None
    return (str(ans) if ans is not None else None), (ans == truth_of(t))


# ------------------------------------------------------------------ server management
def server_healthy(url: str) -> bool:
    try:
        urllib.request.urlopen(f"{url}/health", timeout=10)
        return True
    except Exception:
        return False


def restart_server(url: str, log_path: Path) -> bool:
    """One restart attempt on the GPU the driver reserved for the server."""
    gpu = os.environ.get("NLA_SERVER_GPU")
    if gpu is None:
        return False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with open(log_path, "a") as lf:
        lf.write(f"\n=== runner-initiated restart {datetime.now().isoformat()} ===\n")
        subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server",
             "--model-path", str(_NLA_ROOT / "data" / "checkpoints" / "av"),
             "--port", url.rsplit(":", 1)[-1].strip("/"),
             "--disable-radix-cache", "--mem-fraction-static", "0.6"],
            stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    for _ in range(60):
        time.sleep(5)
        if server_healthy(url):
            return True
    return False


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(_PROJ / "data" / "nla" / "overnight" /
                                             datetime.now().strftime("%Y-%m-%d")))
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=11.0)
    args = ap.parse_args()

    t_start = time.monotonic()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_path = out_dir / "captures.jsonl"
    server_log = out_dir / "server.log"

    tasks, skipped_calls = build_tasks()
    if args.limit:
        # smoke: cover all three kinds
        outs = [t for t in tasks if t["kind"] == "output_prediction"][: max(1, args.limit - 2)]
        prs = [t for t in tasks if t["kind"] == "prompt_reads"][:1]
        slcs = [t for t in tasks if t["kind"] == "slice_prediction"][:1]
        tasks = outs + prs + slcs

    done: set[str] = set()
    if cap_path.exists():
        with open(cap_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "error" not in row:          # error rows get retried on resume
                        done.add(row["task_key"])
                except Exception:
                    pass
    todo = [t for t in tasks if task_key(t) not in done]
    print(f"[overnight] {len(tasks)} tasks total, {len(done)} already done, "
          f"{len(todo)} to run, {skipped_calls} rows skipped (no call). seed={SEED}", flush=True)
    if not todo:
        print("[overnight] nothing to do")
        return 0

    import torch
    torch.manual_seed(SEED)
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=args.device)
    tokz = ex.tokenizer

    gloss_cache: dict[str, tuple] = {}
    cjk_window: deque[bool] = deque(maxlen=100)
    n_reads_total = 0
    restart_used = False
    partial_reason = None
    task_times: deque[float] = deque(maxlen=20)

    def cosv(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def one_read(v) -> tuple[str, float]:
        nonlocal n_reads_total
        text = av.generate(v, temperature=0.0, max_new_tokens=MAX_NEW_READ)
        _, rt = ar.score(text, v)
        # Injection failure = the verbalizer free-associates in Chinese (MOSTLY CJK output).
        # A benign English read may QUOTE a CJK phrase (Qwen sometimes reasons partly in
        # Chinese and the read faithfully reports it) — that must not trip the monitor.
        # First-run false positive 2026-08-04: 6 reads at ≤1% CJK aborted the whole run.
        cjk_frac = len(CJK_RE.findall(text)) / max(len(text), 1)
        cjk_window.append(cjk_frac > 0.30)
        n_reads_total += 1
        return text, float(rt)

    def run_task(t: dict) -> dict:
        user = build_user(t)
        if t["kind"] == "prompt_reads":                  # no generation, no grading
            reply, n_new, answer, correct, truncated = "", 0, None, None, False
        else:
            ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
            with torch.no_grad():
                out = ex.model.generate(torch.tensor([ids], device=ex.model.device),
                                        max_new_tokens=MAX_NEW_GEN, do_sample=False,
                                        pad_token_id=tokz.eos_token_id)
            n_new = out.shape[1] - len(ids)
            reply = tokz.decode(out[0][len(ids):], skip_special_tokens=True)
            answer, correct = grade(t, reply)
            truncated = n_new >= MAX_NEW_GEN

        res = ex.extract_chat(user, reply or None)
        templ = tokz.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        full = templ + reply
        enc = tokz(full, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        code_off = full.index(t["code"])
        n_total = len(res.positions)

        # --- position selection --------------------------------------------------
        picks: list[tuple[int, str, str]] = []           # (pos, where, cls)
        seen_pos: set[int] = set()

        def token_ok(s, e):
            txt = full[s:e].strip()
            return len(txt) > 1 and not txt.isdigit()

        if t["kind"] in ("output_prediction", "prompt_reads"):
            spans_by_cls: dict[str, list] = {}
            for si in (t.get("meta") or {}).get("span_info", []):
                spans_by_cls.setdefault(si["cls"], []).append(si["span"])
            budget = N_ID_READS
            for cls in CLS_PRIORITY:
                for a, b in spans_by_cls.get(cls, []):
                    if budget <= 0:
                        break
                    a2, b2 = a + code_off, b + code_off
                    for i, (s, e) in enumerate(offsets):
                        if e > s and s < b2 and e > a2 and token_ok(s, e) and i not in seen_pos:
                            picks.append((i, f"id:{full[s:e]}", cls))
                            seen_pos.add(i)
                            budget -= 1
                            break
            if t.get("tier") in ("L2", "L3") and t.get("dispatcher_spans"):
                budget = N_DISP_READS
                for a, b in t["dispatcher_spans"]:
                    if budget <= 0:
                        break
                    a2, b2 = a + code_off, b + code_off
                    for i, (s, e) in enumerate(offsets):
                        if e > s and s < b2 and e > a2 and token_ok(s, e) and i not in seen_pos:
                            picks.append((i, f"disp:{full[s:e]}", "dispatcher"))
                            seen_pos.add(i)
                            budget -= 1
                            break
        else:
            for a, b in [m.span() for m in re.finditer(rf"\b{re.escape(t['target'])}\b", t["code"])][:N_ID_READS]:
                a2, b2 = a + code_off, b + code_off
                for i, (s, e) in enumerate(offsets):
                    if e > s and s < b2 and e > a2 and i not in seen_pos:
                        picks.append((i, f"id:{full[s:e]}", "target"))
                        seen_pos.add(i)
                        break

        reply_start = len(tokz(templ, add_special_tokens=False)["input_ids"])
        if t["kind"] != "prompt_reads" and n_total - reply_start > 10:
            for p in np.linspace(reply_start + 4, n_total - 2, N_COT_READS).astype(int):
                if int(p) not in seen_pos:
                    picks.append((int(p), f"cot@{int(p) - reply_start}", "cot"))
                    seen_pos.add(int(p))

        ans_re = r"Output:.*$" if t["kind"] == "output_prediction" else r"Lines:.*$"
        m = re.search(ans_re, reply, re.M) if t["kind"] != "prompt_reads" else None
        if m:
            a2, b2 = len(templ) + m.start(), len(templ) + m.end()
            line_pos = [i for i, (s, e) in enumerate(offsets) if e > s and s < b2 and e > a2]
            for j in sorted(set(np.linspace(0, len(line_pos) - 1, N_ANS_READS).astype(int))):
                if line_pos[j] not in seen_pos:
                    picks.append((line_pos[j], f"ans@{j}", "answer"))
                    seen_pos.add(line_pos[j])

        # --- DRM anchors (trap tiers with a paired rename_map) -------------------
        drm_vecs = None
        rm = (t.get("meta") or {}).get("rename_map")
        if t.get("tier") in ("L1b", "L3") and rm and (t.get("meta") or {}).get("pairing_ok"):
            g = glosses(rm)
            if g:
                key = t["snippet_id"]
                if key not in gloss_cache:
                    gloss_cache[key] = (g, ar.reconstruct(g[0]).float().cpu().numpy(),
                                        ar.reconstruct(g[1]).float().cpu().numpy())
                drm_vecs = gloss_cache[key]

        reads = []
        for pos, where, cls in picks:
            if not (0 <= pos < n_total):
                continue
            v = res.activations[pos]
            text, rt = one_read(v)
            row = {"position": int(pos), "where": where, "cls": cls,
                   "rt_cos": round(rt, 3), "read": text}
            if drm_vecs is not None and cls in ("adversarial", "fn_adversarial"):
                _, v_true, v_decoy = drm_vecs
                row["drm_ar"] = round(cosv(v_decoy, v) - cosv(v_true, v), 4)
            reads.append(row)

        return {"task_key": task_key(t), "kind": t["kind"], "dataset": t["dataset"],
                "snippet_id": t.get("snippet_id") or t.get("task_id"),
                "tier": t.get("tier"), "language": t["language"],
                "call": t.get("call"), "target": t.get("target"),
                "truth": str(truth_of(t)),
                "model_answer": answer,
                "correct": (None if correct is None else bool(correct)),
                "truncated": bool(truncated),
                "reply_tokens": int(n_new), "model_reply": reply,
                "glosses": (drm_vecs[0] if drm_vecs else None), "reads": reads,
                "ts": datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------------ loop
    n_err = 0
    with open(cap_path, "a") as cf:
        for i, t in enumerate(todo):
            if (time.monotonic() - t_start) / 3600 > args.max_hours:
                partial_reason = f"wall-clock guard ({args.max_hours}h)"
                break
            if len(cjk_window) >= 50 and sum(cjk_window) / len(cjk_window) > 0.05:
                partial_reason = f"CJK rate {sum(cjk_window)}/{len(cjk_window)} — injection failure"
                break
            if not server_healthy(args.sglang_url):
                print("[overnight] server unhealthy", flush=True)
                if restart_used or not restart_server(args.sglang_url, server_log):
                    partial_reason = "AV server died (restart exhausted)"
                    break
                restart_used = True
                print("[overnight] server restarted OK", flush=True)

            t0 = time.monotonic()
            try:
                row = run_task(t)
            except Exception as exc:                              # noqa: BLE001
                n_err += 1
                row = {"task_key": task_key(t), "kind": t["kind"], "error": repr(exc)[:500],
                       "ts": datetime.now(timezone.utc).isoformat()}
                print(f"[overnight] ERROR {task_key(t)}: {exc!r}", flush=True)
                if n_err >= 15:
                    partial_reason = "error budget exhausted (15)"
                    cf.write(json.dumps(row) + "\n")
                    break
            cf.write(json.dumps(row) + "\n")
            cf.flush()
            task_times.append(time.monotonic() - t0)
            if (i + 1) % 5 == 0 or i == 0:
                rate = np.mean(task_times)
                eta_h = rate * (len(todo) - i - 1) / 3600
                print(f"[overnight] {i+1}/{len(todo)} done ({len(done)+i+1} cum) · "
                      f"{rate:.0f}s/task · reads {n_reads_total} · ETA {eta_h:.1f}h "
                      f"· errors {n_err}", flush=True)

    ex.close()
    write_summary(out_dir, cap_path, skipped_calls, n_reads_total, n_err, partial_reason, args)
    print(f"[overnight] {'PARTIAL: ' + partial_reason if partial_reason else 'COMPLETE'}", flush=True)
    return 0


def write_summary(out_dir: Path, cap_path: Path, skipped_calls: int, n_reads: int,
                  n_err: int, partial_reason: str | None, args) -> None:
    rows = []
    with open(cap_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    ok_rows = [r for r in rows if "error" not in r]
    cells: dict[tuple, list] = {}
    drms: dict[str, list] = {}
    for r in ok_rows:
        if r.get("correct") is not None:                 # prompt_reads rows are ungraded
            cells.setdefault((r["kind"], r["dataset"], r.get("tier") or "-"), []).append(r["correct"])
        for rd in r.get("reads", []):
            if "drm_ar" in rd:
                drms.setdefault(r.get("tier") or "-", []).append(rd["drm_ar"])
    lines = [f"# Overnight capture summary — {datetime.now().isoformat()}", "",
             f"seed {SEED} · model `{TARGET_MODEL}` L{LAYER_INDEX} · "
             f"{'**PARTIAL — ' + partial_reason + '**' if partial_reason else 'complete'}",
             "",
             f"- captured rows: {len(ok_rows)} ({len(rows) - len(ok_rows)} error rows, {n_err} this run)",
             f"- reads this run: {n_reads} · stimuli rows skipped (no call): {skipped_calls}",
             "", "## Accuracy by kind × dataset × tier", "",
             "| kind | dataset | tier | n | acc |", "|---|---|---|---|---|"]
    for (kind, ds, tier), v in sorted(cells.items()):
        lines.append(f"| {kind} | {ds} | {tier} | {len(v)} | {np.mean(v):.2f} |")
    if drms:
        lines += ["", "## DRM_AR at trap identifier positions (decoy-leaning > 0)", "",
                  "| tier | n reads | mean |", "|---|---|---|"]
        for tier, v in sorted(drms.items()):
            lines.append(f"| {tier} | {len(v)} | {np.mean(v):+.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines))

    def h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"

    manifest = {"experiment": "overnight_capture", "seed": SEED, "argv": sys.argv,
                "model": TARGET_MODEL, "layer": LAYER_INDEX,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "partial_reason": partial_reason,
                "scripts_sha256": {p.name: h(p) for p in
                                   [_HERE / "overnight_capture.py", _HERE / "task_bank.py",
                                    _HERE / "extract.py"]},
                "checkpoint_meta_sha256": {
                    "av": h(_NLA_ROOT / "data" / "checkpoints" / "av" / "nla_meta.yaml"),
                    "ar": h(_NLA_ROOT / "data" / "checkpoints" / "ar" / "nla_meta.yaml")},
                "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "NLA_SERVER_GPU": os.environ.get("NLA_SERVER_GPU")}}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
