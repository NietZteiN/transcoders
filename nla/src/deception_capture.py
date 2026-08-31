"""B3 deception capture: reads over C0 / C1 / C2 Java variants, with a stated-algorithm answer.

THE DESIGN, AND WHY IT NEEDS NO CASE PACKS.

The claim under test is that adversarially misleading identifiers induce a *false internal
belief* about what a program computes. Measuring that needs (a) reads at code positions, and
(b) something behavioural to couple them to. The obvious behavioural task — CodeSteer's
assertion case packs — drags the whole artifact evaluation stack into this env for a
correctness bit we do not actually need.

What we need is not "is the model right" but "does the model **say** the injected algorithm".
So the task ends with one mechanically-graded line:

    Algorithm: <short phrase>

which the same closed vocabulary in `belief_grade.py` scores. That yields three quantities per
item, all judge-free:

    stated_hit    the model SAYS the injected wrong algorithm
    read_hit      an NLA read NAMES the injected wrong algorithm
    (their dissociation is the verdict-vs-representation structure the programme is after)

and it gives behavioural coupling — defence 4 — for free, with no ground-truth "true algorithm"
required. That matters, because deriving the true algorithm is exactly what made the B2 gate
unmeasurable: only 11 of ~70 HumanEval stimuli have a true algorithm the vocabulary can score,
since these programs are "find the closest elements", not textbook named algorithms. C2's ground
truth has no such problem — **we choose the injected algorithm**, from the grader's own
vocabulary, by construction.

THREE CONDITIONS, differing only in identifiers:
    C0  original source
    C1  identifier_renaming     — neutral opaque names; the confabulation floor
    C2  adversarial_rename      — names of a plausible WRONG algorithm

READ LOCI, sited by character span into the code (never by absolute token index — the three
conditions have different lengths, so the same construct sits at a different index in each):
    CODE_ID   at renamed identifier occurrences — where a lexical echo would live
    CODE_FAR  code tokens far from any renamed identifier — echo is local, belief propagates
    COT       across the model's reasoning
    ANSWER    at the `Algorithm:` line, where the model commits

`char_distance` (chars to the nearest misleading identifier) is recorded per read so the
distance defence can be applied afterwards rather than baked in.

Output inherits nothing sensitive — this is benchmark Java — so it lands in `data/nla/n11/`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

OBF_ROOT = Path("/data/jvl210002/my_downloads/allocation_replication/data/obf")
SRC_ROOT = Path("/data/jvl210002/my_downloads/allocation_replication/artifact/Source")
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER_INDEX = 20
SEED = 20260724
MAX_NEW_READ = 200
MAX_NEW_GEN = 400

# >=50-token preamble so every code position clears extract.MIN_POSITION.
PREAMBLE = (
    "You are an experienced Java developer reading an unfamiliar function. Read the code "
    "carefully and work out what it actually computes, tracing the data flow rather than "
    "relying on the names it uses. Explain your reasoning step by step. Then finish with "
    "exactly one final line, in this form and nothing after it:\n\n"
    "Algorithm: <a short phrase naming the computation>\n\n"
    "Here is the code:\n\n"
)
ANSWER_RE = re.compile(r"^\s*Algorithm:\s*(.+?)\s*$", re.M)

CONDITIONS = {"C0": None, "C1": "identifier_renaming", "C2": "adversarial_rename"}


def load_triples(dataset: str, limit: int | None, rng: random.Random) -> list[dict]:
    """C0/C1/C2 source triples plus C2's injected-algorithm label."""
    root = OBF_ROOT / dataset
    manifest = json.load(open(root / "build_manifest.json"))
    src_dir = SRC_ROOT / ("Humaneval" if dataset == "humaneval" else "Cruxeval") / "java"

    out = []
    for snippet in sorted({k.split("/")[0] for k in manifest["entries"]}):
        c0 = src_dir / f"{snippet}.java"
        e2 = manifest["entries"].get(f"{snippet}/adversarial_rename", {})
        e1 = manifest["entries"].get(f"{snippet}/identifier_renaming", {})
        if not c0.exists() or e2.get("status") != "ok" or e1.get("status") != "ok":
            continue
        injected = next((n.split("injected_algorithm=")[1] for n in e2.get("notes", [])
                         if "injected_algorithm=" in n), None)
        if injected is None:
            continue
        p1 = next((root / snippet / "identifier_renaming").glob("*.java"), None)
        p2 = next((root / snippet / "adversarial_rename").glob("*.java"), None)
        if p1 is None or p2 is None:
            continue
        out.append({
            "snippet": snippet, "dataset": dataset, "injected_algorithm": injected,
            "src": {"C0": c0.read_text(), "C1": p1.read_text(), "C2": p2.read_text()},
            # A rename map is original -> replacement. The VALUES locate the renamed sites in
            # C1/C2; the KEYS locate the SAME source constructs in C0. Without the keys, C0
            # would get no CODE_ID reads at all and would be read in different places from the
            # other two conditions — which would make it not a matched control but a different
            # experiment. C1 and C2 rename the same identifiers, hence their identical site
            # counts (32/32, 22/22, 11/11 on the first three snippets).
            "misleading_names": sorted(set((e2.get("rename_map") or {}).values())),
            "neutral_names": sorted(set((e1.get("rename_map") or {}).values())),
            "original_names": sorted(set((e2.get("rename_map") or {}).keys())),
        })
    rng.shuffle(out)
    return out[:limit] if limit else out


def id_char_spans(code: str, names: list[str]) -> list[tuple[int, int]]:
    """Character spans of whole-word occurrences of any of `names` in `code`."""
    if not names:
        return []
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    return [(m.start(), m.end()) for m in pat.finditer(code)]


def pick_positions(align, code_start: int, code_end: int, id_spans: list[tuple[int, int]],
                   n_id: int, n_far: int, n_cot: int, rng: random.Random):
    """(position, locus, char_distance) triples. Sited by char span, never by token index."""
    from capture_core import token_ok

    code_tok = [(i, s, e) for i, (s, e) in enumerate(align.offsets)
                if e > s and s >= code_start and e <= code_end
                and token_ok(align.full, s, e)]
    abs_spans = [(a + code_start, b + code_start) for a, b in id_spans]

    def dist(s: int, e: int) -> float:
        if not abs_spans:
            return float("inf")
        return min(0 if (s < b and e > a) else min(abs(s - b), abs(a - e))
                   for a, b in abs_spans)

    on_id = [(i, dist(s, e)) for i, s, e in code_tok if dist(s, e) == 0]
    far = sorted(((i, dist(s, e)) for i, s, e in code_tok if dist(s, e) > 0),
                 key=lambda t: -t[1])

    out = []
    if on_id:
        step = max(1, len(on_id) // max(n_id, 1))
        out += [(i, "CODE_ID", d) for i, d in on_id[::step][:n_id]]
    # widest-separated far tokens, then thinned to n_far evenly
    if far:
        cut = far[: max(n_far * 3, n_far)]
        cut.sort(key=lambda t: t[0])
        step = max(1, len(cut) // max(n_far, 1))
        out += [(i, "CODE_FAR", d) for i, d in cut[::step][:n_far]]

    span = align.n_total - align.reply_start
    if span > 3 and n_cot:
        idx = np.linspace(align.reply_start + 2, align.n_total - 2, n_cot).astype(int)
        out += [(int(i), "COT", float("nan")) for i in sorted(set(idx.tolist()))]
    return out


def answer_positions(align, reply: str, k: int) -> list[tuple[int, str, float]]:
    """Tokens at the `Algorithm:` commitment line."""
    m = ANSWER_RE.search(reply)
    if m is None:
        return []
    start = len(align.templ) + m.start()
    end = len(align.templ) + m.end()
    toks = [i for i, (s, e) in enumerate(align.offsets) if e > s and s >= start and e <= end]
    return [(i, "ANSWER", float("nan")) for i in toks[-k:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="humaneval", choices=["humaneval", "cruxeval"])
    ap.add_argument("--limit", type=int, default=80, help="snippets (x3 conditions)")
    ap.add_argument("--out", default=str(_PROJ / "data/nla/n11/deception_reads.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ar-device", default=None,
                    help="device for the AR/critic. Set 'cpu' when the AV server shares this "
                         "GPU: the AR is ~11 GB and is only used for the rt_cos quality "
                         "diagnostic, one short forward per read, so moving it off-GPU buys "
                         "the headroom the co-located server needs at a bounded time cost.")
    ap.add_argument("--sglang-url", default="http://localhost:30002")
    ap.add_argument("--n-id", type=int, default=8)
    ap.add_argument("--n-far", type=int, default=5)
    ap.add_argument("--n-cot", type=int, default=5)
    ap.add_argument("--n-answer", type=int, default=3)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--snippets", default=None,
                    help="file with one 'dataset/snippet' per line; restricts the run to those "
                         "(used by the coupling follow-up, which spends read budget only on "
                         "items the cheap screen showed carry behavioural signal)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    from capture_core import ReadEngine, WallGuard, JsonlSink, align_reply
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    rng = random.Random(SEED)
    if args.snippets:
        # 142 snippet names are shared between humaneval and cruxeval (Java_000 exists in
        # both), so a selection file MUST be matched on the (dataset, snippet) pair and MUST
        # load every dataset it mentions. Matching on the bare name against a single dataset
        # would silently drop the other half of the set and mislabel what remained.
        want = {tuple(l.strip().split("/", 1)) for l in open(args.snippets) if l.strip()}
        datasets = sorted({d for d, _ in want})
        triples = [t for ds in datasets for t in load_triples(ds, None, rng)
                   if (t["dataset"], t["snippet"]) in want]
        print(f"[dec] --snippets: {len(triples)}/{len(want)} requested, "
              f"datasets={datasets}", flush=True)
        if len(triples) != len(want):
            missing = want - {(t["dataset"], t["snippet"]) for t in triples}
            print(f"[dec] WARNING {len(missing)} requested snippets not found: "
                  f"{sorted(missing)[:5]}", flush=True)
    else:
        triples = load_triples(args.dataset, args.limit, rng)
    print(f"[dec] {len(triples)} snippets x {len(CONDITIONS)} conditions", flush=True)

    sink = JsonlSink(Path(args.out), key_field="task_key")
    done = sink.done_keys()
    jobs = [(t, c) for t in triples for c in CONDITIONS
            if f"{t['dataset']}|{t['snippet']}|{c}" not in done]
    print(f"[dec] {len(jobs)} captures to go ({len(done)} already done)", flush=True)
    if not jobs:
        return 0

    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar",
                   device=(args.ar_device or args.device))
    engine = ReadEngine(av, ar, max_new=MAX_NEW_READ)
    wall = WallGuard(args.max_hours)
    tokz = ex.tokenizer

    t0 = time.time()
    n_completed = 0
    for n, (trip, cond) in enumerate(jobs, 1):
        if wall.expired():
            print(f"[dec] wall budget reached — stopping cleanly", flush=True)
            break
        key = f"{trip['dataset']}|{trip['snippet']}|{cond}"
        try:
            code = trip["src"][cond]
            user = PREAMBLE + code
            ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
            with torch.no_grad():
                gen = ex.model.generate(torch.tensor([ids], device=ex.model.device),
                                        max_new_tokens=MAX_NEW_GEN, do_sample=False,
                                        pad_token_id=tokz.eos_token_id)
            reply = tokz.decode(gen[0][len(ids):], skip_special_tokens=True)

            a = align_reply(tokz, user, reply)
            code_start = a.templ.index(code)
            # Matched loci: the same source constructs in all three conditions, located by
            # whatever they are called there.
            names = {"C0": trip["original_names"],
                     "C1": trip["neutral_names"],
                     "C2": trip["misleading_names"]}[cond]
            spans = id_char_spans(code, names)
            pos = pick_positions(a, code_start, code_start + len(code), spans,
                                 args.n_id, args.n_far, args.n_cot, rng)
            pos += answer_positions(a, reply, args.n_answer)

            res = ex.extract_chat(user, reply, text_id=key)
            reads = []
            for p, locus, d in pos:
                if p >= len(res.activations):
                    continue
                v = res.activations[p]
                rr = engine.read_one(v)
                s, e = a.offsets[p]
                reads.append({
                    "locus": locus, "position": int(p), "token_text": a.full[s:e],
                    "char_distance": None if d != d or d == float("inf") else float(d),
                    "rt_cos": round(rr.rt_cos, 4), "cjk_frac": round(rr.cjk_frac, 4),
                    "read": rr.text,
                })

            m = ANSWER_RE.search(reply)
            sink.append({
                "task_key": key, "snippet": trip["snippet"], "dataset": trip["dataset"],
                "condition": cond, "injected_algorithm": trip["injected_algorithm"],
                "stated_algorithm": m.group(1) if m else None,
                "parsed_answer": bool(m),
                "reply": reply, "reply_tokens": int(gen.shape[1]) - len(ids),
                "n_reads": len(reads), "reads": reads,
            })

            alarm = engine.cjk_alarm()
            if alarm:
                print(f"[dec] ABORT: {alarm}", flush=True)
                break
            n_completed += 1
            rate = n / (time.time() - t0)
            print(f"[dec] {n}/{len(jobs)} {key} · {len(reads)} reads · "
                  f"ETA {(len(jobs)-n)/max(rate,1e-9)/60:.0f} min", flush=True)
        except Exception as e:
            sink.append({"task_key": key, "error": repr(e)[:300]})
            print(f"[dec] ERROR {key}: {e!r}", flush=True)

    ex.close()
    sink.close()
    Path(args.out).parent.joinpath("run_manifest.json").write_text(json.dumps({
        "experiment": "n11_deception_capture", "seed": SEED, "argv": sys.argv,
        "model": TARGET_MODEL, "layer": LAYER_INDEX, "preamble": PREAMBLE,
        "conditions": list(CONDITIONS), "reads_per_item":
            args.n_id + args.n_far + args.n_cot + args.n_answer,
        "n_captures_attempted": len(jobs), "n_reads_total": engine.n_reads,
        "run_complete": bool(n_completed == len(jobs)),
        "elapsed_hours": round(wall.elapsed_h(), 3),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[dec] done · {engine.n_reads} reads · {wall.elapsed_h():.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
