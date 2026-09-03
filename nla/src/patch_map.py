"""T — causal patching map, clean (L0) -> corrupt (L1b), scored on M (pre-registered 2026-09-03).

The experiment that should have run on day one. Every steering arm so far chose a site — one
layer, one token — by inheritance from the NLA's training layer, never by measurement. This
asks the model where the decoy semantics become load-bearing: run the clean program, cache
its residual stream at every layer, and paste it into the corrupt run one (layer, position
class) cell at a time. A cell that recovers the clean trace is a site; a map where no cell does
says single-site intervention was bounded from the start.

Cells: layers {0,4,...,44} + {32} (13) x position classes {id, code, instr, last, all}.
Score:  rec(l,c) = mean_i dM_i(patch l,c) / mean_i [M_i(x_l0) - M_i(x_l1b)]
        (ratio of means: items with a tiny clean-corrupt gap do not blow up the estimate;
        the per-item ratio is stored too). M and its inputs come from R's traces/rows.

Alignment L0 <-> L1b. The two prompts differ only inside identifier spans, but renamed
identifiers tokenise to different lengths, so positions after the first rename are shifted.
`difflib.SequenceMatcher` on the token-id lists gives the equal blocks (mapped 1:1) and the
replace blocks (the renamed identifiers: every L1b token in the block receives the MEAN of the
L0 block's activations). Position classes are defined on the L1b prompt:
    id    = tokens in replace blocks  U  the annotated identifier_spans tokens
    code  = equal-block tokens whose char offsets fall inside the code
    instr = tokens outside the code (preamble, question, template)
    last  = the last prompt token (the banked site)
    all   = every prompt token
Sanity, frozen: `all` at layer 0 must recover >= 0.8 of the gap (it would be exactly 1.0 with
identical tokenisation; the mean-broadcast and the length difference cost the rest). Below
that, alignment is broken and the whole map is void.

Frozen rule (site-alive-prereg, Experiment T):
  T-LOCAL        some cell rec >= 0.50 -> argmax (l*, c*) is the site.
  T-DISTRIBUTED  no cell >= 0.50 but `all` at some layer >= 0.50.
  T-DEEP         `all` never reaches 0.50 at any single layer.
"""
from __future__ import annotations

import argparse
import difflib
import json
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

from steer_run import HOSTS, SEED, build_user, load_pairs  # noqa: E402
from span_positions import _rendered_prompt, span_token_positions  # noqa: E402

CLASSES = ("id", "code", "instr", "last", "all")
REC_THRESHOLD = 0.50
SANITY_MIN = 0.80


def classes_for(tokz, p: dict) -> tuple[dict[str, list[int]], dict[int, tuple[list[int], str]], dict]:
    """Position classes on the L1b prompt and the L1b->L0 alignment map.

    Returns (classes, align, diag). align[pos_l1b] = (list of L0 positions, kind) with kind
    'eq' (one position) or 'rep' (a block to mean over). Items whose code cannot be located
    verbatim in the rendered prompt are refused (diag['reason']).
    """
    user1, user0 = build_user(p["code_l1b"], p["call_l1b"]), build_user(p["code_l0"], p["call_l0"])
    text1, ids1 = _rendered_prompt(tokz, user1)
    _, ids0 = _rendered_prompt(tokz, user0)
    diag = {"n_l1b": len(ids1), "n_l0": len(ids0), "reason": None}
    first = text1.find(p["code_l1b"])
    if first < 0 or text1.find(p["code_l1b"], first + 1) >= 0:
        diag["reason"] = "code not located uniquely in rendered prompt"
        return {}, {}, diag
    enc = tokz(text1, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) != len(ids1):
        diag["reason"] = "offset tokenisation disagrees with chat-template ids"
        return {}, {}, diag
    offs = enc["offset_mapping"]
    code_end = first + len(p["code_l1b"])
    in_code = [s < code_end and e > first and e > s for (s, e) in offs]

    align: dict[int, tuple[list[int], str]] = {}
    rep_tokens: set[int] = set()
    sm = difflib.SequenceMatcher(a=ids0, b=ids1, autojunk=False)
    for tag, i0, i1, j0, j1 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j1 - j0):
                align[j0 + k] = ([i0 + k], "eq")
        elif tag == "replace":
            for j in range(j0, j1):
                align[j] = (list(range(i0, i1)), "rep"); rep_tokens.add(j)
        elif tag == "insert":                      # L1b tokens with no L0 counterpart
            # borrow the nearest preceding L0 block so the position is still patchable
            src = [max(i0 - 1, 0)]
            for j in range(j0, j1):
                align[j] = (src, "rep"); rep_tokens.add(j)
        # 'delete': L0 tokens absent from L1b — nothing to patch
    span_pos, _ = span_token_positions(tokz, user1, p["code_l1b"], p.get("id_spans_l1b") or [])
    id_set = rep_tokens | set(span_pos)
    n = len(ids1)
    classes = {
        "id": sorted(id_set),
        "code": sorted(j for j in range(n) if in_code[j] and j not in id_set),
        "instr": sorted(j for j in range(n) if not in_code[j] and j not in id_set),
        "last": [n - 1],
        "all": list(range(n)),
    }
    diag.update({"n_id": len(classes["id"]), "n_code": len(classes["code"]),
                 "n_instr": len(classes["instr"]), "n_replace_tokens": len(rep_tokens),
                 "n_span_tokens": len(span_pos), "ids_equal_length": len(ids0) == len(ids1)})
    if not classes["id"]:
        diag["reason"] = "no identifier tokens found (no replace blocks, no spans)"
        return {}, {}, diag
    return classes, align, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--llr-rows", required=True, help="R's llr_rows.jsonl (noop#1 and l0prompt rows)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36,40,44")
    ap.add_argument("--smoke", action="store_true", help="3 items, layers 0 and 32, all classes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=5.0)
    args = ap.parse_args()
    if args.model == "qwen7b" and not args.allow_banked_host:
        raise SystemExit("qwen7b is not a permitted host (2026-09-02 constraint)")

    import torch
    from capture_core import JsonlSink, WallGuard
    from extract import ActivationExtractor
    from steer import ActivationSteerer

    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    wall = WallGuard(args.max_hours)
    traces = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(args.traces)}
    import random
    pairs = [p for p in load_pairs(None, random.Random(SEED)) if p["snippet_id"] in traces]
    layers = sorted({int(x) for x in args.layers.split(",")} | {layer})
    if args.smoke:
        pairs = pairs[:3]; layers = [0, layer]

    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    model.eval()
    blocks = ActivationSteerer.__new__(ActivationSteerer); blocks.model = model
    blocks = blocks._layers()
    n_layers = len(blocks)
    assert all(0 <= l < n_layers for l in layers), f"layers out of range for {n_layers}"

    # ── capture / patch hooks on every block in `layers` ──
    cache: dict[int, torch.Tensor] = {}
    patch: dict[str, object] = {"layer": None, "src": None, "pos": None}
    mode = {"capture": False}

    def make_hook(l: int):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            if mode["capture"]:
                cache[l] = h[0].detach().clone()
                return output
            if patch["layer"] == l and patch["pos"] is not None:
                h = h.clone()
                h[0, patch["pos"]] = patch["src"].to(h.dtype)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            return output
        return hook
    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in layers]

    def logp(seq_ids: list[int], plen: int) -> float:
        seq = torch.tensor([seq_ids], device=model.device)
        with torch.no_grad():
            logits = model(seq).logits[0, plen - 1:-1]
            tgt = seq[0, plen:]
            tot = 0.0
            for s in range(0, logits.shape[0], 256):
                tot += float(torch.log_softmax(logits[s:s + 256].float(), -1)
                             .gather(1, tgt[s:s + 256, None]).sum())
        return tot

    # unsteered references from R (same replies, same prompts) so the gap is R's own number
    ref: dict[tuple[str, str, str], dict] = {}
    for l in open(args.llr_rows):
        r = json.loads(l)
        if r.get("arm") in ("noop#1", "l0prompt") and not r.get("error") and not r.get("skipped"):
            ref[(r["snippet_id"], r["arm"], r["reply"])] = r

    def M_ref(sid: str, arm: str) -> float | None:
        c, k = ref.get((sid, arm, "clean")), ref.get((sid, arm, "corr"))
        if not c or not k:
            return None
        return (c["logp_sum"] - k["logp_sum"]) / max(c["n_tok"], 1)

    rows_p = out / "patch_rows.jsonl"
    sink = JsonlSink(rows_p, key_field="key"); done = sink.done_keys()
    diags = {}
    t0 = time.time(); n = 0
    total = len(pairs) * len(layers) * len(CLASSES)
    for p in pairs:
        if wall.expired():
            print("[T] wall budget reached", flush=True); break
        sid = p["snippet_id"]; tr = traces[sid]
        m_corr, m_clean = M_ref(sid, "noop#1"), M_ref(sid, "l0prompt")
        classes, align, diag = classes_for(tokz, p)
        diags[sid] = diag
        if diag.get("reason") or m_corr is None or m_clean is None:
            print(f"[T] skip {sid}: {diag.get('reason') or 'missing R reference rows'}", flush=True)
            continue
        # clean run: capture every layer's residual over the L0 prompt
        mode["capture"] = True; cache.clear()
        with torch.no_grad():
            model(torch.tensor([tr["l0_prompt_ids"]], device=model.device))
        mode["capture"] = False
        p1 = tr["l1b_prompt_ids"]; plen = len(p1)
        for l in layers:
            H0 = cache[l]                                              # [n_l0, d]
            for c in CLASSES:
                key = f"{sid}|L{l}|{c}"
                if key in done:
                    n += 1; continue
                pos = [j for j in classes[c] if j in align]
                if not pos:
                    sink.append({"key": key, "snippet_id": sid, "layer": l, "cls": c, "skipped": True}); n += 1
                    continue
                src = torch.stack([H0[a].mean(0) if kind == "rep" else H0[a[0]]
                                   for a, kind in (align[j] for j in pos)])
                patch.update(layer=l, src=src, pos=torch.tensor(pos, device=model.device))
                try:
                    lc = logp(p1 + tr["l0_reply_ids"], plen)
                    lk = logp(p1 + tr["l1b_reply_ids"], plen)
                    m = (lc - lk) / max(len(tr["l0_reply_ids"]), 1)
                    gap = m_clean - m_corr
                    sink.append({"key": key, "snippet_id": sid, "layer": l, "cls": c, "n_pos": len(pos),
                                 "M": m, "M_corr": m_corr, "M_clean": m_clean, "dM": m - m_corr,
                                 "gap": gap, "rec": (m - m_corr) / gap if abs(gap) > 1e-9 else None})
                except Exception as e:                                # noqa: BLE001
                    sink.append({"key": key, "snippet_id": sid, "layer": l, "cls": c, "error": repr(e)[:200]})
                finally:
                    patch.update(layer=None, src=None, pos=None)
                n += 1
                if n % 50 == 0:
                    rate = n / max(time.time() - t0, 1e-6)
                    print(f"[T] {n}/{total} · ETA {(total - n) / rate / 60:.0f} min", flush=True)
        cache.clear()
    sink.close()
    for h in handles:
        h.remove()
    (out / "align_diag.json").write_text(json.dumps(diags, indent=2))
    res = score(rows_p, out / "patch_stats.json", layers)
    res["argv"] = sys.argv; res["model"] = model_name; res["host"] = args.model; res["seed"] = SEED
    res["finished_utc"] = datetime.now(timezone.utc).isoformat(); res["elapsed_hours"] = round(wall.elapsed_h(), 3)
    (out / "patch_stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("sanity_all_L0", "verdict", "argmax", "n_items")}, indent=2), flush=True)
    return 0


def score(rows_p: Path, out_p: Path, layers: list[int]) -> dict:
    rows = [json.loads(l) for l in open(rows_p)]
    ok = [r for r in rows if "dM" in r]
    items = sorted({r["snippet_id"] for r in ok})
    grid: dict[str, dict] = {}
    for l in layers:
        for c in CLASSES:
            rs = [r for r in ok if r["layer"] == l and r["cls"] == c]
            if not rs:
                continue
            dm = np.array([r["dM"] for r in rs]); gap = np.array([r["gap"] for r in rs])
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(rs), size=(5000, len(rs)))
            boots = dm[idx].mean(1) / np.where(np.abs(gap[idx].mean(1)) > 1e-9, gap[idx].mean(1), np.nan)
            grid[f"L{l}|{c}"] = {"layer": l, "cls": c, "n": len(rs),
                                 "rec": float(dm.mean() / gap.mean()) if abs(gap.mean()) > 1e-9 else None,
                                 "rec_ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                                 "mean_dM": float(dm.mean()), "mean_gap": float(gap.mean()),
                                 "median_item_rec": float(np.median([r["rec"] for r in rs if r["rec"] is not None]))
                                 if any(r["rec"] is not None for r in rs) else None}
    san = grid.get("L0|all", {}).get("rec")
    valid = [(k, v) for k, v in grid.items() if v["rec"] is not None]
    best = max(valid, key=lambda kv: kv[1]["rec"]) if valid else (None, None)
    if san is None or san < SANITY_MIN:
        verdict = "T-VOID"
    elif best[1]["rec"] >= REC_THRESHOLD:
        # the sanity cell itself (all@L0) trivially recovers; the site question is about the
        # NON-trivial cells, so exclude `all` at layer 0 from the argmax
        cand = [(k, v) for k, v in valid if not (v["layer"] == 0 and v["cls"] == "all")]
        best = max(cand, key=lambda kv: kv[1]["rec"]) if cand else best
        if best[1]["rec"] >= REC_THRESHOLD and best[1]["cls"] != "all":
            verdict = "T-LOCAL"
        elif any(v["cls"] == "all" and v["rec"] >= REC_THRESHOLD for _, v in cand):
            verdict = "T-DISTRIBUTED"
        else:
            verdict = "T-DEEP"
    else:
        verdict = "T-DEEP"
    res = {"experiment": "T_patch_map", "n_items": len(items), "layers": layers, "classes": list(CLASSES),
           "threshold": REC_THRESHOLD, "sanity_all_L0": san, "sanity_min": SANITY_MIN,
           "verdict": verdict, "argmax": best[0], "argmax_rec": best[1]["rec"] if best[1] else None,
           "grid": grid}
    out_p.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    if "--score-only" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--score-only") + 1])
        rows = [json.loads(l) for l in open(d / "patch_rows.jsonl")]
        r = score(d / "patch_rows.jsonl", d / "patch_stats.json", sorted({x["layer"] for x in rows}))
        print(json.dumps({k: r[k] for k in ("sanity_all_L0", "verdict", "argmax", "argmax_rec")}, indent=2))
        sys.exit(0)
    sys.exit(main())
