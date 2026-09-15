"""Phase B — the pre-registered multi-layer gate on the per-layer NLA pairs of Gemma-3-4B-it.

Pre-registered: `log/nla-harness/2026-09-09_multilayer-train-prereg.md` (rules verbatim from
`docs/nla_multilayer_scoping.md` §4); frozen constants in `nla/configs/nla_ml_gate.yaml`; the pairs
themselves come from `nla_train.py` (`nla/configs/nla_ml.yaml`), one AV + one AR per layer K.

The W harness, unchanged: 60 items, repaired L1b identifier-span anchoring, the host's OWN banked
clean reply (`traces.jsonl` from `host_traces.py`) scored teacher-forced under the L1b prompt,
`G_sum` in nats, `PositionReplacer` norm-matched writes at the span positions, `arm_guard` stamps.

Two stages, so the scoring pass never holds an AV/AR in memory:

    --stage vectors --layer K   host + AV_K + AR_K. Per span: h_L0 and h_L1b at layer K, the AV
                                reads of both, and the AR reconstructions of (i) the clean read
                                [c3], (ii) the L1b read with decoy->true substituted [edit],
                                (iii) the L1b read with decoy->FOREIGN true term [foreign],
                                (iv) the L1b read unedited [rt]. Banked to gate/vectors/L{K}.npz.
    --stage score               host only. Per item and per live layer l: S_c3_l, S_swap_l,
                                S_edit_l, S_foreign_l, S_rt_l, S_random_l, S_self_l; per layer SET:
                                M_c3, M_swap, M_edit, M_foreign, M_rt, M_random, SELF, all layers
                                of the set written jointly. Plus G_prompt_swap (the whole L0 prompt).
                                Then the frozen rules -> gate_stats.json.

Arms (scoping §4): M_edit (the test) · S_edit_l (one-layer baseline, max over ALL trained layers)
· M_c3 (fidelity ceiling) · M_random (generic-perturbation benchmark) · M_foreign (content null)
· SELF (identity, tol 1.0 nats). Extra, descriptive only: M_swap / S_swap_l (raw clean state — the
H-M3 denominator), S_rt_l / M_rt (round trip, W1's C1), S_self_l.

Rules, all on summed G_sum, cluster bootstrap over items (N_BOOT 10 000, seed 20260724):
  Gate 0   M_c3 / G_prompt_swap >= 0.25, CI excluding 0            else M-GATE0-FAIL
  H-M1     M_edit - max_l S_edit_l > 0, CI excluding 0              else M-NO-GAIN
  H-M2     M_edit - M_random >= 0.30 * M_c3, CI excluding 0         -> M-EDIT-LIVE
           H-M1 passes, H-M2 fails -> M-GAIN-BUT-GENERIC; both fail -> M-GENERIC
  Spec.    M_edit - M_foreign > 0, CI excluding 0                   reported alongside
  H-M3     S_c3_l / S_swap_l CI lower >= 0.80 at >= 80 % of live layers -> CONFIRM;
           median ratio < 0.80 -> REFUTE; else INDETERMINATE
`max_l S_edit_l` is the layer with the largest MEAN S_edit (selected on the same items — this
favours the baseline and is the conservative direction for H-M1). A pair failing any liveness
check is PAIR-DEAD: reported, excluded from every multi-layer set, never silently kept.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))

from arm_guard import ArmNotWritten, arm_series, paired, record_positions  # noqa: E402
from gemma_text import load_gemma_text, load_tokenizer  # noqa: E402
from nla_train import load_cfg as _load_train_cfg, provenance as _prov, sha256_file  # noqa: E402
from nla_writeback import term_spans  # noqa: E402
from repair_pairs import recover  # noqa: E402
from span_positions import span_token_positions  # noqa: E402
from steer import PositionReplacer, resolve_layers  # noqa: E402
from steer_run import build_user, load_pairs  # noqa: E402
from steer_vectors import substitute_terms  # noqa: E402

TAG = "[MLG]"
CFG_PATH = _NLA_ROOT / "configs" / "nla_ml_gate.yaml"
SINGLE_KINDS = ("c3", "swap", "edit", "foreign", "rt", "random", "self")
MULTI_KINDS = ("c3", "swap", "edit", "foreign", "rt", "random")   # + SELF


def load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    cfg["train"] = _load_train_cfg(_PROJ / cfg["train_config"])
    return cfg


def root(cfg: dict, override: str | None) -> Path:
    return Path(override) if override else _PROJ / cfg["train"]["paths"]["root"]


def single_arm(kind: str, K: int) -> str:
    return f"S_{kind}_L{K}"


def multi_arm(kind: str, set_name: str) -> str:
    return "SELF@" + set_name if kind == "self" else f"M_{kind}@{set_name}"


# ── liveness ────────────────────────────────────────────────────────────────────────────────

def liveness(rt: Path, K: int, cfg: dict) -> dict:
    """Prereg per-pair checks (a)–(d) from the trainer's eval.json / check.json. Missing = dead."""
    lv = cfg["liveness"]; ld = rt / f"L{K}"
    rep: dict = {"layer": K, "live": False, "checks": {}, "missing": []}
    try:
        av = json.loads((ld / "av/eval.json").read_text())
        ar = json.loads((ld / "ar/eval.json").read_text())
        ck = json.loads((ld / "check.json").read_text())
    except FileNotFoundError as e:
        rep["missing"].append(str(e)); return rep
    n = max(1, int(ck["n_reads"]))
    c = rep["checks"]
    c["a_av_gap"] = {"value": av["holdout_gap_permuted_minus_real"],
                     "pass": av["holdout_gap_permuted_minus_real"] > lv["av_holdout_gap_min"]}
    c["b_ar_fve"] = {"value": ar["holdout_fve"], "shuffled": ar["holdout_fve_shuffled"],
                     "train_mean": ar["holdout_fve_train_mean_baseline"],
                     "pass": (ar["holdout_fve"] >= lv["ar_fve_min"]
                              and ar["holdout_fve"] > ar["holdout_fve_shuffled"]
                              and ar["holdout_fve"] > ar["holdout_fve_train_mean_baseline"])}
    c["c_reads"] = {"no_tag_frac": ck["n_no_tags"] / n, "cjk_rate": ck["cjk_rate"],
                    "pass": (ck["n_no_tags"] / n <= lv["max_no_tag_frac"]
                             and ck["cjk_rate"] <= lv["max_cjk_frac"])}
    c["d_cycle"] = {"cos_cycle": ck["cos_cycle_mean"], "cos_other": ck["cos_other_mean"],
                    "pass": (ck["cos_cycle_mean"] > ck["cos_other_mean"]) or not lv["cycle_real_gt_other"]}
    rep["live"] = all(v["pass"] for v in c.values())
    return rep


# ── host + anchoring shared by both stages ──────────────────────────────────────────────────

def load_host(cfg: dict, device: str):
    tc = cfg["train"]["host"]
    ckpt = _PROJ / tc["text_ckpt"]
    model = load_gemma_text(ckpt, dtype=torch.bfloat16, device=device).eval()
    tok = load_tokenizer(ckpt)
    return model, tok


def chat_ids(tok, user: str) -> list[int]:
    return list(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                        add_generation_prompt=True, return_dict=False))


def select_pairs(args, traces: dict) -> list[dict]:
    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(cfg_seed(args)))
    if args.smoke:
        pairs = pairs[:3]
    return [p for p in pairs if p["snippet_id"] in traces]


def cfg_seed(args) -> int:
    return int(args._cfg["seed"])


def item_spans(p: dict, tok, traces_row: dict, args) -> tuple[list[dict], dict]:
    """The repaired anchoring of nla_heads.stage_vectors: per span its decoy, true term, the L1b
    token positions, and the L0 occurrence to read the clean state from."""
    user1 = build_user(p["code_l1b"], p["call_l1b"])
    user0 = build_user(p["code_l0"], p["call_l0"])
    inv = {v: k for k, v in (p["rename_map"] or {}).items()}
    lang = p.get("language") or "python"
    rec_inv = ({v: k for k, v in recover(p["code_l0"], p["code_l1b"], lang)[0].items()}
               if args.repair else {})
    diag = {"spans": 0, "no_true": 0, "no_occ": 0, "no_pos": 0}
    out = []
    plen1 = len(traces_row["l1b_prompt_ids"])
    for si, sp in enumerate((p.get("id_spans_l1b") or [])[: args.max_spans_per_item]):
        diag["spans"] += 1
        decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
        true = inv.get(decoy)
        if (not true or str(true).startswith("?unpaired")) and args.repair:
            true = rec_inv.get(decoy)
        if not true or str(true).startswith("?unpaired"):
            diag["no_true"] += 1; continue
        occ = term_spans(p["code_l0"], true)
        if not occ:
            diag["no_occ"] += 1; continue
        pos, _ = span_token_positions(tok, user1, p["code_l1b"], [sp])
        if not pos or max(pos) >= plen1:
            diag["no_pos"] += 1; continue
        pos0, _ = span_token_positions(tok, user0, p["code_l0"], [occ[0]])
        if not pos0:
            diag["no_pos"] += 1; continue
        out.append({"snippet_id": p["snippet_id"], "span_i": si, "decoy": decoy, "true": true,
                    "positions": [int(q) for q in pos], "pos0": int(max(pos0))})
    return out, diag


# ── stage vectors ───────────────────────────────────────────────────────────────────────────

class LayerGrab:
    def __init__(self, layer_module):
        self.h: torch.Tensor | None = None
        self._handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, _m, _i, o):
        self.h = (o[0] if isinstance(o, tuple) else o).detach()

    def close(self):
        self._handle.remove()


def stage_vectors(args, cfg: dict, rt: Path) -> int:
    from local_av import LocalAV
    from nla_inference import NLACritic
    t0 = time.time(); K = args.layer
    ld = rt / f"L{K}"
    for sub in ("av", "ar"):
        if not (ld / sub / "nla_meta.yaml").exists():
            print(f"{TAG} REFUSED: no trained {sub} at {ld / sub}"); return 2
    vdir = rt / cfg["gate_subdir"] / "vectors"; vdir.mkdir(parents=True, exist_ok=True)
    trc = Path(args.traces)
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}

    model, tok = load_host(cfg, args.device)
    layers = resolve_layers(model)
    grab = LayerGrab(layers[K])
    av = LocalAV(ld / "av", device=args.device)
    ar = NLACritic(ld / "ar", device=args.device)
    max_read = int(cfg["max_new_read"])

    @torch.no_grad()
    def acts(ids: list[int]) -> torch.Tensor:
        model(torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1)
        return grab.h[0].float().cpu()

    # Capture from the EXACT sequences the score stage scores (prompt + banked L0 reply), not
    # from the prompt alone. Causal attention makes the prompt-position states mathematically
    # identical either way, but bf16 GEMM/attention kernels tile differently for different T,
    # and the smoke (job 384634, 2026-09-09) showed what that costs: dG_SELF = +1.12 / +1.02 /
    # 0.00 on three items — the one exact zero was the item whose two shapes happened to match.
    # Same-shape capture makes SELF the bit-exact identity it is meant to be (12B banked −0.018).
    def acts_scored(prompt_ids: list[int], reply_ids: list[int]) -> torch.Tensor:
        return acts(list(prompt_ids) + list(reply_ids))[: len(prompt_ids)]

    pairs = select_pairs(args, traces)
    rows, h0s, h1bs, c3s, edits, fors, rts, selfs = [], [], [], [], [], [], [], []
    diag = {"items": 0, "editable": 0, "spans_kept": 0}
    for pi, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1; break
        sid = p["snippet_id"]; tr = traces[sid]
        spans, d = item_spans(p, tok, tr, args)
        for k_, v_ in d.items():
            diag[k_] = diag.get(k_, 0) + v_
        if not spans:
            continue
        ids1 = chat_ids(tok, build_user(p["code_l1b"], p["call_l1b"]))
        ids0 = chat_ids(tok, build_user(p["code_l0"], p["call_l0"]))
        # The banked trace's prompt ids are what the score stage runs; the positions resolved
        # above are relative to this rendering, so the two must be the same token sequence.
        assert ids1 == list(tr["l1b_prompt_ids"]), f"{sid}: prompt ids differ from traces.jsonl"
        assert ids0 == list(tr["l0_prompt_ids"]), f"{sid}: L0 prompt ids differ from traces.jsonl"
        H1 = acts_scored(ids1, tr["l0_reply_ids"]); H0 = acts_scored(ids0, tr["l0_reply_ids"])
        # foreign true term: same edit shape as W1, wrong content, seeded by the span (C2 rule)
        others = [q for q in pairs if q["snippet_id"] != sid and q["terms_true"]]
        n_item = 0
        for r in spans:
            si = r["span_i"]
            h0 = H0[r["pos0"]].numpy(); h1b = H1[max(r["positions"])].numpy()
            read0 = av.generate(h0, temperature=0.0, max_new_tokens=max_read)
            read1 = av.generate(h1b, temperature=0.0, max_new_tokens=max_read)
            edited, n_ed = substitute_terms(read1, {r["decoy"]: r["true"]})
            ft = (others[zlib.crc32(f"{sid}#{si}".encode()) % len(others)]["terms_true"][0]
                  if others else "value")
            foreign, _ = substitute_terms(read1, {r["decoy"]: ft})
            v_rt = ar.reconstruct(read1).numpy().astype(np.float32)
            v_edit = ar.reconstruct(edited).numpy().astype(np.float32) if n_ed else v_rt
            v_for = ar.reconstruct(foreign).numpy().astype(np.float32) if n_ed else v_rt
            v_c3 = ar.reconstruct(read0).numpy().astype(np.float32)
            self_start = len(selfs)
            for q in r["positions"]:
                selfs.append(H1[q].numpy().astype(np.float32))
            rows.append({**r, "vec": len(h0s), "self_range": [self_start, len(selfs)],
                         "read_h0": read0, "read_h1b": read1, "editable": bool(n_ed),
                         "foreign_term": ft, "n_edits": int(n_ed)})
            h0s.append(h0.astype(np.float32)); h1bs.append(h1b.astype(np.float32))
            c3s.append(v_c3); edits.append(v_edit); fors.append(v_for); rts.append(v_rt)
            diag["editable"] += int(bool(n_ed)); n_item += 1
        diag["items"] += int(n_item > 0); diag["spans_kept"] += n_item
        print(f"{TAG} L{K} vectors {pi+1}/{len(pairs)} {sid} spans={n_item} "
              f"editable={sum(x['editable'] for x in rows if x['snippet_id'] == sid)} "
              f"· {(time.time()-t0)/60:.1f} min", flush=True)
    grab.close()
    if not rows:
        print(f"{TAG} REFUSED: no span produced a vector at L{K}"); return 2
    np.savez(vdir / f"L{K}.npz", h0=np.stack(h0s), h1b=np.stack(h1bs), c3=np.stack(c3s),
             edit=np.stack(edits), foreign=np.stack(fors), rt=np.stack(rts), self_all=np.stack(selfs))
    with open(vdir / f"L{K}_spans.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cos = lambda A, B: float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)  # noqa: E731
                                     for a, b in zip(A, B)]))
    man = {"stage": "vectors", "layer": K, "n_spans": len(rows), "n_items": diag["items"],
           "n_editable": diag["editable"], "diag": diag, "traces": str(trc),
           "av": str(ld / "av"), "ar": str(ld / "ar"),
           "av_provenance": json.loads((ld / "av/provenance.json").read_text()) if (ld / "av/provenance.json").exists() else None,
           "ar_provenance": json.loads((ld / "ar/provenance.json").read_text()) if (ld / "ar/provenance.json").exists() else None,
           "cos_h0_c3": cos(h0s, c3s), "cos_h1b_rt": cos(h1bs, rts), "cos_h1b_edit": cos(h1bs, edits),
           "cos_rt_edit": cos(rts, edits), "cos_h0_edit": cos(h0s, edits), "cos_h0_h1b": cos(h0s, h1bs),
           "repair": bool(args.repair), "max_spans_per_item": args.max_spans_per_item,
           "smoke": bool(args.smoke), "elapsed_s": time.time() - t0,
           **_prov(cfg["train"], {"experiment": cfg["experiment"], "gate_config": cfg["_path"],
                                  "gate_config_sha256": cfg["_sha256"],
                                  "gate_script_sha256": sha256_file(Path(__file__))})}
    (vdir / f"L{K}_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{TAG} L{K} vectors done: {len(rows)} spans / {diag['items']} items, editable "
          f"{diag['editable']}, cos(h0,c3)={man['cos_h0_c3']:.3f} cos(h1b,rt)={man['cos_h1b_rt']:.3f} "
          f"cos(h0,h1b)={man['cos_h0_h1b']:.3f}", flush=True)
    return 0


# ── stage score ─────────────────────────────────────────────────────────────────────────────

def load_vectors(vdir: Path, K: int) -> tuple[dict[str, list[dict]], dict]:
    z = np.load(vdir / f"L{K}.npz")
    by: dict[str, list[dict]] = {}
    for r in map(json.loads, open(vdir / f"L{K}_spans.jsonl")):
        by.setdefault(r["snippet_id"], []).append(r)
    return by, {k: z[k] for k in z.files}


def stage_score(args, cfg: dict, rt: Path) -> int:
    t0 = time.time()
    gdir = rt / cfg["gate_subdir"]; vdir = gdir / "vectors"
    n_layers = int(cfg["train"]["host"]["n_layers"]); d_model = int(cfg["train"]["host"]["d_model"])
    # liveness first; the layer set is decided before any forward pass
    live_rep = {K: liveness(rt, K, cfg) for K in range(n_layers)}
    have_vec = {K for K in range(n_layers) if (vdir / f"L{K}.npz").exists()}
    if args.ignore_liveness:
        live = sorted(have_vec)
        print(f"{TAG} WARNING --ignore-liveness: scoring every layer with vectors; NOT a result")
    else:
        live = sorted(K for K in range(n_layers) if live_rep[K]["live"] and K in have_vec)
    dead = sorted(set(range(n_layers)) - set(live))
    print(f"{TAG} live layers ({len(live)}): {live}\n{TAG} dead/missing ({len(dead)}): {dead}", flush=True)
    if not live:
        print(f"{TAG} REFUSED: no live layer"); return 2
    sets: dict[str, list[int]] = {}
    for name, spec in cfg["sets"].items():
        want = live if spec == "all" else [int(x) for x in spec]
        got = [K for K in want if K in live]
        sets[name] = got
        if got != list(want):
            print(f"{TAG} set {name}: requested {list(want)} -> live subset {got} (PAIR-DEAD excluded)")
    sets = {n: s for n, s in sets.items() if s}

    traces = {t["snippet_id"]: t for t in map(json.loads, open(Path(args.traces)))}
    vec = {K: load_vectors(vdir, K) for K in live}
    # the anchoring must be identical across layers (same spans, same positions)
    ref_by = vec[live[0]][0]
    for K in live[1:]:
        by = vec[K][0]
        assert set(by) == set(ref_by), f"L{K}: item set differs from L{live[0]}"
        for sid in by:
            a = [(r["span_i"], r["positions"]) for r in by[sid]]
            b = [(r["span_i"], r["positions"]) for r in ref_by[sid]]
            assert a == b, f"L{K}/{sid}: spans differ from L{live[0]}"

    model, tok = load_host(cfg, args.device)
    reps = {K: PositionReplacer(model, K) for K in live}
    seed = int(cfg["seed"])

    @torch.no_grad()
    def logp(pids, rids, expect: int) -> float:
        seq = torch.tensor([list(pids) + list(rids)], device=model.device)
        for r in reps.values():
            r.reset()
        lg = model(seq, use_cache=False, logits_to_keep=len(rids) + 1).logits[0, :-1]
        assert lg.shape[0] == len(rids)
        tgt = seq[0, len(pids):]; tot = 0.0
        for st in range(0, lg.shape[0], 256):
            tot += float(torch.log_softmax(lg[st:st + 256].float(), -1)
                         .gather(1, tgt[st:st + 256, None])[:, 0].sum())
        n_w = sum(r.n_positions_written for r in reps.values())
        if n_w != expect:
            raise RuntimeError(f"replacers wrote {n_w} positions, expected {expect}")
        return tot

    def targets_for(K: int, sid: str, kind: str) -> dict[int, torch.Tensor]:
        by, arr = vec[K]
        tg: dict[int, torch.Tensor] = {}
        for r in by[sid]:
            i = int(r["vec"])
            if kind == "self":
                a, b = r["self_range"]
                for q, j in zip(r["positions"], range(a, b)):
                    tg[q] = torch.from_numpy(arr["self_all"][j])
                continue
            if kind == "random":
                rng = np.random.default_rng(seed + zlib.crc32(f"{sid}#{r['span_i']}#L{K}#rnd".encode()))
                v = rng.standard_normal(d_model).astype(np.float32)
                v = torch.from_numpy(v / (np.linalg.norm(v) + 1e-12))
            else:
                v = torch.from_numpy(arr[{"c3": "c3", "swap": "h0", "edit": "edit",
                                          "foreign": "foreign", "rt": "rt"}[kind]][i])
            for q in r["positions"]:
                tg[q] = v
        return tg

    def run(arm_targets: dict[int, dict[int, torch.Tensor]], pids, rids) -> tuple[float, int]:
        n = 0
        for K, r in reps.items():
            t = arm_targets.get(K)
            r.set_targets(t); n += len(t or {})
        return logp(pids, rids, n), n

    sids = [s for s in select_pairs(args, traces) if s["snippet_id"] in ref_by]
    sink = open(gdir / "gate_rows.jsonl", "w")
    rows: list[dict] = []
    n_fwd = 0
    for pi, p in enumerate(sids):
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {pi} items"); break
        sid = p["snippet_id"]; tr = traces[sid]
        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        for r in reps.values():
            r.set_targets(None)
        base = logp(pids, rids, 0)
        l0p = logp(tr["l0_prompt_ids"], rids, 0)
        n_fwd += 2
        row = {"snippet_id": sid, "n_tok": len(rids), "logp_U": base, "logp_L0prompt": l0p,
               "G_prompt_swap": l0p - base, "n_spans": len(ref_by[sid]),
               "n_span_pos": sum(len(r["positions"]) for r in ref_by[sid])}
        pos_counts: dict[str, dict] = {}
        for K in live:
            row[f"n_editable_L{K}"] = sum(int(r["editable"]) for r in vec[K][0][sid])
            for kind in SINGLE_KINDS:
                arm = single_arm(kind, K)
                tg = {K: targets_for(K, sid, kind)}
                v, n = run(tg, pids, rids); n_fwd += 1
                row[f"dG_{arm}"] = v - base; pos_counts[arm] = range(n)
        for name, Ks in sets.items():
            for kind in MULTI_KINDS + ("self",):
                arm = multi_arm(kind, name)
                tg = {K: targets_for(K, sid, kind) for K in Ks}
                v, n = run(tg, pids, rids); n_fwd += 1
                row[f"dG_{arm}"] = v - base; pos_counts[arm] = range(n)
        record_positions(row, pos_counts)
        rows.append(row); sink.write(json.dumps(row) + "\n"); sink.flush()
        pr = sets.get("primary")
        msg = (f"M_edit={row.get('dG_M_edit@primary', float('nan')):+.2f} "
               f"M_c3={row.get('dG_M_c3@primary', float('nan')):+.2f} "
               f"M_rnd={row.get('dG_M_random@primary', float('nan')):+.2f} "
               f"SELF={row.get('dG_SELF@primary', float('nan')):+.2f}" if pr else "")
        print(f"{TAG} {pi+1}/{len(sids)} {sid} T={len(pids)+len(rids)} G_swap={row['G_prompt_swap']:+.2f} "
              f"{msg} · {n_fwd} fwd · {(time.time()-t0)/60:.1f} min", flush=True)
    sink.close()
    for r in reps.values():
        r.close()
    meta = {"stage": "score", "live_layers": live, "dead_layers": dead, "sets": sets,
            "liveness": live_rep, "ignore_liveness": bool(args.ignore_liveness),
            "n_items": len(rows), "n_forwards": n_fwd, "ms_per_forward": 1000 * (time.time() - t0) / max(n_fwd, 1),
            "traces": str(args.traces), "smoke": bool(args.smoke), "elapsed_s": time.time() - t0,
            **_prov(cfg["train"], {"experiment": cfg["experiment"], "gate_config": cfg["_path"],
                                   "gate_config_sha256": cfg["_sha256"],
                                   "gate_script_sha256": sha256_file(Path(__file__))})}
    (gdir / "score_manifest.json").write_text(json.dumps(meta, indent=1))
    return 0 if rows else 2


# ── the frozen rules ────────────────────────────────────────────────────────────────────────

def score(rows: list[dict], cfg: dict, meta: dict) -> dict:
    n_boot = int(cfg["n_boot"]); rules = cfg["rules"]
    rng = np.random.default_rng(int(cfg["seed"]))
    n = len(rows)
    idx = rng.integers(0, n, size=(n_boot, n))          # one shared set of item draws
    live = list(meta["live_layers"]); sets = meta["sets"]

    def arm_vals(arm: str) -> np.ndarray:
        # arm_guard: refuse an arm that wrote nothing; SELF may legitimately score exactly 0.
        v, _ = arm_series(rows, arm, allow_exact_zero=arm.startswith("SELF") or "_self_" in arm)
        if len(v) != n:
            raise ArmNotWritten(f"{arm}: wrote on {len(v)} of {n} items")
        return np.asarray(v, float)

    def bm(v: np.ndarray) -> dict:
        d = v[idx].mean(1)
        return {"mean": float(v.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    def bratio(a: np.ndarray, b: np.ndarray) -> dict:
        d = a[idx].mean(1) / b[idx].mean(1)
        return {"ratio": float(a.mean() / b.mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    st: dict = {"experiment": cfg["experiment"], "seed": int(cfg["seed"]), "n_boot": n_boot,
                "n_items": n, "config_sha256": cfg["_sha256"], "live_layers": live,
                "dead_layers": meta["dead_layers"], "sets": sets,
                "ignore_liveness": bool(meta.get("ignore_liveness")), "arms": {}, "per_layer": {}}
    G = np.array([r["G_prompt_swap"] for r in rows], float)
    st["arms"]["G_prompt_swap"] = bm(G)
    for r0 in rows[:1]:
        for k in r0:
            if k.startswith("dG_"):
                st["arms"][k[3:]] = bm(arm_vals(k[3:]))

    # per-layer table + H-M3
    hm3_lb, hm3_ratio = [], []
    for K in live:
        c3 = arm_vals(single_arm("c3", K)); sw = arm_vals(single_arm("swap", K))
        rt_ = bratio(c3, sw)
        pl = {"fidelity_ratio_c3_over_swap": rt_,
              "editable_spans": int(sum(r[f"n_editable_L{K}"] for r in rows)),
              "span_positions": int(sum(r["n_span_pos"] for r in rows))}
        for kind in SINGLE_KINDS:
            pl[kind] = st["arms"][single_arm(kind, K)]
        st["per_layer"][str(K)] = pl
        hm3_lb.append(rt_["ci95"][0]); hm3_ratio.append(rt_["ratio"])
    frac_ok = float(np.mean([lb >= rules["hm3_ratio_min"] for lb in hm3_lb]))
    med = float(np.median(hm3_ratio))
    st["H_M3"] = {"frac_layers_ci_lower_ge_min": frac_ok, "median_ratio": med,
                  "n_live": len(live), "rule": {"ratio_min": rules["hm3_ratio_min"], "confirm_frac": rules["hm3_confirm_frac"]},
                  "verdict": ("CONFIRM" if frac_ok >= rules["hm3_confirm_frac"]
                              else "REFUTE" if med < rules["hm3_ratio_min"] else "INDETERMINATE")}
    peak = max(live, key=lambda K: st["arms"][single_arm("swap", K)]["mean"])
    st["depth_profile"] = {"swap_peak_layer": peak,
                           "swap_means": {str(K): st["arms"][single_arm("swap", K)]["mean"] for K in live},
                           "c3_means": {str(K): st["arms"][single_arm("c3", K)]["mean"] for K in live},
                           "edit_means": {str(K): st["arms"][single_arm("edit", K)]["mean"] for K in live},
                           "random_means": {str(K): st["arms"][single_arm("random", K)]["mean"] for K in live}}
    best_edit_layer = max(live, key=lambda K: st["arms"][single_arm("edit", K)]["mean"])
    st["best_single_edit_layer"] = best_edit_layer

    # the gate per set; the verdict is read on `primary`
    st["gate"] = {}
    for name, Ks in sets.items():
        m = lambda kind: arm_vals(multi_arm(kind, name))  # noqa: E731
        edit, c3, rnd, forg, selfv = m("edit"), m("c3"), m("random"), m("foreign"), m("self")
        g0 = bratio(c3, G)
        gate0 = bool(g0["ratio"] >= rules["gate0_ratio_min"] and g0["ci95"][0] > 0)
        d1 = bm(edit - arm_vals(single_arm("edit", best_edit_layer)))
        hm1 = bool(d1["mean"] > 0 and d1["ci95"][0] > 0)
        # M_edit − M_random and M_edit − M_foreign are same-position contrasts: paired() refuses
        # a mismatch. H-M1 compares a 3-layer write with a 1-layer one BY DESIGN, so it is built
        # from arm_series instead (the zero-position guard still applies).
        dr, _ = paired(rows, multi_arm("edit", name), multi_arm("random", name))
        df, _ = paired(rows, multi_arm("edit", name), multi_arm("foreign", name))
        d2 = bm(np.asarray(dr, float)); thr = rules["hm2_frac_of_c3"] * float(c3.mean())
        hm2 = bool(d2["mean"] >= thr and d2["ci95"][0] > 0)
        ds = bm(np.asarray(df, float)); spec = bool(ds["mean"] > 0 and ds["ci95"][0] > 0)
        self_ok = bool(abs(float(selfv.mean())) <= rules["self_tol_nats"])
        if not gate0:
            verdict = "M-GATE0-FAIL"
        elif hm1 and hm2:
            verdict = "M-EDIT-LIVE"
        elif hm1:
            verdict = "M-GAIN-BUT-GENERIC"
        elif hm2:
            verdict = "M-NO-GAIN"
        else:
            verdict = "M-GENERIC"
        st["gate"][name] = {"layers": Ks, "primary": name == "primary",
                            "gate0": {"c3_over_prompt_swap": g0, "passes": gate0},
                            "H_M1": {"M_edit_minus_best_single_edit": d1, "best_single_layer": best_edit_layer,
                                     "passes": hm1},
                            "H_M2": {"M_edit_minus_M_random": d2, "threshold_0.30_M_c3": thr, "passes": hm2},
                            "specificity": {"M_edit_minus_M_foreign": ds, "passes": spec},
                            "SELF_identity": {"mean": float(selfv.mean()), "tol": rules["self_tol_nats"], "passes": self_ok},
                            "verdict": verdict}
    prim = st["gate"].get("primary")
    st["verdict"] = prim["verdict"] if prim else "NO-PRIMARY-SET"
    st["identity_passes"] = bool(all(g["SELF_identity"]["passes"] for g in st["gate"].values())
                                 and all(abs(st["arms"][single_arm("self", K)]["mean"]) <= rules["self_tol_nats"] for K in live))
    st["reportable"] = bool(st["identity_passes"] and not st["ignore_liveness"])
    return st


def print_stats(st: dict) -> None:
    print(f"\n{TAG} live layers {st['live_layers']}  dead {st['dead_layers']}")
    print(f"{TAG} G_prompt_swap {st['arms']['G_prompt_swap']['mean']:+.2f} {st['arms']['G_prompt_swap']['ci95']}")
    print(f"{TAG} {'L':>3} {'swap':>8} {'c3':>8} {'ratio':>6} {'ci_lo':>6} {'edit':>8} {'random':>8} {'foreign':>8} {'rt':>8} {'self':>7} {'edtbl':>5}")
    for K, pl in st["per_layer"].items():
        r = pl["fidelity_ratio_c3_over_swap"]
        print(f"{TAG} {K:>3} {pl['swap']['mean']:+8.2f} {pl['c3']['mean']:+8.2f} {r['ratio']:6.2f} {r['ci95'][0]:6.2f} "
              f"{pl['edit']['mean']:+8.2f} {pl['random']['mean']:+8.2f} {pl['foreign']['mean']:+8.2f} "
              f"{pl['rt']['mean']:+8.2f} {pl['self']['mean']:+7.3f} {pl['editable_spans']:5d}")
    h3 = st["H_M3"]
    print(f"{TAG} H-M3 {h3['verdict']}: {h3['frac_layers_ci_lower_ge_min']:.2f} of {h3['n_live']} layers clear, median ratio {h3['median_ratio']:.3f}; "
          f"swap peaks at L{st['depth_profile']['swap_peak_layer']}; best single edit L{st['best_single_edit_layer']}")
    for name, g in st["gate"].items():
        print(f"{TAG} set {name} {g['layers']}{' (PRIMARY)' if g['primary'] else ' (secondary)'}: "
              f"gate0 {g['gate0']['c3_over_prompt_swap']['ratio']:.3f} {g['gate0']['c3_over_prompt_swap']['ci95']} {'ok' if g['gate0']['passes'] else 'FAIL'} · "
              f"H-M1 {g['H_M1']['M_edit_minus_best_single_edit']['mean']:+.2f} {g['H_M1']['M_edit_minus_best_single_edit']['ci95']} {'ok' if g['H_M1']['passes'] else 'no'} · "
              f"H-M2 {g['H_M2']['M_edit_minus_M_random']['mean']:+.2f} vs thr {g['H_M2']['threshold_0.30_M_c3']:+.2f} {'ok' if g['H_M2']['passes'] else 'no'} · "
              f"spec {g['specificity']['M_edit_minus_M_foreign']['mean']:+.2f} {'ok' if g['specificity']['passes'] else 'no'} · "
              f"SELF {g['SELF_identity']['mean']:+.3f} -> {g['verdict']}")
    print(f"{TAG} VERDICT {st['verdict']} · identity {'passes' if st['identity_passes'] else 'FAILS'} · reportable={st['reportable']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("vectors", "score"))
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--root", default=None, help="override the training root (smoke runs)")
    ap.add_argument("--layer", type=int, default=None, help="vectors: which trained pair")
    ap.add_argument("--traces", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ignore-liveness", action="store_true",
                    help="score every layer that has vectors even if PAIR-DEAD (smoke only; never a result)")
    ap.add_argument("--score-only", action="store_true", help="re-apply the frozen rules to banked rows")
    ap.add_argument("--max-hours", type=float, default=7.5)
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config)); args._cfg = cfg
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project."); return 2
    args.repair = bool(cfg["repair"]); args.max_spans_per_item = int(cfg["max_spans_per_item"])
    rt = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    if not Path(args.traces).exists():
        print(f"{TAG} REFUSED: missing traces {args.traces}"); return 2
    gdir = rt / cfg["gate_subdir"]; gdir.mkdir(parents=True, exist_ok=True)

    if args.stage == "vectors":
        assert args.layer is not None, "--layer is required for --stage vectors"
        return stage_vectors(args, cfg, rt)
    if not args.score_only:
        rc = stage_score(args, cfg, rt)
        if rc != 0:
            return rc
    rows = [json.loads(l) for l in open(gdir / "gate_rows.jsonl")]
    meta = json.loads((gdir / "score_manifest.json").read_text())
    st = score(rows, cfg, meta)
    (gdir / "gate_stats.json").write_text(json.dumps(st, indent=1))
    print_stats(st)
    if not st["identity_passes"]:
        print(f"{TAG} REFUSED: SELF identity failed — results are NOT reportable.", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
