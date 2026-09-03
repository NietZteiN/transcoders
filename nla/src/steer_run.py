"""B4 — the steering gate: does an NLA-derived direction beat a difference vector that uses no NLA?

THE QUESTION. Everything measured in this programme so far is observational: read the
representation, correlate with behaviour. This is the one experiment that makes the instrument
causal. It injects a direction at layer 20 and asks whether accuracy on adversarially-renamed
code recovers.

**The bar is V3, not zero.** A steering result against "no steering" proves almost nothing —
any large enough perturbation changes outputs. The pre-registered comparison is against a
**contrastive task vector**, `mean(h_L0 − h_L1b)` over *other* programs, which needs no
verbalizer, no reconstructor, and no explanation. If V3 matches V1 then the NLA is buying
interpretability but not capability: a real and publishable finding, but a different paper, and
one we should discover rather than a reviewer.

VECTORS (all injected as `h ← h + α‖h‖Δ̂` at the last prompt token, layer 20):

  V1  gloss        AR("…about fibfib") − AR("…about area, smooth")   natural-language steering
  V2  read-edit    AR(edited AV read) − AR(AV read)                  the NLA paper's own protocol
  V3  task vector  mean over OTHER items of (h_L0 − h_L1b)           NO NLA — the bar
  V4  oracle       this item's own h_L0 − h_L1b                      ceiling, NOT deployable
  R   random       norm-matched noise                                floor
  F   foreign      another item's V1                                 same construction, wrong content
  A   antipodal    −V1                                               must push the other way
  P   prompt-only  a sentence naming the true semantics              the baseline that usually wins

V1 and V2 differ in where the language comes from: V1 from the *stimulus's* own true/decoy
identifier glosses, V2 from editing the *model's* verbalized read. V2 is the paper's protocol
and depends on read quality; V1 does not and costs no AV call, so it runs on every item.

**Balanced Δaccuracy, not flip rate.** A vector that makes the model shout the true algorithm at
everything scores brilliantly on "wrong → right" flips. The statistic here is accuracy change
across *all* items including those already correct, so a direction that destroys correct answers
is penalised for it.

Env `nla-mi`, one GPU. Reads (V2 only) need the AV server; everything else is forward passes.
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

# HOSTS. The banked B4/B5/P0 corpus is Qwen2.5-7B at layer 20 — that is a historical fact about
# the existing results and cannot be re-labelled. New runs go to a non-Chinese host under the
# 2026-09-02 model constraint; gemma12b is the one with a released NLA pair (kitft/nla-gemma3-12b
# -L32-{ar,av}), so it is the only host where the AR-derived conditions could ever run. Layer is
# the NLA's training layer where one exists, mid-depth otherwise.
HOSTS = {
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct", 20),     # banked corpus only — see --model guard
    "gemma12b": ("google/gemma-3-12b-it", 32),
    "llama8b":  ("meta-llama/Llama-3.1-8B-Instruct", 16),
}
DEFAULT_HOST = "gemma12b"
# The AR/AV checkpoint under nla/data/checkpoints/ is the Qwen pair — its config declares
# Qwen2ForCausalLM, hidden_size 3584. Reconstructing into a different model's residual basis is
# not a defined operation and would not fail loudly (a 3584-vector steering a 3840-dim stream
# raises; a same-width host would silently produce nonsense), so the host is pinned explicitly.
# The released Gemma pair (kitft/nla-gemma3-12b-L32-{ar,av}) is in the HF cache but NOT wired in.
AR_HOST = "qwen7b"
TARGET_MODEL, LAYER_INDEX = HOSTS[DEFAULT_HOST]
SEED = 20260724
# 1100, matching `overnight_capture.MAX_NEW_GEN`. The first smoke used 400 and every reply hit
# the wall mid-reasoning (~1,500 chars) without ever emitting the answer line, so all three
# baselines scored 0/3 on UNOBFUSCATED code — where the banked runs get 0.64-0.70. A truncation
# budget that silently zeroes the control is indistinguishable from a broken model.
MAX_NEW_GEN = 1100

PREAMBLE = (
    "You are an expert software engineer taking part in a code comprehension study. "
    "You will be shown one function. Read it carefully, reason about what it computes "
    "step by step, and then predict the exact output of the given call. Be precise "
    "about values and types, and do not guess without tracing the computation. "
    "Here is the function and the call to evaluate:\n\n"
)
ANSWER_RE = re.compile(r"Output:\s*(.+?)\s*$", re.M | re.I)


def _norm(s) -> str:
    """The established harness's normalization (overnight_capture.grade), reused verbatim so
    accuracies here are comparable with the banked corpus rather than merely similar."""
    return re.sub(r"[\s'\"`]", "", str(s)).lower()


def graded(reply: str, truth) -> tuple[str | None, bool]:
    m = ANSWER_RE.search(reply or "")
    got = m.group(1).strip() if m else None
    return got, (got is not None and _norm(got) == _norm(truth))


def identifier_terms(code: str, spans) -> list[str]:
    """Distinct identifier strings at the annotated spans — the material for a gloss."""
    out, seen = [], set()
    for a, b in (spans or []):
        t = code[a:b].strip()
        if t and t not in seen and len(t) > 1:
            seen.add(t)
            out.append(t)
    return out[:8]


def gloss(terms: list[str]) -> str:
    """Same template the banked stimuli used: 'code whose identifiers are about X'."""
    return "code whose identifiers are about " + (", ".join(terms) if terms else "unnamed values")


def load_pairs(limit: int | None, rng: random.Random) -> list[dict]:
    """L0/L1b pairs sharing a snippet_id, with ground truth on the L1b side."""
    rows = []
    for ds in ("dataset_a", "dataset_b"):
        p = _PROJ / "data" / "stimuli" / ds / f"{ds}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p) if l.strip()]
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["snippet_id"], {})[r["tier"]] = r

    out = []
    for sid, tiers in sorted(by.items()):
        a, b = tiers.get("L0"), tiers.get("L1b")
        if not a or not b or not b.get("expected_output"):
            continue
        call_l0, call_l1b = build_call(a), build_call(b)
        if not call_l0 or not call_l1b:
            continue
        out.append({
            "snippet_id": sid, "language": b.get("language"),
            "call_l0": call_l0, "call_l1b": call_l1b,
            "code_l0": a["code"], "code_l1b": b["code"],
            "truth": b["expected_output"],
            "gloss_true": gloss(identifier_terms(a["code"], a.get("identifier_spans"))),
            "gloss_decoy": gloss(identifier_terms(b["code"], b.get("identifier_spans"))),
            # Kept separately from the glosses for V2, which needs the individual terms in
            # order to make exactly ONE substitution rather than swapping the whole gloss.
            "terms_true": identifier_terms(a["code"], a.get("identifier_spans")),
            "terms_decoy": identifier_terms(b["code"], b.get("identifier_spans")),
            # The decoy->true correspondence, from the stimulus's own rename map (original ->
            # decoy, so it is inverted at use). Pairing the two term lists BY INDEX instead
            # looks reasonable and is wrong: adversarial renaming introduces more annotated
            # identifier sites than the original has distinct terms (e.g. Python/100 is 3
            # decoy terms against 1 true one), so the lists differ in length on 52 of 60
            # pairs and any positional zip silently mismatches the rest.
            "rename_map": (b.get("meta") or {}).get("rename_map") or {},
            "id_spans_l1b": b.get("identifier_spans") or [],
        })
    rng.shuffle(out)
    return out[:limit] if limit else out


# Call construction is NOT re-derived here. `task_bank.build_call` already handles every case
# this stimulus set contains — meta.input that is already a call string, meta.fn_name being the
# ORIGINAL name on renamed tiers (mapped through rename_map), stale call heads validated against
# the code, and rows whose head is a `myFunct` placeholder with no usable fn_name. Re-deriving it
# produced `get_bounds(get_bounds(3))` for a program defining `make_a_pile`, and the model
# answered `None` because the function named did not exist — clean-code accuracy 1/16 where the
# banked runs get 0.64-0.70. 17% of L0 rows have an fn_name that is absent from their code.
from task_bank import build_call  # noqa: E402


def build_user(code: str, call: str) -> str:
    """Exact wording of `overnight_capture.build_user`, so accuracies stay comparable."""
    return (PREAMBLE + code +
            f"\n\nWhat is the exact output of `{call}`? Reason step by step, "
            f"then end with one line exactly of the form `Output: <value>`.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alphas", default="0.5,1.0,2.0")
    ap.add_argument("--frozen-alpha", type=float, default=None,
                    help="skip the sweep and run the full battery at this alpha")
    ap.add_argument("--out-dir", default=str(_PROJ / "data/nla/n12"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sglang-url", default="http://localhost:30004")
    ap.add_argument("--with-v2", action="store_true",
                    help="include V2, the minimal single-word-edit vector (no AV server needed "
                         "— the edit is applied to the gloss, same text V1 reconstructs)")
    ap.add_argument("--positions", default="last_prompt",
                    choices=["last_prompt", "all_reply", "all"],
                    help="where to inject. The banked B4 run used last_prompt (the site the "
                         "NLA paper steers) and hardcoded it; the mechanism always supported "
                         "the others, so a null at one position was never a null in general.")
    # P0.4. LAYER_INDEX=20 is where the released NLA pair was trained, not a measured
    # choice; P0.1 found cross-item coherence of the task direction peaks at 13 instead.
    # V3/V4/R are pure activation arithmetic and are defined at every layer, so the read
    # site and the write site move together and no autoencoder is involved.
    ap.add_argument("--layer", type=int, default=LAYER_INDEX,
                    help="decoder block whose OUTPUT is both read and written. Read and write "
                         "must be the same site: ActivationExtractor and ActivationSteerer both "
                         "hook layers[K], which nla/src/p04_gate.py verifies numerically.")
    # 2026-08-29: two runs of the identical command on the SAME physical GPU disagree on 6 of
    # 60 items (log/nla-harness/2026-08-29_reliability-floor-and-length.md). Greedy decoding draws
    # no RNG, so `torch.manual_seed` never touches this path — the divergence is float kernel
    # execution: per-process cuBLAS/cuDNN autotuning and attention-backend selection. This flag
    # pins all of it. It is OFF by default because the math SDPA path is materially slower and
    # because turning it on would make new runs incomparable with every banked one.
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuBLAS/cuDNN/SDPA to deterministic kernels. Needs "
                         "CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment BEFORE CUDA "
                         "initialises — the runner checks and refuses if it is unset.")
    # 2026-08-29: of the 10 items that flipped between two same-card runs, 3 had one run fail to
    # emit an `Output:` line at all, and disagreeing items ran ~25% longer than agreeing ones. If
    # the reproducibility floor is partly replies hitting this wall, raising it should shrink the
    # floor. Default unchanged at 1100 so banked runs stay comparable.
    ap.add_argument("--max-new-gen", type=int, default=MAX_NEW_GEN,
                    help="generation budget in tokens (default 1100, matching the banked corpus)")
    ap.add_argument("--baseline-only", action="store_true",
                    help="run the unsteered baseline pass and stop — no steering battery, no "
                         "prompt-only control. For collecting labels and reply lengths cheaply.")
    ap.add_argument("--model", default=DEFAULT_HOST, choices=sorted(HOSTS),
                    help="subject host. Default is the non-Chinese host required by the "
                         "2026-09-02 constraint; qwen7b is retained ONLY so the banked corpus "
                         "stays reproducible and requires --allow-banked-host.")
    ap.add_argument("--allow-banked-host", action="store_true",
                    help="acknowledge that --model qwen7b re-runs a model the project is no "
                         "longer permitted to run, and is being used to reproduce banked rows.")
    ap.add_argument("--multilayer", action="store_true",
                    help="Close the KV bypass: write at EVERY layer 0..--layer during prefill, "
                         "so the K/V entries later tokens attend to carry the correction. The "
                         "single-layer hook leaves layers 0..L untouched at the edited position "
                         "(measured: max |delta| = 0.0), so downstream attention reads the "
                         "un-edited decoy through the bottom L+1 layers. AR-derived conditions "
                         "are unavailable here — the AR exists at one layer only.")
    ap.add_argument("--multilayer-bank", default=None,
                    help="per-layer contrastive differences from multilayer_vectors.py; "
                         "defaults to the bank for the chosen --model host")
    ap.add_argument("--max-hours", type=float, default=11.0)
    # The B4 run stored answers and reply LENGTHS but not the replies themselves, so "what did
    # steering actually do to the text" was unanswerable after the fact. These two flags make a
    # small illustrative re-run cheap: pick the items that flipped, keep the generations.
    ap.add_argument("--snippets", default=None,
                    help="comma-separated snippet_ids (or a file of them) to restrict the run to")
    ap.add_argument("--only-conditions", default=None,
                    help="comma-separated condition names to run, e.g. V1_gloss,V3_taskvec")
    ap.add_argument("--save-replies", action="store_true",
                    help="store the generated text alongside the grade")
    args = ap.parse_args()
    TARGET_MODEL, host_layer = HOSTS[args.model]
    if args.model == "qwen7b" and not args.allow_banked_host:
        raise SystemExit(
            "--model qwen7b is the banked host and the project is no longer permitted to run "
            "Chinese models. Pass --allow-banked-host only to reproduce banked rows.")
    # --layer defaults to the OLD constant, so an unset --layer must follow the chosen host or a
    # gemma run would silently steer layer 20 of 48 while every artifact says otherwise.
    if args.layer == LAYER_INDEX and args.model != DEFAULT_HOST:
        args.layer = host_layer
    if args.multilayer_bank is None:
        args.multilayer_bank = str(_PROJ / "data/nla/p0/steerv2" / args.model
                                   / "multilayer_bank.npz")

    import os

    import torch
    torch.manual_seed(SEED)

    if args.deterministic:
        # cuBLAS reads this when it creates its handle, which happens on first CUDA use. Setting
        # it from inside Python after torch has touched the GPU is silently too late, so this
        # refuses rather than producing a run that merely looks deterministic.
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic requires CUBLAS_WORKSPACE_CONFIG=:4096:8 exported "
                             "before python starts; got "
                             f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
        # warn_only: some attention/index kernels have no deterministic implementation. A hard
        # error would abort the run and tell us nothing; a warning names the op and lets the
        # measurement proceed, which is the point of the experiment.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Flash and mem-efficient attention both reduce in a nondeterministic order. The math
        # path is the slow, reproducible one.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[b4] deterministic mode: cuBLAS workspace pinned, cudnn.benchmark off, "
              "SDPA forced to math", flush=True)
    from capture_core import WallGuard, JsonlSink, align_reply
    from extract import ActivationExtractor
    from nla_inference import NLACritic
    from steer import ActivationSteerer, SteerSpec
    from steer_vectors import (ActivationPair, TaskVectorBank, antipodal, nla_edit_direction,
                               random_direction, substitute_terms, word_edit_direction)

    rng = random.Random(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.limit, rng)
    if args.snippets:
        want = Path(args.snippets).read_text().split() if Path(args.snippets).exists() \
            else [x.strip() for x in args.snippets.split(",")]
        want = {w for w in want if w}
        pairs = [p for p in pairs if p["snippet_id"] in want]
        print(f"[b4] restricted to {len(pairs)} of {len(want)} requested snippets "
              f"(applied before the baseline pass)", flush=True)
    print(f"[b4] {len(pairs)} L0/L1b pairs with ground truth", flush=True)

    # The AR reconstructor was trained at layer 20. Reconstructing into another layer's basis
    # is not a defined operation, so the AR-derived conditions are unavailable off-layer — and
    # loading an 11 GB model that nothing will call is pure waste. Resolve the battery first.
    keep = ({c.strip() for c in args.only_conditions.split(",") if c.strip()}
            if args.only_conditions else None)
    AR_CONDS = {"V1_gloss", "V2_wordedit", "F_foreign", "A_antipodal"}
    need_ar = (keep is None) or bool(keep & AR_CONDS) or args.with_v2
    if need_ar and args.model != AR_HOST:
        raise SystemExit(
            f"--model {args.model} requests AR-derived conditions "
            f"({sorted((keep or AR_CONDS) & AR_CONDS)}), but the AR checkpoint on disk belongs to "
            f"{AR_HOST} ({HOSTS[AR_HOST][0]}). Restrict with "
            f"--only-conditions V3_taskvec,V4_oracle,R_random.")
    if args.layer != host_layer and need_ar:
        raise SystemExit(
            f"--layer {args.layer} requests AR-derived conditions "
            f"({sorted((keep or AR_CONDS) & AR_CONDS)}), but the AR checkpoint is trained at "
            f"layer {host_layer}. Restrict with --only-conditions V3_taskvec,V4_oracle,R_random.")

    ex = ActivationExtractor(TARGET_MODEL, args.layer, device=args.device)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar",
                   device=args.device) if need_ar else None
    tokz = ex.tokenizer
    if args.multilayer:
        # The AR guard above already rejects AR-derived conditions off-layer; multilayer mode
        # is off-layer at every layer but the target, so it must reject them at L20 too.
        if need_ar:
            raise SystemExit(
                "--multilayer cannot run AR-derived conditions: the AR is trained at layer "
                f"{host_layer} only, so no licensed per-layer NLA direction exists. Restrict "
                "with --only-conditions V3_taskvec,V4_oracle,R_random.")
        from steer_multilayer import MultiLayerSpec, MultiLayerSteerer
        steerer = MultiLayerSteerer(ex.model, range(args.layer + 1))
        print(f"[b4] MULTILAYER: writing layers 0..{args.layer} during prefill "
              f"({args.layer + 1} sites)", flush=True)
    else:
        steerer = ActivationSteerer(ex.model, args.layer)
    print(f"[b4] layer {args.layer} (read and write) · AR "
          f"{'loaded' if need_ar else 'skipped — no AR-derived condition requested'}", flush=True)
    wall = WallGuard(args.max_hours)

    def gen(code: str, call: str, spec=None, extra: str = "") -> tuple[str, int]:
        user = build_user(code, call) + extra
        ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
        if args.multilayer:
            # MultiLayerSpec resolves "last_prompt" inside the hook from the prefill call's own
            # sequence length, so it needs no prompt_len — and must not be handed one, or the
            # two position conventions could silently diverge.
            steerer.set_spec(spec)
        else:
            steerer.set_spec(spec, prompt_len=len(ids), max_total=len(ids) + args.max_new_gen)
        with torch.no_grad():
            o = ex.model.generate(torch.tensor([ids], device=ex.model.device),
                                  max_new_tokens=args.max_new_gen, do_sample=False,
                                  pad_token_id=tokz.eos_token_id)
        steerer.reset()
        return tokz.decode(o[0][len(ids):], skip_special_tokens=True), len(ids)

    def last_tok_act(code: str, call: str) -> np.ndarray:
        """Activation at the final prompt token — the site the paper steers."""
        res = ex.extract_chat(build_user(code, call), None, text_id="x")
        return res.activations[-1]

    # ── stage 1: baselines + per-item activations ─────────────────────────
    base_p = out_dir / "baseline.jsonl"
    sink = JsonlSink(base_p, key_field="snippet_id")
    done = sink.done_keys()
    acts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    base: dict[str, dict] = {}
    if base_p.exists():
        for l in open(base_p):
            r = json.loads(l)
            base[r["snippet_id"]] = r

    for i, p in enumerate(pairs, 1):
        if wall.expired():
            break
        sid = p["snippet_id"]
        h0, h1 = last_tok_act(p["code_l0"], p["call_l0"]), last_tok_act(p["code_l1b"], p["call_l1b"])
        acts[sid] = (h0, h1)
        if sid in done:
            continue
        rep, _ = gen(p["code_l1b"], p["call_l1b"])
        got, ok = graded(rep, p["truth"])
        rep0, _ = gen(p["code_l0"], p["call_l0"])
        got0, ok0 = graded(rep0, p["truth"])
        # `reply_chars` has always meant the L1b reply and is kept under that name so banked
        # files stay readable. The L0 length was never recorded, which left the clean tier with
        # no length baseline — the limitation that blocked the 2026-08-30 tier result. Both are
        # now written explicitly.
        row = {"snippet_id": sid, "truth": p["truth"], "l1b_answer": got, "l1b_correct": ok,
               "l0_answer": got0, "l0_correct": ok0, "reply_chars": len(rep),
               "l1b_reply_chars": len(rep), "l0_reply_chars": len(rep0),
               "l1b_parsed": got is not None, "l0_parsed": got0 is not None}
        sink.append(row)
        base[sid] = row
        print(f"[b4] baseline {i}/{len(pairs)} {sid} L0={ok0} L1b={ok}", flush=True)
    sink.close()

    usable = [p for p in pairs if p["snippet_id"] in base and p["snippet_id"] in acts]
    n_wrong = sum(1 for p in usable if not base[p["snippet_id"]]["l1b_correct"])
    print(f"[b4] {len(usable)} usable · {n_wrong} wrong at L1b baseline (the recovery target)",
          flush=True)

    if args.baseline_only:
        steerer.close()
        ex.close()
        (out_dir / "run_manifest.json").write_text(json.dumps({
            "experiment": "n12_b4_baseline_only", "seed": SEED, "argv": sys.argv,
            "model": TARGET_MODEL, "host": args.model, "layer": args.layer, "baseline_only": True,
            "max_new_gen": args.max_new_gen, "n_pairs": len(usable),
            "n_wrong_baseline": n_wrong, "elapsed_hours": round(wall.elapsed_h(), 3),
            "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
        print(f"[b4] baseline-only done · {wall.elapsed_h():.2f} h", flush=True)
        return 0

    # ── vectors ───────────────────────────────────────────────────────────
    bank = TaskVectorBank([ActivationPair(p["snippet_id"],
                                          torch.tensor(acts[p["snippet_id"]][0]),
                                          torch.tensor(acts[p["snippet_id"]][1]))
                           for p in usable])

    def v1(p) -> torch.Tensor:                       # NLA, natural language, no AV call
        return (ar.reconstruct(p["gloss_true"]).float()
                - ar.reconstruct(p["gloss_decoy"]).float()).cpu()

    v1_cache = {p["snippet_id"]: v1(p) for p in usable} if need_ar else {}

    # ── V2: the same edit, but minimal ────────────────────────────────────
    # V1 replaces the WHOLE decoy gloss with the true one, so a difference between them could
    # come from any of up to eight swapped terms, or from the sentence being different overall.
    # V2 changes exactly one word. If the belief is carried by a single lexical item, V2 should
    # do most of V1's work; if it needs the whole description, V2 should do much less. That
    # contrast is the point, and it is why `word_edit_direction(require_single=True)` raises
    # rather than silently making two substitutions.
    #
    # No AV call is needed: the edit is applied to the gloss `load_pairs` already builds, which
    # is the same text V1 reconstructs. An AV-read-and-edit variant would additionally measure
    # the verbalizer's paraphrase, which is a different question.
    def v2_mapping(p) -> dict[str, str] | None:
        """One decoy->true substitution, taken from the stimulus's own rename map.

        Deterministic tie-break: among the decoy terms that appear exactly once in the gloss,
        take the one appearing earliest in the gloss, so the choice does not depend on dict
        ordering. Items whose rename map yields no single-occurrence term are skipped and
        counted rather than forced — V2's whole claim is that *one* word changed.
        """
        rmap = p.get("rename_map") or {}
        cands = []
        for true_name, decoy_name in rmap.items():
            if not decoy_name or true_name.lower() == decoy_name.lower():
                continue
            _, n = substitute_terms(p["gloss_decoy"], {decoy_name: true_name})
            if n == 1:
                cands.append((p["gloss_decoy"].lower().find(decoy_name.lower()),
                              decoy_name, true_name))
        if not cands:
            return None
        cands.sort()
        return {cands[0][1]: cands[0][2]}

    v2_cache: dict[str, torch.Tensor] = {}
    v2_skipped: list[str] = []
    if args.with_v2:
        for p in usable:
            m = v2_mapping(p)
            if m is None:
                v2_skipped.append(p["snippet_id"])
                continue
            try:
                v2_cache[p["snippet_id"]] = word_edit_direction(
                    ar, p["gloss_decoy"], m, require_single=True).cpu()
            except ValueError as e:
                v2_skipped.append(p["snippet_id"])
                print(f"[b4] V2 skip {p['snippet_id']}: {e}", flush=True)
        print(f"[b4] V2: {len(v2_cache)} directions built, {len(v2_skipped)} skipped "
              f"(no minimal single-word edit available)", flush=True)

    # The AR has done its whole job by this point: V1 and V2 are cached, and V3/V4 are pure
    # activation arithmetic while the controls derive from v1_cache. Holding it costs ~11 GB
    # for the rest of the run, and on a card also hosting the AV server (21.6 GB) plus the
    # subject model (15.2 GB) that leaves ~800 MB free — at which point the CUDA allocator
    # thrashes on every generation. Measured effect: throughput decayed from 5.7 to 2.0
    # rows/min over the first two hours, pushing the ETA past the wall guard. Freeing it here
    # returns ~12 GB of headroom to the generation loop.
    del ar
    ar = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    conds: dict[str, callable] = {
        "V1_gloss":      lambda p: v1_cache[p["snippet_id"]],
        "V3_taskvec":    lambda p: bank.direction_for(exclude=p["snippet_id"]),
        "V4_oracle":     lambda p: bank.oracle_for(p["snippet_id"]),
        "R_random":      lambda p: random_direction(ex.d_model,
                                                    seed=SEED + hash(p["snippet_id"]) % 9973,
                                                    like=v1_cache.get(p["snippet_id"])),
        "F_foreign":     lambda p: v1_cache[[q["snippet_id"] for q in usable
                                             if q["snippet_id"] != p["snippet_id"]][
                                                 hash(p["snippet_id"]) % max(len(usable) - 1, 1)]],
        "A_antipodal":   lambda p: antipodal(v1_cache[p["snippet_id"]]),
    }
    if args.with_v2:
        # Only items with a well-defined minimal edit. Every other condition still runs on the
        # full set, so V2's denominator differs — steer_stats pairs per condition, and the V2
        # row must be read against its own n, not the battery's.
        conds["V2_wordedit"] = lambda p: v2_cache[p["snippet_id"]]

    if args.multilayer:
        # Per-layer V3/V4 replace their single-layer namesakes wholesale, keeping the condition
        # NAMES identical so steer_stats pairs multilayer rows against the banked single-layer
        # rows without a translation table. The resume key carries |ML, so they never collide.
        bank_npz = np.load(args.multilayer_bank, allow_pickle=True)
        Dml = bank_npz["deltas"].astype(np.float32)          # [n_pairs, n_layers, d]
        ml_idx = {str(k): i for i, k in enumerate(bank_npz["item_ids"])}
        missing = [p["snippet_id"] for p in usable if p["snippet_id"] not in ml_idx]
        if missing:
            raise SystemExit(f"--multilayer bank is missing {len(missing)} usable items "
                             f"(first: {missing[:3]}); re-run multilayer_vectors.py")
        # A bank of the wrong width or depth is the silent failure this whole mode is exposed
        # to: a [n, 28, 3584] qwen bank against a 48-layer 3840-wide gemma would raise on the
        # first write, but a same-width host would not, and would steer with vectors extracted
        # at layers that do not correspond. Check both against the live model, not the filename.
        if Dml.shape[2] != ex.d_model:
            raise SystemExit(f"bank width {Dml.shape[2]} != model d_model {ex.d_model} "
                             f"({args.multilayer_bank}) — wrong host's bank")
        if Dml.shape[1] != ex.n_layers:
            raise SystemExit(f"bank depth {Dml.shape[1]} != model n_layers {ex.n_layers} "
                             f"({args.multilayer_bank}) — wrong host's bank")
        if args.layer >= Dml.shape[1]:
            raise SystemExit(f"--layer {args.layer} outside bank depth {Dml.shape[1]}")
        print(f"[b4] bank {tuple(Dml.shape)} matches host "
              f"(d_model {ex.d_model}, {ex.n_layers} layers)", flush=True)
        n_ml = Dml.shape[0]
        Dsum = Dml.sum(axis=0)

        def _ml_v3(p):
            """Leave-one-out mean, per layer. Fit on pairs, applied to items — so the applied
            item must be excluded or the direction leaks its own answer."""
            i = ml_idx[p["snippet_id"]]
            V = (Dsum - Dml[i]) / (n_ml - 1)
            return {l: torch.from_numpy(V[l].copy()) for l in range(args.layer + 1)}

        def _ml_v4(p):
            V = Dml[ml_idx[p["snippet_id"]]]
            return {l: torch.from_numpy(V[l].copy()) for l in range(args.layer + 1)}

        def _ml_rand(p):
            """Per-layer random directions. The hook normalises each and scales by the LOCAL
            activation norm, so these are already norm-matched to V3 layer by layer."""
            g = torch.Generator().manual_seed(SEED + hash(p["snippet_id"]) % 9973)
            return {l: torch.randn(Dml.shape[2], generator=g)
                    for l in range(args.layer + 1)}

        conds = {"V3_taskvec": _ml_v3, "V4_oracle": _ml_v4, "R_random": _ml_rand}
        if keep is not None:
            conds = {k: v for k, v in conds.items() if k in keep}
        print(f"[b4] multilayer conditions: {sorted(conds)} · bank {tuple(Dml.shape)}", flush=True)

    # ── stage 2: alpha sweep, then the frozen battery ─────────────────────
    alphas = ([args.frozen_alpha] if args.frozen_alpha
              else [float(a) for a in args.alphas.split(",")])
    res_p = out_dir / "steer_results.jsonl"
    rsink = JsonlSink(res_p, key_field="key")
    rdone = rsink.done_keys()

    # The resume key MUST carry the injection position, or a position sweep silently inherits
    # the previous position's rows as "done" and reports last_prompt numbers under a different
    # label. It is suffixed rather than inserted so the 420 banked last_prompt rows — whose
    # keys predate this flag — still resume instead of re-running for 2.3 GPU-h.
    def run_key(sid: str, cname: str, a: float) -> str:
        base = f"{sid}|{cname}|{a}"
        if args.positions != "last_prompt":
            base = f"{base}|{args.positions}"
        # Same reasoning, one layer down: the banked rows carry no layer tag and are L20, so
        # L20 keeps the bare key and every other layer is suffixed.
        if args.layer != host_layer:
            base = f"{base}|L{args.layer}"
        # The banked rows are qwen7b and carry no host tag, so qwen7b keeps the bare key and
        # every other host is suffixed. Without this a gemma row would resume as a banked one.
        if args.model != "qwen7b":
            base = f"{base}|{args.model}"
        # Multilayer rows are a different intervention at the same layer and alpha. Without this
        # they would collide with the banked single-layer rows and resume as "done".
        return f"{base}|ML" if args.multilayer else base

    if args.only_conditions:
        keep = {c.strip() for c in args.only_conditions.split(",") if c.strip()}
        conds = {k: v for k, v in conds.items() if k in keep}
        print(f"[b4] conditions restricted to {sorted(conds)}", flush=True)

    t0 = time.time()
    todo = [(p, cname, a) for p in usable for cname in conds for a in alphas]
    todo = [t for t in todo if run_key(t[0]["snippet_id"], t[1], t[2]) not in rdone]
    print(f"[b4] {len(todo)} steered generations to run", flush=True)

    for n, (p, cname, a) in enumerate(todo, 1):
        if wall.expired():
            print("[b4] wall budget reached — stopping cleanly", flush=True)
            break
        sid = p["snippet_id"]
        key = run_key(sid, cname, a)
        try:
            delta = conds[cname](p)
            spec = (MultiLayerSpec(deltas=delta, alpha=a, positions=args.positions)
                    if args.multilayer
                    else SteerSpec(delta=delta, alpha=a, positions=args.positions))
            rep, plen = gen(p["code_l1b"], p["call_l1b"], spec)
            got, ok = graded(rep, p["truth"])
            row = {"key": key, "snippet_id": sid, "condition": cname, "alpha": a,
                   "positions": args.positions, "layer": args.layer,
                   "multilayer": bool(args.multilayer),
                   "n_layers_written": (args.layer + 1) if args.multilayer else 1,
                   "answer": got, "correct": ok, "parsed": got is not None,
                   "baseline_correct": base[sid]["l1b_correct"],
                   "reply_chars": len(rep)}
            if args.save_replies:
                # the edit itself is the point of V1/V2: the direction IS this pair of sentences
                row.update({"reply": rep, "truth": p["truth"],
                            "gloss_decoy": p.get("gloss_decoy"),
                            "gloss_true": p.get("gloss_true"),
                            "code_l1b": p.get("code_l1b"), "call_l1b": p.get("call_l1b")})
            rsink.append(row)
        except Exception as e:
            rsink.append({"key": key, "snippet_id": sid, "condition": cname, "alpha": a,
                          "error": repr(e)[:200]})
        if n % 20 == 0:
            rate = n / (time.time() - t0)
            print(f"[b4] {n}/{len(todo)} · {rate*60:.1f}/min · "
                  f"ETA {(len(todo)-n)/max(rate,1e-9)/60:.0f} min", flush=True)

    # ── prompt-only baseline (no steering; the control that usually wins) ──
    for p in usable:
        if wall.expired():
            break
        key = f"{p['snippet_id']}|P_prompt|0"
        if key in rdone:
            continue
        rep, _ = gen(p["code_l1b"], p["call_l1b"],
                     extra=f"\n\nNote: the identifiers are misleading. This is {p['gloss_true']}.")
        got, ok = graded(rep, p["truth"])
        rsink.append({"key": key, "snippet_id": p["snippet_id"], "condition": "P_prompt",
                      "alpha": 0.0, "answer": got, "correct": ok, "parsed": got is not None,
                      "baseline_correct": base[p["snippet_id"]]["l1b_correct"]})
    rsink.close()
    steerer.close()
    ex.close()

    (out_dir / "run_manifest.json").write_text(json.dumps({
        "experiment": "n12_b4_steering_gate", "seed": SEED, "argv": sys.argv,
        "model": TARGET_MODEL, "host": args.model, "layer": args.layer, "alphas": alphas,
        "deterministic": bool(args.deterministic), "max_new_gen": args.max_new_gen,
        "n_pairs": len(usable), "n_wrong_baseline": n_wrong,
        "conditions": list(conds) + ["P_prompt"],
        "elapsed_hours": round(wall.elapsed_h(), 3),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[b4] done · {wall.elapsed_h():.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
