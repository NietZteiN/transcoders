"""Can steering move accuracy forward? — CodeSteer and baselines on ONE model, via THEIR runtime.

Pre-registered: log/nla-harness/2026-09-14_steering-bakeoff-prereg.md (+ arms amendment).

WHY EVERY ARM GOES THROUGH THEIR `SteeredCausalLM`. The steered arms must use their attention-steering
implementation, and if the unsteered control used *my* generate loop instead, any difference could be
the harness rather than the steering. So the control is `steering_config = None` through the same
object: prompt assembly, sampling loop and decoding are then bit-for-bit shared across arms and the
only thing that varies is the steering config.

THE PAPER'S CONFIGURATION, read off their code AND their README command (obfuscation/main.py;
README: `--steer --prior slice_hybrid --beta-post 0.8 --steer-last-n-layers 8 --head-subset-mode none`):
    enabled_levels=[2]            # `enabled = [2] if args.steer else []` -- level-2 steering only
    prior="slice_hybrid"          # main.py FORCES this for humaneval/cruxeval + counterfactual_tf
    n_bins=12, binning="equal_count"
    beta_post=0.8                 # README. `level2_post` is `if beta == 0.0: return attn_probs` --
                                  # at the SteeringConfig DEFAULT (0.0) steering is the IDENTITY.
    steer_layer_start/end         # last 8 layers, set after build() from num_hidden_layers as main.py
                                  # does (CodeLlama-7B: 24..31)
    head_subset_mode="none"       # README-exact arm `codesteer`; the paper's calibrated sparse head
                                  # subset (Eq. 10, top-4 heads/layer) is the separate arm `codesteer_auto`
HARNESS FAULT 2026-09-14 (bakeoff-beta-fault entry): the first `codesteer`/`rand_prior` runs used
beta_post=0.0 (the dataclass default, which I had mis-read as the paper's setting) and so were the
identity -- 150/150 generations equal between the two arms. Those outputs are kept as
`codesteer_beta0.jsonl` / `rand_prior_beta0.jsonl` (noise-floor controls: same split-prefill code path,
no steering). The liveness gate then counted `steer_calls`, which increments even when beta=0; it is
replaced by an EFFECT gate that counts level-2 applications whose output attention differs from the input.
`slice_hybrid` is `SlicingHybridPrior(SlicingPrior(ASTPrior))` -- AST-based, NOT the Joern prior, so
it needs only `javalang`. Joern is absent from this host and is not used; `JoernSlicePrior` is a
different class we never touch.

MODEL: CodeLlama-7b-Instruct, chosen because it is the ONLY model measured so far with damage worth
recovering -- H-R1c gave original 0.7995 -> renamed 0.6866 (drop +0.1129), against Llama-3.1-8B's
+0.0300 (DAMAGE-ABSENT) and CodeGemma-7B's +0.0760. It is a Llama architecture, so their
`install_llama_steering` applies unmodified.

RESIDUAL ARMS (bakeoff-amendment-arms, 2026-09-14). Written with OUR `PositionReplacer` at the output of
layer K of THEIR model object, at the renamed prompt's identifier token positions (`ase_align_java.align`
+ the BOS offset measured per prompt, never assumed):
    swap_oracle : the ORIGINAL prompt's own state at the aligned positions -- per token when the original
                  and decoy spans tokenise to the same length, else the original span's mean at every
                  decoy token. The ceiling for any span-position write.
    foreign     : a DIFFERENT snippet's span state (seeded), same positions -- the specificity control.
    erasure     : h1b_s + dn^(-i) * unit(E^(-i)), the leave-one-item-out mean-difference vector with the
                  LOO magnitude (= `erase_loonorm`, the oracle-free arm of H-E14), where Delta_s = h0_s -
                  h1b_s over every span of every OTHER snippet in the run. Deployable: nothing from the
                  written item's original enters its vector.
    combined    : `codesteer` (their level-2 attention steering, decode-only) + `erasure` (our write at
                  prefill) together -- different channels; the amendment reads super-/sub-additivity.
CHAT TEMPLATE (H-R7, 2026-09-15_chat-template-rerun-prereg). `--chat-template` wraps the string their
`_build_prompt` returns in the tokenizer's chat template (`[INST] ... [/INST]` for CodeLlama-Instruct),
the ids our H-R1 harness fed the same model. It is installed as the INSTANCE's `_build_prompt`, so their
generation, their prior's token alignment, their Eq. 10 calibration and our residual alignment all see the
one templated string; their code is untouched. Why: their raw prompt drops CodeLlama-Instruct to 0.545 on
clean code (chance 0.500) and inverts the renaming damage, so the raw-prompt bake-off ranked arms on a
deficit with the wrong sign (parse-rate-correction entry). Gate, every prompt: the ids the tokenizer
produces from the templated text must equal `apply_chat_template(..., tokenize=True)` exactly.
Identity gate before any residual arm, per snippet: writing the renamed prompt's OWN exact per-token
state must leave the next-token logits at the last prompt position within --self-tol of the unhooked
forward, and `n_positions_written` must equal the number of aligned decoy tokens; any failure exits 3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase_align_java import align                    # noqa: E402  (6/6 tests, refuses on mismatch)
from steer import PositionReplacer                  # noqa: E402  (the same writer as every banked run)

TAG = "[STEER]"
SEED = 20260724
TEMP, TOP_P = 0.7, 1.0
ANSWER_PREFIX = "\n\nJSON answer:\n"

PAPER_CFG = dict(enabled_levels=[2], prior="slice_hybrid", n_bins=12, binning="equal_count",
                 beta_bias=0.0, beta_post=0.8, lambda_attn=1.0, lambda_mlp=1.0,
                 head_subset_mode="none", head_mask_apply_to="both")
STEER_LAST_N_LAYERS = 8                    # README `--steer-last-n-layers 8`
# `codesteer_beta0` is the IDENTITY-FLOOR control: their steering machinery with beta_post=0, where
# `level2_post` returns attn_probs unchanged. It is in ATTN_ARMS so it takes the same split-prefill code
# path AND the same effect gate as `codesteer` -- the gate is the positive check that it really is the
# identity (l2_effective_calls must be 0, where `codesteer` gets 4096/4096). It enters no verdict; it
# measures the sampling + code-path variance the other arms' differences are read against (H-R9's +-0.05
# floor gate, log/nla-harness/2026-09-15_cn-rescore-results.md). In the raw-prompt bake-off this arm came
# from the 2026-09-14 beta fault rather than by request; here it is asked for explicitly.
ATTN_ARMS = ("codesteer", "codesteer_auto", "rand_prior", "uniform_prior", "ast_prior", "combined",
             "codesteer_beta0")


class Level2Effect:
    """Effect gate for their level-2 steering: wraps `level2_post` in the llama backend's namespace and
    counts the applications whose OUTPUT differs from the INPUT attention. `runtime.steer_calls` is not
    that -- it increments before the beta check, so it was > 0 on the beta=0 identity runs."""

    def __init__(self, backend_module):
        self.mod = backend_module
        self.orig = backend_module.level2_post
        self.calls = 0; self.effective = 0; self.shift = 0.0

        def wrapped(attn_probs, prior, beta, eps=1e-6):
            out = self.orig(attn_probs, prior, beta, eps)
            self.calls += 1
            if out is not attn_probs:
                d = float((out.float() - attn_probs.float()).abs().sum(-1).mean().item())  # L1 shift per row
                if d > 0.0:
                    self.effective += 1; self.shift += d
            return out
        backend_module.level2_post = wrapped

    def reset(self):
        self.calls = 0; self.effective = 0; self.shift = 0.0

    def snapshot(self) -> dict:
        return {"l2_calls": self.calls, "l2_effective_calls": self.effective,
                "l2_mean_l1_shift": (self.shift / self.effective) if self.effective else 0.0}


RESIDUAL = ("swap_oracle", "foreign", "erasure", "combined", "ridge_map", "role_proto")
# H-R6 (nla/configs/ase_vectors.yaml): `ridge_map` / `role_proto` write PRE-FITTED vectors from
# ase_vectors.py (fit on the non-test pool, never on the written snippet); `prompt_types` is their
# prompting baseline (declared type + inferred role as text, no true names).
VECTOR_ARMS = ("ridge_map", "role_proto")
METHOD_RE = __import__("re").compile(r"\bs\.([A-Za-z_]\w*)\s*\(")


def _method_name(pack: dict) -> str | None:
    for c in pack.get("cases", []):
        m = METHOD_RE.search(c.get("expr", ""))
        if m:
            return m.group(1)
    return None


def _encode(tok, prompt: str):
    """Their runner tokenises with special tokens; `align` works without. Measure the offset and check
    that the two id sequences agree after it, so a tokenizer that does more than prepend BOS is caught."""
    full = tok(prompt)["input_ids"]
    bare = tok(prompt, add_special_tokens=False)["input_ids"]
    off = len(full) - len(bare)
    if off < 0 or full[off:] != bare:
        raise RuntimeError("special-token layout is not a pure prefix; positions cannot be offset")
    return full, off


def chat_wrap(tok, raw: str) -> str:
    """The chat-templated TEXT whose tokenisation equals apply_chat_template(tokenize=True) -- the ids our
    H-R1 harness generated from. The rendered template starts with a literal `<s>`; with the tokenizer's
    default `add_bos_token=True` the text path yields a double BOS, and stripping the literal instead
    changes the next token (`[INST]` -> `_[INST]`, id 29961 -> 518, because SentencePiece adds its dummy
    prefix space after a text start but not after a special token). So `install_chat_template` sets
    `add_bos_token=False` and the literal `<s>` carries the BOS; this gate proves the equality per prompt."""
    msgs = [{"role": "user", "content": raw}]
    ref = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False))
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if tok(text)["input_ids"] != ref:
        raise RuntimeError("chat-template ids gate: tokenizer(templated text) != apply_chat_template ids")
    return text


def install_chat_template(lm) -> None:
    """Replace the instance's `_build_prompt` (a staticmethod on their class) so every consumer -- run_llama,
    calibrate_head_subset, prepare_residual -- receives the templated string, and switch the instance's
    tokenizer to `add_bos_token=False` so the template's literal `<s>` is the only BOS (see chat_wrap).
    Bound per instance, never on the class, so nothing else that imports their module is affected.
    `_encode` then measures a BOS offset of 0 and `align` sees `<s>` as token 0 -- positions stay absolute."""
    base = type(lm)._build_prompt
    tok = lm.tokenizer
    tok.add_bos_token = False

    def build(code_snippet, *, instruction, language, answer_prefix=""):
        return chat_wrap(tok, base(code_snippet, instruction=instruction, language=language,
                                   answer_prefix=answer_prefix))
    lm._build_prompt = build


@torch.no_grad()
def _layer_states(lm, prompt: str, K: int) -> torch.Tensor:
    """[T, d] output of decoder layer K (== hidden_states[K+1]) for one prompt, no hooks active."""
    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
    # the BASE model (no lm_head): the decoder-layer hooks fire identically, and the full-sequence
    # vocab projection (T x 32k floats) is never materialised
    out = lm.model.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    return out.hidden_states[K + 1][0].float().cpu()


@torch.no_grad()
def _last_logits(lm, prompt: str) -> torch.Tensor:
    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
    h = lm.model.model(**enc, use_cache=False, return_dict=True).last_hidden_state[:, -1]
    return lm.model.lm_head(h)[0].float().cpu()


def prepare_residual(lm, packs_ren: list, packs_orig_path: str, manifest_path: str, K: int, tag: str):
    """Per snippet: aligned decoy positions in the RENAMED prompt (with BOS offset), the original span
    states h0 (per token and mean), the renamed prompt's own per-token state h1b, and the prompt text.
    Snippets that fail alignment are EXCLUDED with the reason recorded (never best-effort written)."""
    import counterfactual_eval as ce
    orig = {json.loads(l)["snippet"]: json.loads(l) for l in open(packs_orig_path) if l.strip()}
    man = {json.loads(l)["snippet"]: json.loads(l) for l in open(manifest_path) if l.strip()}
    tok = lm.tokenizer
    prep, excluded = {}, {}
    for rec in packs_ren:
        sid = rec["snippet"]
        if sid not in orig or "pack" not in orig[sid] or sid not in man:
            excluded[sid] = "no original pack / rename map"; continue
        o = orig[sid]
        op = lm._build_prompt(Path(o["java_path"]).read_text(),
                              instruction=ce.build_counterfactual_instruction(o["pack"]),
                              language="java", answer_prefix=ANSWER_PREFIX)
        rp = lm._build_prompt(Path(rec["java_path"]).read_text(),
                              instruction=ce.build_counterfactual_instruction(rec["pack"]),
                              language="java", answer_prefix=ANSWER_PREFIX)
        al = align(sid, op, rp, man[sid]["rename_map"], tok, method_name=_method_name(o["pack"]))
        if not al.ok:
            excluded[sid] = al.reason; continue
        _, off_o = _encode(tok, op)
        ids_r, off_r = _encode(tok, rp)
        H0 = _layer_states(lm, op, K)
        H1 = _layer_states(lm, rp, K)
        spans = []
        for sp in al.spans:
            ot = [t + off_o for t in sp.orig_tokens]
            rt = [t + off_r for t in sp.ren_tokens]
            spans.append({"name": sp.name, "decoy": sp.decoy, "ren_pos": rt,
                          "h0_tok": H0[ot], "h0_mean": H0[ot].mean(0),
                          "h1b_tok": H1[rt], "h1b_mean": H1[rt].mean(0)})
        prep[sid] = {"ren_prompt": rp, "prompt_len": len(ids_r), "spans": spans}
    print(f"{tag} residual prep: {len(prep)} aligned · {len(excluded)} excluded "
          f"({sum(len(v['spans']) for v in prep.values())} spans) · K={K}", flush=True)
    for k, v in excluded.items():
        print(f"{tag}   excluded {k}: {v}", flush=True)
    return prep, excluded


def residual_targets(arm: str, sid: str, prep: dict, rng: np.random.Generator,
                     vectors: dict | None = None) -> dict[int, torch.Tensor]:
    """{absolute renamed position: vector} for one snippet under one arm."""
    P = prep[sid]; T: dict[int, torch.Tensor] = {}
    if arm in VECTOR_ARMS:
        # one pre-fitted vector per span, keyed by the capture's span index; a missing span is a
        # harness fault (the pool capture and this run must have aligned identically)
        V = vectors[arm][sid]
        for j, sp in enumerate(P["spans"]):
            if j not in V:
                raise RuntimeError(f"{arm}: no vector for {sid} span {j} ({sp['decoy']})")
            for p_ in sp["ren_pos"]:
                T[p_] = V[j].clone()
        return T
    if arm in ("erasure", "combined"):
        # leave-one-ITEM-out over every span of every OTHER aligned snippet: direction AND magnitude
        D = [(s["h0_mean"] - s["h1b_mean"]).double() for o, v in prep.items() if o != sid for s in v["spans"]]
        D = torch.stack(D)
        E = D.mean(0); E = E / (E.norm() + 1e-12)
        dn = float(D.norm(dim=1).mean())
    if arm == "foreign":
        others = [s for o, v in prep.items() if o != sid for s in v["spans"]]
    for j, sp in enumerate(P["spans"]):
        pos = sp["ren_pos"]
        if arm == "swap_oracle":
            vecs = sp["h0_tok"] if len(pos) == sp["h0_tok"].shape[0] else sp["h0_mean"].expand(len(pos), -1)
        elif arm == "foreign":
            vecs = others[int(rng.integers(len(others)))]["h0_mean"].expand(len(pos), -1)
        else:  # erasure / combined
            vecs = (sp["h1b_mean"].double() + dn * E).float().expand(len(pos), -1)
        for p_, v in zip(pos, vecs):
            T[p_] = v.clone()
    return T


def self_gate(lm, prep: dict, replacer: PositionReplacer, tol: float, tag: str) -> bool:
    """Write each renamed prompt's OWN exact per-token state at its decoy positions: logits at the last
    prompt position must match the unhooked forward within `tol`, and the writer must have touched
    exactly the aligned positions. A `PositionReplacer` that is mis-cursored or mis-offset fails here."""
    worst = 0.0
    for sid, P in prep.items():
        T = {p_: v.clone() for sp in P["spans"] for p_, v in zip(sp["ren_pos"], sp["h1b_tok"])}
        replacer.set_targets(None)
        ref = _last_logits(lm, P["ren_prompt"])
        replacer.set_targets(T)
        got = _last_logits(lm, P["ren_prompt"])
        n_exp = len(T)
        d = float((got - ref).abs().max())
        worst = max(worst, d)
        if d > tol or replacer.n_positions_written != n_exp:
            print(f"{tag} SELF-GATE FAIL {sid}: max|dlogit|={d:.4f} (tol {tol}) · "
                  f"written {replacer.n_positions_written} vs expected {n_exp}", flush=True)
            replacer.set_targets(None); return False
    replacer.set_targets(None)
    print(f"{tag} self gate PASS on {len(prep)} snippets · worst max|dlogit| {worst:.4f} (tol {tol})", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="/scratch/juno/jvl210002/ase2026/LLM-Attention-Fixation_submission")
    ap.add_argument("--packs", required=True)
    ap.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--arm", required=True,
                    choices=("ridge_map", "role_proto", "prompt_types", "unsteered", "codesteer", "codesteer_auto", "rand_prior", "uniform_prior", "ast_prior",
                             "prompt", "swap_oracle", "foreign", "erasure", "combined",
                             "codesteer_beta0"))
    # residual (NLA-style) arms -- see RESIDUAL ARMS below
    ap.add_argument("--packs-orig", default=None, help="original-condition packs (same snippets)")
    ap.add_argument("--manifest", default="/scratch/juno/jvl210002/ase2026/rename_manifest.jsonl")
    ap.add_argument("--layer", type=int, default=7, help="residual write layer (pre-registered: 7)")
    ap.add_argument("--vectors", default=None, help="ase_vectors.py output (ridge_map / role_proto arms)")
    ap.add_argument("--vectors-config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ase_vectors.yaml"))
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--self-tol", type=float, default=0.05,
                    help="identity gate: max |dlogit| at the last prompt position when writing the exact own state")
    ap.add_argument("--chat-template", action="store_true",
                    help="H-R7: wrap the prompt in the tokenizer's chat template (ids == our H-R1 harness)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="H-R15: seeds generation sampling (torch), the `foreign` pick and their "
                         "`rand` prior. Defaults to the frozen SEED, so omitting it reproduces every "
                         "run banked before 2026-09-17 bit-for-bit as far as the RNG is concerned.")
    ap.add_argument("--greedy", action="store_true",
                    help="H-R18: argmax decoding instead of their T=0.7 sampler. Their own "
                         "`_sample_next_token` takes do_sample=False and returns argmax, so this is "
                         "THEIR code path, not ours. H-R15b showed the pipeline is exactly reproducible, "
                         "so greedy removes decoding noise entirely and one run per case suffices -- but "
                         "it is a DEVIATION from the paper's protocol, whose Pass@k presupposes sampling "
                         "(under greedy pass@1 == pass@2 == pass@3), so it complements the sampled runs "
                         "rather than replacing them.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=2.0)
    # Their models.DEFAULT_CACHE_DIR is the RELATIVE path ".cache/models", which is empty on this host,
    # so with HF_HUB_OFFLINE=1 their build() raises LocalEntryNotFoundError even though the model is
    # cached. Point it at the real hub cache. (My own gate runner used the default HF location and so
    # loaded the same model without trouble -- the failure was their default, not the cache.)
    ap.add_argument("--cache-dir", default=None,
                    help="HF cache dir for THEIR runner (defaults to $HF_HOME/hub)")
    args = ap.parse_args()
    if args.greedy and args.runs != 1:
        # every greedy run would be byte-identical (H-R15b: fixed seed -> per-item agreement 1.0000),
        # so 3 runs cost 3x the GPU for zero extra information. Refuse rather than silently spend it.
        print(f"{TAG} REFUSED: --greedy with --runs {args.runs}; greedy runs are identical, use --runs 1")
        return 2

    low = args.model_id.lower()
    if any(v in low for v in ("qwen", "deepseek", "yi-", "glm", "internlm", "baichuan")):
        print(f"{TAG} REFUSED: {args.model_id} is a Chinese model."); return 2

    R = Path(args.artifact); sys.path.insert(0, str(R))
    import counterfactual_eval as ce                     # their case/parse/score code
    from models import SteeredCausalLM                   # their runner + steering integration
    from steering import SteeringConfig                  # their config

    packs = [json.loads(l) for l in open(args.packs) if l.strip()]
    packs = [p for p in packs if "pack" in p]
    print(f"{TAG} arm={args.arm} · model={args.model_id} · {len(packs)} snippets · "
          f"{sum(len(p['pack']['cases']) for p in packs)} cases", flush=True)

    cache_dir = args.cache_dir or (os.path.join(os.environ["HF_HOME"], "hub")
                                   if os.environ.get("HF_HOME") else None)
    lm = SteeredCausalLM()
    lm.config(model_name=args.model_id, max_new_tokens=args.max_new_tokens,
              temperature=TEMP, top_p=TOP_P, key_scope="prompt", cache_dir=cache_dir)
    print(f"{TAG} cache_dir={cache_dir}", flush=True)
    sc = None
    if args.arm in ATTN_ARMS:
        cfg = dict(PAPER_CFG)
        cfg["prior"] = {"codesteer": "slice_hybrid", "codesteer_auto": "slice_hybrid", "rand_prior": "rand",
                        "combined": "slice_hybrid", "uniform_prior": "uniform", "ast_prior": "ast",
                        "codesteer_beta0": "slice_hybrid"}[args.arm]
        if cfg["prior"] == "rand":
            cfg["rand_seed"] = args.seed
        if args.arm == "codesteer_auto":
            cfg["head_subset_mode"] = "auto"       # paper Eq. 10: calibrated top-k heads per layer
        if args.arm == "codesteer_beta0":
            cfg["beta_post"] = 0.0             # deliberate: see ATTN_ARMS note. The assert below guards
        else:                                  # against the fault re-entering the REAL steering arms.
            assert cfg["beta_post"] > 0.0, "beta_post=0 makes level-2 steering the identity (2026-09-14 fault)"
        sc = SteeringConfig(**cfg)
        lm.set_steering_config(sc)
        print(f"{TAG} steering ON: {cfg}", flush=True)
    else:
        print(f"{TAG} steering OFF (control arm shares the same runner)", flush=True)
    lm.build()
    if args.chat_template:
        install_chat_template(lm)
        probe = lm._build_prompt("int x = 1;", instruction="say hi", language="java", answer_prefix=ANSWER_PREFIX)
        print(f"{TAG} chat template ON: {probe[:40]!r} ... {probe[-24:]!r} ({len(lm.tokenizer(probe)['input_ids'])} ids)", flush=True)
    else:
        print(f"{TAG} chat template OFF (their raw prompt)", flush=True)
    # H-R15: the ONLY knob that makes a replicate a replicate. Their sampler draws with
    # torch.multinomial from the global RNG (models.py `_sample_next_token`), so this seeds every
    # generated token. NOTE it does not make a run deterministic -- 2026-08-29 measured 0.8333
    # per-item agreement on this cluster with deterministic kernels PINNED -- which is precisely
    # what the same-seed replicate in H-R15 is there to quantify for this pipeline.
    torch.manual_seed(args.seed)
    steered = args.arm in ATTN_ARMS
    effect = None
    if steered:
        # last-N-layer window, exactly as obfuscation/main.py sets it after build()
        n_layers = int(lm.model.config.num_hidden_layers)
        sc.steer_layer_start = max(0, n_layers - STEER_LAST_N_LAYERS)
        sc.steer_layer_end = n_layers - 1
        print(f"{TAG} steer window: layers {sc.steer_layer_start}..{sc.steer_layer_end} of {n_layers} · "
              f"beta_post={sc.beta_post} · head_subset_mode={sc.head_subset_mode}", flush=True)
        import steering.backends.llama_backend as llama_backend
        effect = Level2Effect(llama_backend)

    prep, replacer, rrng, vectors = None, None, None, None
    if args.arm in VECTOR_ARMS:
        if not args.vectors:
            print(f"{TAG} {args.arm} needs --vectors"); return 2
        vb = torch.load(args.vectors, weights_only=False)
        if int(vb["layer"]) != int(args.layer):
            print(f"{TAG} FATAL: vectors fitted at layer {vb['layer']}, run asks {args.layer}"); return 3
        if args.arm == "ridge_map" and not vb.get("gate_pass", False):
            print(f"{TAG} REFUSED: ridge_map failed its pre-GPU gate (MAP-LEARNS-NOTHING); not run"); return 2
        vectors = vb["arms"]
        print(f"{TAG} vectors {args.vectors}: {args.arm} for {len(vectors[args.arm])} snippets · "
              f"selected {vb.get('selected')}", flush=True)
    if args.arm in RESIDUAL:
        if not args.packs_orig:
            print(f"{TAG} residual arm needs --packs-orig"); return 2
        prep, excluded = prepare_residual(lm, packs, args.packs_orig, args.manifest, args.layer, TAG)
        if not prep:
            print(f"{TAG} FATAL: no snippet aligned"); return 3
        replacer = PositionReplacer(lm.model, args.layer, beta=1.0)   # beta=1 for the exact-state gate
        if not self_gate(lm, prep, replacer, args.self_tol, TAG):
            print(f"{TAG} FATAL: identity gate failed -> harness fault, nothing reportable"); return 3
        replacer.set_beta(args.beta)
        rrng = np.random.default_rng(args.seed)
        packs = [p for p in packs if p["snippet"] in prep]     # excluded snippets are not written
        json.dump({"layer": args.layer, "beta": args.beta, "seed": args.seed,
                   "decode": "greedy" if args.greedy else f"sample T{TEMP} p{TOP_P}",
                   "aligned": sorted(prep), "excluded": excluded,
                   "n_spans": {k: len(v["spans"]) for k, v in prep.items()}},
                  open(str(args.out) + ".residual_meta.json", "w"), indent=1)

    # The prompting baseline is the only arm that alters the prompt; it prepends the same warning our
    # own accuracy work used, so the comparison to CLAUDE.md's mandatory prompting baseline is like-for-like.
    WARN = ("Note: the identifiers in this code may be misleading. Reason about what the code actually "
            "does, not what the names suggest.\n\n")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["snippet"] for l in open(out) if l.strip()} if out.exists() else set()
    sink = open(out, "a")
    t0 = time.time()
    for i, rec in enumerate(packs, 1):
        sid = rec["snippet"]
        if sid in done:
            continue
        pack = rec["pack"]; cases = pack.get("cases", [])
        if not cases:
            continue
        code = Path(rec["java_path"]).read_text()
        instruction = ce.build_counterfactual_instruction(pack)
        if args.arm == "prompt":
            instruction = WARN + instruction
        elif args.arm == "prompt_types":
            # the same declared-type + inferred-role facts the H-R6 vectors use, given as text
            import ase_roles
            man = {json.loads(l)["snippet"]: json.loads(l) for l in open(args.manifest) if l.strip()}
            tags = ase_roles.tag_snippet(rec["java_path"], man[sid]["rename_map"])
            tmpl = __import__("yaml").safe_load(open(args.vectors_config))["prompt_types"]["line"]
            lines = "\n".join(t.prompt_line(tmpl) for t in tags.values())
            instruction = WARN + "Declared types and roles of the identifiers in question:\n" + lines + "\n\n" + instruction
        case_ids = [c["case_id"] for c in cases]
        truth = {c["case_id"]: bool(c["expected_bool"]) for c in cases}

        per_run = []
        calib = None
        if args.arm == "codesteer_auto":
            # per snippet, as obfuscation/main.py does (vocab_tokens=[] there too); 3 short sampled runs
            # with betas forced to 0 select top-4 heads/layer in the window and store the mask in-config
            calib = lm.calibrate_head_subset(code_snippet=code, instruction=instruction, language="java",
                                             vocab_tokens=[], snippet_name=sid)
            if int(calib.get("active_total", 0)) <= 0:
                print(f"{TAG} FATAL: auto calibration selected no heads on {sid}: {calib}", flush=True)
                sink.close(); return 3
        if replacer is not None:
            replacer.set_targets(residual_targets(args.arm, sid, prep, rrng, vectors))
        for _ in range(args.runs):
            if replacer is not None:
                replacer.reset()          # absolute cursor restarts with every generate() call
            if effect is not None:
                effect.reset()
            # record_layers/record_attention OFF (their `--record-layers off`). Their run_llama
            # defaults both to True, which sets generate(output_attentions=True); under transformers
            # 5.12 the per-step attentions come back EMPTY and attn_postprocess raises
            # "No layer vectors to pool." (job 399325). The steering hooks live inside the forward
            # pass and never touch the recorder, so switching it off changes nothing about steering.
            res = lm.run_llama(code, instruction, language="java",
                               answer_prefix=ANSWER_PREFIX, do_sample=not args.greedy,
                               record_layers=False, record_attention=False)
            text = res.get("generated_text", "") or ""
            dbg = res.get("steering_debug") or {}
            eff = effect.snapshot() if effect is not None else {}
            # The effect gate reads in BOTH directions, because the identity floor's whole job is to be
            # the identity: `codesteer_beta0` must report 0 effective calls (a nonzero count would mean
            # beta_post=0 does NOT neutralise level-2, and the raw-prompt floor was mislabelled), every
            # other steered arm must report > 0 (job 403742/405553, 2026-09-15).
            want_effect = steered and args.arm != "codesteer_beta0"
            n_eff = int(eff.get("l2_effective_calls", 0))
            if want_effect and n_eff <= 0:
                # A steered arm whose level-2 op never changed an attention row would be an unsteered
                # arm wearing a label (exactly what the beta=0 runs were).
                print(f"{TAG} FATAL: arm={args.arm} but no effective level-2 call on {sid}: {eff} {dbg}",
                      flush=True)
                sink.close(); return 3
            if steered and not want_effect and n_eff > 0:
                print(f"{TAG} FATAL: identity floor arm={args.arm} changed {n_eff} attention rows on "
                      f"{sid}: {eff} {dbg}", flush=True)
                sink.close(); return 3
            pred, meta = ce.parse_predicted_labels(text, case_ids, strict_json=True)
            per_run.append({"n_parsed": len(pred), "parse_mode": meta.get("mode"),
                            "pred": pred, "reply_chars": len(text),
                            "steer_calls": int(dbg.get("steer_calls", 0)),
                            "steer_enabled": bool(dbg.get("enabled", False)),
                            **eff,
                            "head_mask_active_total": int(dbg.get("head_mask_active_total", 0)),
                            "n_positions_written": (replacer.n_positions_written if replacer else 0)})
            if replacer is not None and replacer.n_positions_written != len(replacer._targets):
                print(f"{TAG} FATAL: wrote {replacer.n_positions_written} of {len(replacer._targets)} "
                      f"positions on {sid}", flush=True)
                sink.close(); return 3
        passk = {}
        for k in range(1, args.runs + 1):
            passk[f"pass@{k}"] = float(np.mean(
                [any(per_run[j]["pred"].get(c) == truth[c] for j in range(k)) for c in case_ids]))
        row = {"snippet": sid, "arm": args.arm, "model": args.model_id, "n_cases": len(cases),
               "chat_template": bool(args.chat_template),
               "parsed_frac": float(np.mean([r["n_parsed"] / len(cases) for r in per_run])),
               **passk, "runs": per_run}
        if calib is not None:
            row["head_calibration"] = {k: calib.get(k) for k in
                                       ("active_total", "layers_with_heads", "topk_per_layer",
                                        "calib_runs_valid", "layer_start", "layer_end")}
            # the calibrated head SET itself ({layer: [head ids]}), kept for the head comparison against
            # the NLA-carrying heads (H-R5, CHECKLIST.md) -- their selection is observational:
            # top-k heads per layer by agree[h] = sum_k P_last[h,k] * prior[k] at the first decode step
            row["codesteer_heads"] = dict(dbg.get("head_subset_selected_heads") or {})
        sink.write(json.dumps(row) + "\n"); sink.flush()
        print(f"{TAG} {i}/{len(packs)} {sid} " +
              " ".join(f"P@{k}={passk[f'pass@{k}']:.3f}" for k in range(1, args.runs + 1)) +
              f" parsed={row['parsed_frac']:.2f} · {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {i}; re-run to resume", flush=True); break
    sink.close()

    rows = [json.loads(l) for l in open(out) if l.strip()]
    tc = sum(r["n_cases"] for r in rows)
    print(f"\n{TAG} === {args.arm} · {len(rows)} snippets · {tc} cases ===")
    for k in range(1, args.runs + 1):
        cw = sum(r[f"pass@{k}"] * r["n_cases"] for r in rows) / tc if tc else float("nan")
        print(f"{TAG}   Pass@{k} (case-weighted) {cw:.4f}")
    print(f"{TAG}   parsed {np.mean([r['parsed_frac'] for r in rows]):.4f}")
    print(f"{TAG}   mean steer_calls/run {np.mean([x['steer_calls'] for r in rows for x in r['runs']]):.1f}")
    if any("l2_effective_calls" in x for r in rows for x in r["runs"]):
        xs = [x for r in rows for x in r["runs"] if "l2_effective_calls" in x]
        print(f"{TAG}   mean effective level-2 calls/run {np.mean([x['l2_effective_calls'] for x in xs]):.1f} · "
              f"mean L1 attention shift {np.mean([x['l2_mean_l1_shift'] for x in xs]):.4f}")
    print(f"{TAG}   CodeLlama-7B reference: original 0.7995 · renamed_unsteered 0.6866 (H-R1c)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
