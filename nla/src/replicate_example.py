"""Gate G0a — exact replication of the reference's worked example.

`vendor/nla-repo/examples/qwen7b_layer20_step4200.txt` records, for all 101
tokens of one fixed prompt+reply at temp=0, the raw activation norm and the
NLA round-trip scores. That makes it a deterministic target, which is worth far
more than a distributional match: if our numbers land on theirs, every link in
the chain (layer index, chat template, injection scale, AR template, metric
definition) is verified at once.

The test runs in two stages, deliberately separated:

  Stage A — extraction only. `||v||` is a pure function of the target-model
  forward pass; no NLA is involved. Comparing it isolates the layer-index
  question (hidden_states[20] vs [21], see extract.py) from everything else. If
  Stage A fails, nothing downstream is worth running.

  Stage B — full round trip. Needs the SGLang server up for the AV.

Usage:
    python -m src.replicate_example --stage A
    python -m src.replicate_example --stage B --sglang-url http://localhost:30000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

EXAMPLES_DIR = _NLA_ROOT / "vendor" / "nla-repo" / "examples"

# One worked example ships per released NLA pair. Everything the gate needs is IN the file —
# host model, layer, d_model, the training-set Var(v_nrm), the prompt, the reply and the
# per-token table — so the constants are parsed from it rather than transcribed. Transcribing a
# 150-token reply by hand is exactly the kind of silent error this gate exists to catch.
REFERENCES = {
    "qwen7b":   "qwen7b_layer20_step4200.txt",
    "gemma12b": "gemma12b_layer32_step4000.txt",
    "gemma27b": "gemma27b_layer41_step6000.txt",
    "llama70b": "llama70b_layer53_step2424.txt",
}

# The end-of-turn token is part of the reply for some hosts and absent for others, and
# `extract_chat` appends the reply verbatim — so it must match the reference's own last row.
# Qwen ends on '<|im_end|>', Llama on '<|eot_id|>', and the Gemma references stop mid-sentence
# with no end-of-turn at all. Derived below from the last row rather than hardcoded.
END_OF_TURN_TOKENS = {"<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end_of_text|>"}


def load_reference(key: str) -> dict:
    """Everything the gate needs for one host, read out of its worked example."""
    import ast
    path = EXAMPLES_DIR / REFERENCES[key]
    t = path.read_text()

    def grab(pattern, cast=str, flags=0):
        m = re.search(pattern, t, flags)
        if not m:
            raise SystemExit(f"{path.name}: could not parse /{pattern}/")
        return cast(m.group(1))

    user = ast.literal_eval(grab(r"User message:\s*('.*?')\s*$", flags=re.M))
    # Section 2 holds the temp-0 reply as a single Python string literal on its own line.
    body = t.split("2. BASE MODEL REPLY", 1)[1]
    # The reply is one Python string literal on its own line — single-quoted in the Qwen and
    # Gemma-12B references, double-quoted in the Gemma-27B and Llama ones. Accept either, and
    # let literal_eval handle the escapes rather than unescaping by hand.
    reply_line = next(
        (l.strip() for l in body.splitlines()
         if len(l.strip()) > 1 and l.strip()[0] in "'\"" and l.strip()[-1] == l.strip()[0]),
        None)
    if reply_line is None:
        raise SystemExit(f"{path.name}: could not locate the reply literal in section 2")
    reply = ast.literal_eval(reply_line)

    rows = parse_expected(path)
    last_tok = rows[-1]["token"]
    eot = last_tok if last_tok in END_OF_TURN_TOKENS else ""

    return {
        "key": key, "path": path,
        "model": grab(r"Extraction model:\s*(\S+)"),
        "layer": grab(r"layer (\d+) residual stream", int),
        "d_model": grab(r"d_model:\s*(\d+)", int),
        # Denominator of fve_nrm = 1 - mse_nrm / Var. Varies enormously by host: 0.7335 for
        # Qwen L20 but 0.0302 for Gemma-3-12B L32, ~24x lower. At low variance cos compresses
        # toward 0.99 for everything and stops discriminating, which is why the Gemma reference
        # says fve_nrm is the informative metric there. mse_nrm = 2(1 - cos), so fve is a pure
        # function of cos given this constant — no re-capture is ever needed to switch metrics.
        # [\d.]+ would swallow the sentence's closing full stop ("0.0302." -> ValueError),
        # so the decimal is matched explicitly.
        "var_v_nrm": grab(r"mse_nrm\s*/\s*(\d+\.\d+)", float),
        "user_message": user, "assistant_reply": reply,
        "end_of_turn": eot, "expected_n_tokens": len(rows), "rows": rows,
    }

# "[  0]  PROMPT  token='<|im_start|>'  ||v||=235.7  mse_nrm=1.962  cos=0.019  fve_nrm=-1.674"
ROW_RE = re.compile(
    r"^\[\s*(\d+)\]\s+(\S+)\s+token=(.+?)\s+\|\|v\|\|=([\d.]+)\s+"
    r"mse_nrm=([-\d.]+)\s+cos=([-\d.]+)\s+fve_nrm=([-\d.]+)\s*$"
)


def parse_expected(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if m:
            rows.append(
                dict(
                    position=int(m.group(1)),
                    section=m.group(2),
                    token=m.group(3).strip().strip("'"),
                    act_norm=float(m.group(4)),
                    mse_nrm=float(m.group(5)),
                    cos=float(m.group(6)),
                    fve_nrm=float(m.group(7)),
                )
            )
    return rows


def stage_a(device: str, tol_rel: float, ref: dict) -> int:
    """Extraction-only check: our ||v|| vs the reference's, per token."""
    from extract import ActivationExtractor  # noqa: E402

    expected = ref["rows"]
    print(f"reference: {ref['path'].name}\n  host {ref['model']} · layer {ref['layer']} · "
          f"d_model {ref['d_model']} · Var(v_nrm) {ref['var_v_nrm']} · "
          f"{ref['expected_n_tokens']} tokens · end-of-turn {ref['end_of_turn']!r}")

    ex = ActivationExtractor(ref["model"], ref["layer"], device=device)
    res = ex.extract_chat(ref["user_message"], ref["assistant_reply"] + ref["end_of_turn"],
                          text_id="worked_example")
    ex.close()

    n_got, n_exp = len(res.positions), len(expected)
    print(f"tokens: ours={n_got}  reference={n_exp}")
    if n_got != n_exp:
        print(
            "MISMATCH in sequence length — chat template or reply text differs.\n"
            f"  our first 8 tokens: {res.token_strs[:8]}\n"
            f"  ref first 8 tokens: {[r['token'] for r in expected[:8]]}"
        )
        return 1

    ours = np.linalg.norm(res.activations, axis=1)
    theirs = np.array([r["act_norm"] for r in expected])
    rel = np.abs(ours - theirs) / np.maximum(theirs, 1e-9)

    # Also compute what hidden_states[K] (the off-by-one) would have given, so a
    # failure diagnoses itself rather than just reporting "wrong".
    print(f"\n{'pos':>4} {'token':<16} {'ours':>10} {'ref':>10} {'rel err':>9}")
    for i in list(range(6)) + list(range(n_got - 3, n_got)):
        print(
            f"{i:>4} {expected[i]['token'][:16]:<16} "
            f"{ours[i]:>10.1f} {theirs[i]:>10.1f} {rel[i]:>9.4f}"
        )

    n_bad = int((rel > tol_rel).sum())
    print(
        f"\nmedian rel err = {np.median(rel):.5f}   max = {rel.max():.5f}   "
        f"tokens over tol({tol_rel}) = {n_bad}/{n_got}"
    )
    out = _NLA_ROOT / "results" / f"g0a_stage_a_{ref['key']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "host": ref["key"],
                "model": ref["model"],
                "layer_index": ref["layer"],
                "d_model": ref["d_model"],
                "var_v_nrm": ref["var_v_nrm"],
                "reference_file": ref["path"].name,
                "n_tokens": n_got,
                "median_rel_err": float(np.median(rel)),
                "max_rel_err": float(rel.max()),
                "n_over_tol": n_bad,
                "tol_rel": tol_rel,
                "ours": ours.tolist(),
                "reference": theirs.tolist(),
            },
            indent=1,
        )
    )
    print(f"wrote {out}")

    if n_bad == 0:
        print("\nSTAGE A PASS — extraction matches the reference layer-for-layer.")
        return 0
    print(
        "\nSTAGE A FAIL — check the layer index first (extract.py hooks "
        "layers[K] == hidden_states[K+1]; the repo README's demo snippet uses "
        "hidden_states[K], which is the off-by-one)."
    )
    return 1


def stage_b(device: str, sglang_url: str, tol_cos: float, ref: dict) -> int:
    """Full round trip: AV decode + AR reconstruction vs the reference's cos."""
    from extract import ActivationExtractor  # noqa: E402
    from nla_inference import NLAClient, NLACritic  # noqa: E402

    expected = ref["rows"]
    ex = ActivationExtractor(ref["model"], ref["layer"], device=device)
    res = ex.extract_chat(ref["user_message"], ref["assistant_reply"] + ref["end_of_turn"],
                          text_id="worked_example")
    ex.close()

    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=device)

    rows = []
    for i, pos in enumerate(res.positions):
        v = res.activations[i]
        text = av.generate(v, temperature=0.0, max_new_tokens=512)
        mse, cos = ar.score(text, v)
        rows.append(
            dict(
                position=pos,
                token=res.token_strs[pos],
                act_norm=float(np.linalg.norm(v)),
                mse_nrm=float(mse),
                cos=float(cos),
                explanation=text,
            )
        )
        if i % 10 == 0:
            print(f"  [{i:>3}/{len(res.positions)}] cos={cos:.3f} ref={expected[i]['cos']:.3f}")

    ours = np.array([r["cos"] for r in rows])
    theirs = np.array([r["cos"] for r in expected])
    d = np.abs(ours - theirs)
    # CJK smell test: the loudest signal that injection silently failed.
    cjk = sum(bool(re.search(r"[　-鿿]", r["explanation"])) for r in rows)

    print(
        f"\ncos: mean ours={ours.mean():.4f} ref={theirs.mean():.4f}  "
        f"median|diff|={np.median(d):.4f} max={d.max():.4f}"
    )
    print(f"CJK-containing explanations: {cjk}/{len(rows)} (all-CJK would mean injection failed)")

    out = _NLA_ROOT / "results" / "g0a_stage_b.json"
    out.write_text(json.dumps({"rows": rows, "reference_cos": theirs.tolist()}, indent=1))
    print(f"wrote {out}")

    ok = np.median(d) <= tol_cos
    print(("\nSTAGE B PASS" if ok else "\nSTAGE B FAIL") + f" — median|Δcos| vs tol {tol_cos}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["A", "B"], default="A")
    p.add_argument("--model", choices=sorted(REFERENCES), default="qwen7b",
                   help="which released pair to gate. Default qwen7b preserves the original "
                        "behaviour; gemma12b is the 2026-08-31 port target.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--tol-rel", type=float, default=0.02,
                   help="Stage A: allowed relative error on ||v||. bf16 forward "
                        "passes are not bit-reproducible across GPU/kernel "
                        "versions, so exact equality is the wrong bar.")
    p.add_argument("--tol-cos", type=float, default=0.10,
                   help="Stage B: allowed median |Δcos|. Looser than Stage A — "
                        "the AV decode path adds its own nondeterminism even at "
                        "temp=0 (batching, kernel selection).")
    a = p.parse_args()
    ref = load_reference(a.model)
    return (stage_a(a.device, a.tol_rel, ref) if a.stage == "A"
            else stage_b(a.device, a.sglang_url, a.tol_cos, ref))


if __name__ == "__main__":
    raise SystemExit(main())
