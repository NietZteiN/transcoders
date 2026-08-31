"""Export per-snippet steering directions for the composed runner (B5 / E6).

The composed experiment runs in the `codesteer` environment, where CodeSteer's attention
backend works; the NLA lives in `nla-mi`. Rather than make those two coexist in one
interpreter — they pin transformers 4.57.1 and 5.12.1 respectively — the NLA side is reduced
to what the other side actually needs: **one direction vector per snippet**, computed here and
written as `.npz`. The runner then loads plain numpy and needs no NLA code at all.

Four families, matching the pre-registered ladder:

  v1      AR(gloss of the TRUE identifiers) - AR(gloss of the DECOY identifiers)
          The NLA-derived belief correction. Needs the reconstructor and nothing else.
  v3      mean(h_clean - h_obf) over all OTHER snippets — a contrastive task vector that uses
          no NLA whatsoever. This is the bar v1 must clear; the leakage guard (excluding the
          item itself) is not optional, because fitting a direction on the item it is applied
          to inflates every steering number and is invisible afterwards.
  v4      h_clean - h_obf for THIS snippet. An oracle: it uses the clean program, so it is not
          deployable and is reported as a ceiling, never as a method.
  random  norm-matched random direction, seeded per snippet.

ON PROMPT CONTEXT, a deliberate choice worth stating. The activations here are taken at the
final token of `deception_capture.PREAMBLE + code` — the same prompt shape B3 read. The
composed runner will apply these directions under the artifact's own counterfactual prompt,
which additionally lists the case expressions. So the direction is *estimated* in a slightly
different context from where it is *applied*.

That is deliberate. The alternative — estimating under the artifact's prompt — would make each
direction depend on which assertions that snippet happened to draw, turning a property of the
program pair into a property of the test set. A difference-of-activations direction should
describe "clean minus obfuscated", not "clean minus obfuscated, given these twelve assertions".
It also keeps these vectors in the same space as the 6,171 banked B3 reads.

Usage (nla-mi env):
    python -m src.export_directions --dataset humaneval --out data/nla/b5_directions
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from deception_capture import PREAMBLE, load_triples  # noqa: E402
from extract import ActivationExtractor, sha256_file  # noqa: E402

LAYER_INDEX = 20
SEED = 20260724


def gloss(terms: list[str]) -> str:
    """Same template the banked stimuli and steer_run.py use — kept identical so a v1 vector
    built here is the same object as the one B4 tested, not a near-relative."""
    return "code whose identifiers are about " + (", ".join(terms) if terms else "unnamed values")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="humaneval", choices=["humaneval", "cruxeval"])
    ap.add_argument("--out", default=str(_PROJ / "data/nla/b5_directions"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ar-device", default="cpu",
                    help="cpu keeps ~11 GB off a card that may be shared with a grid worker")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    from nla_inference import NLACritic  # noqa: E402
    import torch

    rng = random.Random(a.seed)
    triples = load_triples(a.dataset, a.limit, rng)
    print(f"[dirs] {len(triples)} C0/C2 pairs from {a.dataset}", flush=True)

    ex = ActivationExtractor(a.model, LAYER_INDEX, device=a.device)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=a.ar_device)

    def last_tok(code: str) -> np.ndarray:
        """Activation at the final prompt token — the site the composed runner steers."""
        res = ex.extract_chat(PREAMBLE + code, None, text_id="x")
        return res.activations[-1].astype(np.float32)

    v1: dict[str, np.ndarray] = {}
    v4: dict[str, np.ndarray] = {}
    diffs: dict[str, np.ndarray] = {}
    skipped_v1: list[str] = []

    for i, t in enumerate(triples, 1):
        sid = t["snippet"]
        h_clean = last_tok(t["src"]["C0"])
        h_obf = last_tok(t["src"]["C2"])
        d = h_clean - h_obf
        diffs[sid] = d
        v4[sid] = d.copy()

        true_names, decoy_names = t["original_names"], t["misleading_names"]
        if not true_names or not decoy_names:
            skipped_v1.append(sid)
        else:
            g_true = ar.reconstruct(gloss(true_names)).float().cpu().numpy()
            g_dec = ar.reconstruct(gloss(decoy_names)).float().cpu().numpy()
            v1[sid] = (g_true - g_dec).astype(np.float32)

        if i % 20 == 0:
            print(f"[dirs] {i}/{len(triples)}", flush=True)

    # v3: held-out mean. Excluding the item itself is the leakage guard; without it every
    # v3 number is inflated by the item's own contribution and nothing downstream reveals it.
    ids = sorted(diffs)
    total = np.sum([diffs[s] for s in ids], axis=0)
    v3 = {s: ((total - diffs[s]) / max(len(ids) - 1, 1)).astype(np.float32) for s in ids}

    # random: norm-matched to v1 where it exists, else to v4, so "random" is not trivially
    # distinguishable from the real directions by magnitude alone.
    rnd = {}
    for s in ids:
        ref = v1.get(s, v4[s])
        g = np.random.RandomState(abs(hash(s)) % (2**31)).randn(ref.shape[0]).astype(np.float32)
        rnd[s] = (g / np.linalg.norm(g) * np.linalg.norm(ref)).astype(np.float32)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, dd in (("v1", v1), ("v3", v3), ("v4", v4), ("random", rnd)):
        np.savez(out / f"{name}.npz", **dd)
        print(f"[dirs] wrote {name}.npz  ({len(dd)} snippets)")

    (out / "manifest.json").write_text(json.dumps({
        "dataset": a.dataset, "model": a.model, "layer": LAYER_INDEX, "seed": a.seed,
        "n_pairs": len(triples), "n_v1": len(v1), "n_v3": len(v3), "n_v4": len(v4),
        "skipped_v1_no_rename_map": skipped_v1,
        "prompt_context": "deception_capture.PREAMBLE + code (B3 read context)",
        "script_sha256": sha256_file(__file__),
        "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    ex.close()
    print(f"[dirs] done. v1={len(v1)} v3={len(v3)} v4={len(v4)} skipped_v1={len(skipped_v1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
