"""Wiring test for the residual arms of ase_steer_run: alignment + BOS offset + capture + identity gate +
target construction, on a random-init 2-layer Llama with the REAL CodeLlama tokenizer (offline cache).
Their generate loop is not exercised here; the cluster smoke does that."""
import json, os, sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
A = Path("/scratch/juno/jvl210002/ase2026")
ART = A / "LLM-Attention-Fixation_submission"
pytestmark = pytest.mark.skipif(not (A / "smoke_packs.jsonl").exists() or not ART.exists(),
                                reason="ASE artifact not on this host")


class _LM:
    """The four attributes of their SteeredCausalLM that the residual path touches."""
    def __init__(self, model, tok):
        self.model, self.tokenizer = model, tok

    @staticmethod
    def _build_prompt(code_snippet, *, instruction, language, answer_prefix=""):
        return f"{instruction}\n\n```{language}\n{code_snippet}\n```{answer_prefix or ''}"


@pytest.fixture(scope="module")
def lm():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).eval()
    return _LM(m, tok)


@pytest.fixture(scope="module")
def prep(lm):
    sys.path.insert(0, str(ART))
    import ase_steer_run as R
    packs = [json.loads(l) for l in open(A / "smoke_packs.jsonl") if l.strip()]
    packs = [p for p in packs if "pack" in p]
    P, exc = R.prepare_residual(lm, packs, str(A / "gate/packs_orig_subset.jsonl"),
                                str(A / "rename_manifest.jsonl"), K=0, tag="[T]")   # K=0 of 2 so a write propagates to the last position
    return R, P, exc


def test_alignment_and_offset(lm, prep):
    R, P, exc = prep
    assert P, exc
    for sid, v in P.items():
        ids = lm.tokenizer(v["ren_prompt"])["input_ids"]
        assert v["prompt_len"] == len(ids)
        for sp in v["spans"]:
            # every decoy position decodes to a piece of the decoy name, i.e. the BOS offset is right
            assert all(0 < p < len(ids) for p in sp["ren_pos"])
            txt = lm.tokenizer.decode([ids[p] for p in sp["ren_pos"]])
            assert txt.strip().replace(" ", "") in sp["decoy"] or sp["decoy"] in txt.replace(" ", ""), (sp["decoy"], txt)
            assert sp["h1b_tok"].shape == (len(sp["ren_pos"]), 64)


def test_self_gate_passes_and_counts(lm, prep):
    R, P, _ = prep
    rep = R.PositionReplacer(lm.model, 0, beta=1.0)
    try:
        assert R.self_gate(lm, P, rep, tol=1e-3, tag="[T]")
    finally:
        rep.close()


def test_self_gate_catches_wrong_offset(lm, prep):
    R, P, _ = prep
    import copy
    bad = copy.deepcopy(P)
    for v in bad.values():
        for sp in v["spans"]:
            sp["ren_pos"] = [p - 1 for p in sp["ren_pos"]]     # shifted by one: writes the wrong tokens
    rep = R.PositionReplacer(lm.model, 0, beta=1.0)
    try:
        assert not R.self_gate(lm, bad, rep, tol=1e-3, tag="[T]")
    finally:
        rep.close()


def test_targets_per_arm(lm, prep):
    R, P, _ = prep
    rng = np.random.default_rng(0)
    sid = sorted(P)[0]
    n_pos = sum(len(sp["ren_pos"]) for sp in P[sid]["spans"])
    T_sw = R.residual_targets("swap_oracle", sid, P, rng)
    T_fo = R.residual_targets("foreign", sid, P, rng)
    T_er = R.residual_targets("erasure", sid, P, rng)
    assert len(T_sw) == len(T_fo) == len(T_er) == n_pos
    # erasure = h1b_mean + dn * E  with dn, E built from the OTHER snippet only
    others = [(s["h0_mean"] - s["h1b_mean"]).double() for o, v in P.items() if o != sid for s in v["spans"]]
    D = torch.stack(others); E = D.mean(0); E = E / E.norm(); dn = D.norm(dim=1).mean()
    sp0 = P[sid]["spans"][0]
    exp = (sp0["h1b_mean"].double() + dn * E).float()
    assert torch.allclose(T_er[sp0["ren_pos"][0]], exp, atol=1e-5)
    # swap_oracle on an equal-length span is the exact per-token original state
    for sp in P[sid]["spans"]:
        if len(sp["ren_pos"]) == sp["h0_tok"].shape[0]:
            assert torch.equal(T_sw[sp["ren_pos"][0]], sp["h0_tok"][0])
    # foreign never equals own original state
    assert not torch.allclose(T_fo[sp0["ren_pos"][0]], sp0["h0_mean"])
