"""CPU tests for nla/src/nla_train.py on a tiny random Gemma-3 text model + the real tokenizer."""
import math
import sys
from pathlib import Path

import pytest
import torch

_PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ / "nla" / "src"))
import nla_train as nt  # noqa: E402
from nla_inference import load_nla_config  # noqa: E402

CKPT = _PROJ / "data/nla/ml/gemma4b/host_text"
pytestmark = pytest.mark.skipif(not (CKPT / "tokenizer.json").exists(), reason="text checkpoint absent")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CKPT)


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=8,
                           pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    return Gemma3ForCausalLM(cfg).eval()


def test_ceil_2sf():
    assert nt.ceil_2sf(123.4) == 130
    assert nt.ceil_2sf(0.0123) == pytest.approx(0.013)
    assert nt.ceil_2sf(2445.0) == 2500
    assert nt.ceil_2sf(50.0) == 50


def test_sample_positions_deterministic():
    ids = list(range(1000)); special = {0, 1, 2}
    a = nt.sample_positions(ids, 10, special, "corpus:s:1", 42, 50)
    b = nt.sample_positions(ids, 10, special, "corpus:s:1", 42, 50)
    c = nt.sample_positions(ids, 10, special, "corpus:s:2", 42, 50)
    assert a == b and a != c and len(a) == 10 and min(a) >= 50 and len(set(a)) == 10
    assert nt.sample_positions(list(range(40)), 10, special, "x", 42, 50) == []


def test_tokens_block(tok):
    tb = nt.tokens_block(tok)
    assert tb["injection_token_id"] == 246566
    assert tb["injection_left_neighbor_id"] == 236813 and tb["injection_right_neighbor_id"] == 954
    assert tb["critic_suffix_ids"] == [1005, 236813, 655, 6011, 236813]


def test_lr_schedule():
    assert nt.lr_at(0, 100, 1.0, 0.1, 10) == pytest.approx(0.1)
    assert nt.lr_at(9, 100, 1.0, 0.1, 10) == pytest.approx(1.0)
    assert nt.lr_at(100, 100, 1.0, 0.1, 10) == pytest.approx(0.1)
    assert 0.1 < nt.lr_at(55, 100, 1.0, 0.1, 10) < 1.0


def test_clean_explanation():
    raw = "blah <analysis>\n- **First feature** here\n\n2. second one\n</analysis> trailing"
    out = nt.clean_explanation(raw, 2)
    assert out == "First feature here\n\nsecond one"
    assert nt.clean_explanation("<analysis>only one</analysis>", 2) is None
    assert nt.clean_explanation("no tags", 2) is None


def test_av_batcher_masks(tok):
    rows = [{"explanation": "a cat\n\nsits"}, {"explanation": "x"}]
    b = nt.AVBatcher(tok, rows, 320)
    vecs = torch.randn(2, 32)
    batch = b.collate([0, 1], vecs)
    P = len(b.prompt)
    assert (batch["labels"][:, :P] == -100).all()
    for i in range(2):
        n = len(b.resp[i])
        assert batch["labels"][i, P:P + n].tolist() == b.resp[i]
        assert b.resp[i][-1] == b.eot
        assert (batch["labels"][i, P + n:] == -100).all()
        assert batch["attention_mask"][i].sum() == P + n
    assert b.inj_pos == [93]


def test_embedding_injector(tiny, tok):
    rows = [{"explanation": "a\n\nb"}, {"explanation": "c\n\nd"}]
    b = nt.AVBatcher(tok, rows, 320)
    vecs = torch.randn(2, 32) * 100
    batch = b.collate([0, 1], vecs)
    inj = nt.EmbeddingInjector(tiny.model.embed_tokens, b.inj_id, b.left, b.right, 7.0)
    with torch.no_grad():
        base = tiny.model.embed_tokens(batch["input_ids"])          # vectors None → untouched
        inj.vectors = vecs
        out = tiny.model.embed_tokens(batch["input_ids"])
    assert inj.n_written == 2
    p = b.inj_pos[0]
    assert torch.allclose(out[:, p].norm(dim=-1), torch.tensor([7.0, 7.0]))
    assert torch.allclose(out[:, p] / 7.0, vecs / vecs.norm(dim=-1, keepdim=True), atol=1e-5)
    mask = torch.ones(out.shape[1], dtype=torch.bool); mask[p] = False
    assert torch.equal(out[:, mask], base[:, mask])
    inj.close()


def test_av_loss_matches_full_ce(tiny, tok):
    rows = [{"explanation": "a\n\nb"}, {"explanation": "longer text\n\nsecond"}]
    b = nt.AVBatcher(tok, rows, 320)
    batch = b.collate([0, 1], torch.randn(2, 32))
    ref = torch.zeros(())
    with torch.no_grad():
        tot, n = nt.av_loss(tiny, batch, "cpu", chunk=5)
        for i in range(2):    # one row at a time: full 262k-vocab logits for a batch exceed the login-node ulimit
            L = int(batch["attention_mask"][i].sum())
            logits = tiny(input_ids=batch["input_ids"][i:i + 1, :L]).logits[0]
            ref = ref + torch.nn.functional.cross_entropy(logits[:-1], batch["labels"][i, 1:L],
                                                          ignore_index=-100, reduction="sum")
            del logits
    assert n == int((batch["labels"] != -100).sum())
    assert torch.allclose(tot, ref, rtol=1e-4)


def test_ar_loss_is_two_one_minus_cos():
    p, g = torch.randn(4, 32), torch.randn(4, 32)
    cos = torch.nn.functional.cosine_similarity(p, g, dim=-1)
    assert torch.allclose(nt.ar_loss(p, g, math.sqrt(32)), 2 * (1 - cos), atol=1e-5)
    assert nt.fve(g, g, math.sqrt(32)) == pytest.approx(1.0)


def test_ar_batcher_suffix(tok):
    suf = nt.tokens_block(tok)["critic_suffix_ids"]
    b = nt.ARBatcher(tok, [{"explanation": "hello world\n\nfoo"}], suf)
    assert b.n_bad_suffix == 0 and b.ids[0][0] == tok.bos_token_id
    batch = b.collate([0])
    assert batch["last"][0] == len(b.ids[0]) - 1


def test_sidecar_roundtrip(tmp_path, tok):
    cfg = {"host": {"d_model": 2560, "model_id": "google/gemma-3-4b-it", "n_layers": 34}}
    tok.save_pretrained(tmp_path)
    nt.write_sidecar(tmp_path, "av", cfg, 12, tok, 2500.0, {"lr": 1e-5})
    c = load_nla_config(tmp_path, tok)
    assert c.injection_scale == 2500.0 and c.d_model == 2560 and c.injection_token_id == 246566
    nt.write_sidecar(tmp_path, "ar", cfg, 12, tok, None, {})
    import yaml
    m = yaml.safe_load((tmp_path / "nla_meta.yaml").read_text())
    assert m["role"] == "ar" and m["critic"]["extraction_layer_index"] == 12
    assert m["extraction"]["mse_scale"] == pytest.approx(math.sqrt(2560))
