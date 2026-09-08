"""H-W35 arm construction: the null arms must write DIFFERENT content at the SAME positions.

Every failure mode here is silent. A sibling draw that lands on the span's own vector, or a foreign
draw that lands in the same item, turns the specificity control into a copy of the arm it is meant
to discriminate against — and it would still produce a plausible number.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_heads import item_targets, ARMS, NULL_ARMS  # noqa: E402

D = 8


def _spans(sid="s0", n=3, base=0):
    return [{"snippet_id": sid, "span_i": i, "positions": [10 * (base + i), 10 * (base + i) + 1],
             "vec": base + i} for i in range(n)]


def _vecs(n):
    h0 = np.arange(n * D, dtype=np.float32).reshape(n, D) + 1.0
    return h0, h0 * -1.0


def test_sibling_never_uses_its_own_vector():
    spans = _spans(n=4)
    h0, c3 = _vecs(4)
    tg = item_targets(spans, h0, c3, ("P_patch", "N_sibling"))
    for r in spans:
        for q in r["positions"]:
            own = torch.from_numpy(h0[r["vec"]])
            assert torch.equal(tg["P_patch"][q], own)
            assert not torch.equal(tg["N_sibling"][q], own), f"sibling reused own vector at {q}"
            # and it must be some other span's vector of this item
            assert any(torch.equal(tg["N_sibling"][q], torch.from_numpy(h0[s["vec"]]))
                       for s in spans if s["vec"] != r["vec"])


def test_single_span_item_has_no_sibling_arm():
    """One span => no sibling exists. The arm must be ABSENT, never silently the span itself."""
    spans = _spans(n=1)
    h0, c3 = _vecs(1)
    tg = item_targets(spans, h0, c3, ("P_patch", "N_sibling"))
    assert "P_patch" in tg
    assert "N_sibling" not in tg, "a single-span item must not produce a sibling arm"


def test_foreign_never_draws_from_its_own_item():
    mine, theirs = _spans("mine", 2, base=0), _spans("theirs", 2, base=2)
    h0, c3 = _vecs(4)
    tg = item_targets(mine, h0, c3, ("N_foreign",), pool=mine + theirs)
    foreign_vecs = {2, 3}
    for r in mine:
        for q in r["positions"]:
            assert any(torch.equal(tg["N_foreign"][q], torch.from_numpy(h0[v])) for v in foreign_vecs), \
                "foreign draw came from the item's own vectors"


def test_foreign_absent_when_pool_has_only_this_item():
    spans = _spans("only", 2)
    h0, c3 = _vecs(2)
    tg = item_targets(spans, h0, c3, ("N_foreign",), pool=spans)
    assert "N_foreign" not in tg, "with no other item in the pool the arm must be absent"


def test_random_is_unit_norm_and_not_the_content():
    spans = _spans(n=2)
    h0, c3 = _vecs(2)
    tg = item_targets(spans, h0, c3, ("N_random",))
    for r in spans:
        for q in r["positions"]:
            v = tg["N_random"][q]
            assert abs(float(v.norm()) - 1.0) < 1e-5
            assert not torch.equal(v, torch.from_numpy(h0[r["vec"]]))


def test_draws_are_reproducible_and_position_independent():
    spans = _spans(n=3)
    h0, c3 = _vecs(3)
    a = item_targets(spans, h0, c3, NULL_ARMS, pool=spans + _spans("other", 2, base=0))
    b = item_targets(spans, h0, c3, NULL_ARMS, pool=spans + _spans("other", 2, base=0))
    for arm in a:
        for q in a[arm]:
            assert torch.equal(a[arm][q], b[arm][q]), f"{arm} not reproducible at {q}"
    # every position of one span carries the SAME drawn vector (the draw is per span, not per token)
    for r in spans:
        qs = r["positions"]
        assert torch.equal(a["N_sibling"][qs[0]], a["N_sibling"][qs[1]])


def test_all_arms_cover_exactly_the_same_positions():
    """The contrast is about content, so the arms must be position-matched — arm_guard.paired
    refuses them otherwise, and that refusal should never be reachable by construction."""
    spans = _spans(n=3)
    h0, c3 = _vecs(3)
    tg = item_targets(spans, h0, c3, ARMS + NULL_ARMS, pool=spans + _spans("other", 2, base=0))
    want = {q for r in spans for q in r["positions"]}
    for arm, t in tg.items():
        assert set(t) == want, f"{arm} wrote {len(t)} positions, expected {len(want)}"


# ── H-W39 dose arms ──────────────────────────────────────────────────────────

def test_dose_arms_interpolate_toward_the_same_foreign_vector_as_N_foreign():
    """alpha=1 must be the BANKED N_foreign arm, not an independent draw — otherwise the dose
    curve's endpoint is a different experiment from the one it is being compared against."""
    from nla_heads import DOSE_ARMS
    mine, theirs = _spans("mine", 2, base=0), _spans("theirs", 2, base=2)
    h0, c3 = _vecs(4)
    pool = mine + theirs
    tg = item_targets(mine, h0, c3, ("N_foreign",) + DOSE_ARMS, pool=pool)
    for r in mine:
        q = r["positions"][0]
        own = h0[r["vec"]]
        far = tg["N_foreign"][q].numpy()
        # D_75 must lie much closer to `far` than D_25 does
        d25, d75 = tg["D_25"][q].numpy(), tg["D_75"][q].numpy()
        n = lambda v: v / (np.linalg.norm(v) + 1e-12)
        assert float(n(d75) @ n(far)) > float(n(d25) @ n(far))
        assert float(n(d25) @ n(own)) > float(n(d75) @ n(own))


def test_dose_vectors_are_unit_norm_and_cosine_is_monotone():
    from nla_heads import DOSE_ARMS
    spans = _spans("mine", 2)
    h0, c3 = _vecs(4)
    tg = item_targets(spans, h0, c3, DOSE_ARMS, pool=spans + _spans("other", 2, base=2))
    for arm in DOSE_ARMS:
        for v in tg[arm].values():
            assert abs(float(v.norm()) - 1.0) < 1e-5
    cos = item_targets.last_cosines
    assert cos["D_25"] > cos["D_50"] > cos["D_75"], f"dose not monotone: {cos}"


def test_dose_arms_absent_without_a_foreign_pool():
    """No other item => no interpolation target. Absent, never silently the own state."""
    from nla_heads import DOSE_ARMS
    spans = _spans("only", 2)
    h0, c3 = _vecs(2)
    tg = item_targets(spans, h0, c3, DOSE_ARMS, pool=spans)
    assert not any(a in tg for a in DOSE_ARMS)
