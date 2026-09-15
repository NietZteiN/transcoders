"""CPU unit tests for fsdp_util — the partition/scaling contracts the GPU equivalence test relies on."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fsdp_util import (DistCtx, all_reduce_sum, rank_microbatches, reduce_replicated_grads,  # noqa: E402
                       scale_for_world)


def test_single_gpu_ctx_is_identity():
    """world==1 must leave every loop exactly as the proven single-GPU path."""
    c = DistCtx()
    assert not c.enabled and c.is_main and c.world == 1 and c.device == "cuda"
    assert rank_microbatches(7, c) == list(range(7))
    loss = torch.tensor(3.0)
    assert scale_for_world(loss, c) is loss          # no tensor op at all when disabled
    assert all_reduce_sum(1.5, 2.5, ctx=c) == (1.5, 2.5)


@pytest.mark.parametrize("n_micro,world", [(8, 2), (4, 2), (16, 2), (8, 4), (4, 4), (12, 4)])
def test_partition_is_exact_cover(n_micro, world):
    """Union over ranks == range(n_micro), pairwise disjoint, sizes within one.

    A partition that dropped or duplicated a microbatch would change the effective batch
    composition without changing the loss scale, i.e. it would be invisible in the loss curve."""
    parts = [rank_microbatches(n_micro, DistCtx(rank=r, world=world, enabled=True)) for r in range(world)]
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(n_micro))
    assert len(flat) == len(set(flat))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_scale_for_world_cancels_mean_reduction():
    """THE NORMALIZER TRAP: per-rank loss x world, then FSDP's mean over ranks, must reproduce the
    single-GPU gradient. Simulated here on a 1-parameter model with an explicit mean."""
    torch.manual_seed(0)
    w_single = torch.nn.Parameter(torch.tensor([2.0]))
    xs = [torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([5.0]), torch.tensor([7.0])]
    N = 4.0
    (sum((w_single * x).sum() for x in xs) / N).backward()
    g_single = w_single.grad.clone()

    world = 2
    grads = []
    for r in range(world):
        ctx = DistCtx(rank=r, world=world, enabled=True)
        w = torch.nn.Parameter(torch.tensor([2.0]))
        mine = [xs[i] for i in rank_microbatches(len(xs), ctx)]
        scale_for_world(sum((w * x).sum() for x in mine) / N, ctx).backward()
        grads.append(w.grad.clone())
    g_fsdp = torch.stack(grads).mean(0)                      # what FSDP reduces to
    torch.testing.assert_close(g_fsdp, g_single)

    # And the control: WITHOUT the scaling it is wrong by exactly 1/world.
    bad = []
    for r in range(world):
        ctx = DistCtx(rank=r, world=world, enabled=True)
        w = torch.nn.Parameter(torch.tensor([2.0]))
        (sum((w * x).sum() for x in [xs[i] for i in rank_microbatches(len(xs), ctx)]) / N).backward()
        bad.append(w.grad.clone())
    torch.testing.assert_close(torch.stack(bad).mean(0), g_single / world)


@pytest.mark.parametrize("n_micro,world", [(7, 2), (1, 2), (9, 4), (3, 4)])
def test_uneven_split_raises_instead_of_hanging(n_micro, world):
    """An uneven microbatch split means ranks issue different numbers of FSDP collectives, whose
    real-world symptom is a 10-minute NCCL watchdog hang. It must raise instead."""
    with pytest.raises(RuntimeError, match="different numbers of FSDP collectives"):
        rank_microbatches(n_micro, DistCtx(rank=0, world=world, enabled=True))


def test_replicated_grad_reduction_count_is_data_independent(monkeypatch):
    """REGRESSION (job 389569): the collective count must not depend on which grads happen to exist.

    The first version skipped params with `.grad is None`; ranks process different microbatches, so
    the counts could diverge and deadlock on a 4-byte all-reduce. Every param must be reduced."""
    import torch.distributed as dist
    calls = []
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None: calls.append(tuple(t.shape)), raising=False)
    monkeypatch.setattr(dist, "ReduceOp", type("R", (), {"AVG": "avg"}), raising=False)
    ps = [torch.nn.Parameter(torch.zeros(3)) for _ in range(3)]
    ps[1].grad = torch.ones(3)                      # only ONE has a gradient
    reduce_replicated_grads(ps, DistCtx(rank=0, world=2, enabled=True))
    assert len(calls) == 3, f"expected one collective per param, got {len(calls)}"
    assert all(p.grad is not None for p in ps)
    assert torch.equal(ps[0].grad, torch.zeros(3))  # materialised, not skipped
