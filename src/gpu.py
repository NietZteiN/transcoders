"""GPU selection for a shared, scheduler-free host (../CLAUDE.md §1).

No SLURM. Pick an *idle* GPU from `nvidia-smi` and pin CUDA_VISIBLE_DEVICES **before**
torch is imported. This module is deliberately torch-free so it can run first.

Typical use at the top of an entrypoint, before any `import torch`:

    from src import gpu
    gpu.pin(gpu.pick_free_gpus(1))   # or gpu.pin([0]) to force a specific index
    import torch                      # now sees only the pinned device(s)
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GpuStat:
    index: int
    mem_used_mb: int
    util_pct: int

    def is_idle(self, max_mem_used_mb: int, max_util_pct: int) -> bool:
        return self.mem_used_mb <= max_mem_used_mb and self.util_pct <= max_util_pct


def query() -> list[GpuStat]:
    """Parse `nvidia-smi` for per-GPU memory + utilization. Empty list if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    stats = []
    for line in out.strip().splitlines():
        idx, mem, util = (p.strip() for p in line.split(","))
        stats.append(GpuStat(int(idx), int(mem), int(util)))
    return stats


def pick_free_gpus(
    n: int = 1,
    max_mem_used_mb: int = 1000,
    max_util_pct: int = 5,
) -> list[int]:
    """Return indices of up to `n` idle GPUs (most-free first).

    Raises RuntimeError if fewer than `n` are idle — deliberately: on a shared box you
    wait rather than stomp on someone else's job (../CLAUDE.md §1).
    """
    stats = query()
    if not stats:
        raise RuntimeError("nvidia-smi unavailable — cannot verify a GPU is idle; refusing to guess.")
    idle = [s for s in stats if s.is_idle(max_mem_used_mb, max_util_pct)]
    idle.sort(key=lambda s: (s.mem_used_mb, s.util_pct))
    if len(idle) < n:
        busy = ", ".join(f"gpu{s.index}({s.mem_used_mb}MB,{s.util_pct}%)" for s in stats)
        raise RuntimeError(f"Need {n} idle GPU(s), found {len(idle)}. Current: {busy}. Wait — no scheduler here.")
    return [s.index for s in idle[:n]]


def pin(indices: list[int]) -> str:
    """Set CUDA_VISIBLE_DEVICES. Call BEFORE importing torch. Returns the value set."""
    value = ",".join(str(i) for i in indices)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return value


if __name__ == "__main__":
    # `python -m src.gpu` — quick status print.
    for s in query():
        flag = "IDLE" if s.is_idle(1000, 5) else "busy"
        print(f"gpu{s.index}: {s.mem_used_mb:>6} MB used, {s.util_pct:>3}% util  [{flag}]")
