"""transcoders/ — Instrument 3 (SAE + transcoder feature & circuit analysis).

Phase-0 scaffold. Import-light on purpose: nothing here imports torch at module load,
so that CUDA_VISIBLE_DEVICES can be pinned *before* torch is first imported
(see ../CLAUDE.md §1 and src/gpu.py). Heavy imports live inside functions.
"""

__all__ = ["seedutil", "gpu", "provenance", "configs", "data"]
