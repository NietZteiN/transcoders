"""Pretrained dictionary (SAE / transcoder) loading + encoding.

Registry: configs/dictionaries.yaml (repo ids + revisions pinned, HfApi-verified). Formats
differ per release family — this module hides that behind one interface:

    dic = load_dictionary(dictionary_spec, layer=15, device="cpu")
    feats = dic.encode(hidden)           # [n, d_model] -> [n, d_sae] sparse features
    recon = dic.decode(feats)            # [n, d_sae]   -> [n, d_model]

Implemented loaders:
  * llama_scope (SAELens registry): release llama_scope_lxr_{8x,32x}, sae_id l<L>r_<W>x —
    stock SAELens handles the fnlp alias + norm scaling. Used for E1 on Llama-3.1-8B.
  * qwen_scope (plain torch .pt dicts): layer<L>.sae.pt with W_enc/W_dec/b_enc/b_dec, TopK.
Other families (LXTC / R1 / Gemma npz / CLTs) raise with a pointer to their loading notes
in dictionaries.yaml — add loaders when their experiments come up.

Provenance: record `identity()` (repo, revision, layer, loader) in every run manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LoadedDictionary:
    sae: Any                    # underlying object (SAELens SAE or QwenScopeSAE)
    repo: str
    revision: str | None
    layer: int
    loader: str
    d_in: int
    d_sae: int

    def encode(self, hidden):
        return self.sae.encode(hidden)

    def decode(self, feats):
        return self.sae.decode(feats)

    def identity(self) -> dict:
        return {"repo": self.repo, "revision": self.revision, "layer": self.layer,
                "loader": self.loader, "d_in": self.d_in, "d_sae": self.d_sae}


class QwenScopeSAE:
    """Qwen-Scope residual SAE: plain torch dict {W_enc,(d_sae,d_in) W_dec,(d_in,d_sae) b_enc,b_dec}, TopK k=50."""

    def __init__(self, state: dict, k: int = 50):
        import torch

        self.W_enc = state["W_enc"].float()      # (d_sae, d_in)
        self.W_dec = state["W_dec"].float()      # (d_in, d_sae)
        self.b_enc = state["b_enc"].float()
        self.b_dec = state["b_dec"].float()
        self.k = k
        self._torch = torch

    def encode(self, hidden):
        pre = (hidden.float() - self.b_dec) @ self.W_enc.T + self.b_enc
        topk = self._torch.topk(pre, self.k, dim=-1)
        out = self._torch.zeros_like(pre)
        return out.scatter_(-1, topk.indices, self._torch.relu(topk.values))

    def decode(self, feats):
        return feats @ self.W_dec.T + self.b_dec


def load_dictionary(spec: dict, layer: int, device: str = "cpu") -> LoadedDictionary:
    """Load one layer's dictionary per the registry spec (configs/dictionaries.yaml entry)."""
    repo = spec.get("repo")
    if not repo:
        raise ValueError("dictionary spec has no repo — registry entry unpinned?")

    if "LXR" in repo:                                       # Llama Scope residual, via SAELens
        width = "32x" if "32x" in repo else "8x"
        release, sae_id = f"llama_scope_lxr_{width}", f"l{layer}r_{width}"
        from sae_lens import SAE

        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        return LoadedDictionary(sae=sae, repo=repo, revision=spec.get("revision"), layer=layer,
                                loader=f"sae_lens:{release}/{sae_id}",
                                d_in=int(sae.cfg.d_in), d_sae=int(sae.cfg.d_sae))

    if repo.startswith("Qwen/SAE-Res-"):                    # Qwen-Scope plain .pt
        import torch
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, f"layer{layer}.sae.pt", revision=spec.get("revision"))
        state = torch.load(path, map_location=device, weights_only=True)
        k = 100 if "L0_100" in repo else 50
        sae = QwenScopeSAE(state, k=k)
        return LoadedDictionary(sae=sae, repo=repo, revision=spec.get("revision"), layer=layer,
                                loader="qwen_scope_pt",
                                d_in=sae.W_enc.shape[1], d_sae=sae.W_enc.shape[0])

    raise NotImplementedError(
        f"No loader for {repo!r} yet — see its `notes` in configs/dictionaries.yaml "
        f"(LXTC/R1 need OpenMOSS lm_sae; Gemma Scope is npz; CLTs load via circuit-tracer)."
    )
