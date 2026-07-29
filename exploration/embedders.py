"""Pluggable embedder registry for exploration studies.

One interface, many models: every embedder takes a list of BGR player crops
and returns an (N, D) embedding matrix. Swapping models = changing a string.

Models compared here (all open weights, all frozen feature extractors —
none of them are trained by us):

    siglip_v1   google/siglip-base-patch16-224        (~203M params, ~400MB)
    siglip_v2   google/siglip2-base-patch16-224       (~203M params, ~400MB)
    dinov2      facebook/dinov2-small                 (~22M params, ~90MB)

Why these three:
    * SigLIP v1 is the current pipeline choice (contrastive image-text).
    * SigLIP v2 is its 2025 successor — same interface, better retrieval.
    * DINOv2 is self-supervised (no text), purpose-built for visual
      similarity — and 10x smaller. The interesting challenger.

Each entry exposes:
    name        — registry key
    hf_path     — HuggingFace model path
    description — one line for tables/plots
    load()      — returns an Embedder instance with .embed(crops)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


class Embedder:
    """Uniform wrapper: crops (BGR uint8) -> (N, D) float32 embeddings."""

    def __init__(self, hf_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(hf_path)
        # SigLIP's AutoModel is the full image-TEXT model and demands
        # input_ids; we only want the vision tower. DINOv2/CLIP-family
        # models work through plain AutoModel.
        if "siglip" in hf_path.lower():
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(hf_path).to(device)
        else:
            self.model = AutoModel.from_pretrained(hf_path).to(device)
        self.model.eval()
        self._torch = torch

    def embed(self, crops: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Mean-pooled last_hidden_state per crop, L2-normalized.

        L2 normalization puts every vector on the unit sphere, so Euclidean
        distance downstream behaves like cosine similarity — the natural
        metric for "do these crops look alike".
        """
        import supervision as sv

        embs: List[np.ndarray] = []
        with self._torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = [sv.cv2_to_pillow(c) for c in crops[i : i + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embs.append(pooled)
        out = np.concatenate(embs).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    hf_path: str
    description: str


EMBEDDER_REGISTRY: dict[str, EmbedderSpec] = {
    "siglip_v1": EmbedderSpec(
        name="siglip_v1",
        hf_path="google/siglip-base-patch16-224",
        description="SigLIP v1 base (image-text contrastive, ~203M params)",
    ),
    "siglip_v2": EmbedderSpec(
        name="siglip_v2",
        hf_path="google/siglip2-base-patch16-224",
        description="SigLIP 2 base (2025 successor, ~203M params)",
    ),
    "dinov2": EmbedderSpec(
        name="dinov2",
        hf_path="facebook/dinov2-small",
        description="DINOv2 small (self-supervised visual similarity, ~22M params)",
    ),
}


def load_embedder(name: str, device: str = "cpu") -> tuple[EmbedderSpec, Embedder]:
    """Instantiate a registered embedder by name."""
    if name not in EMBEDDER_REGISTRY:
        raise KeyError(
            f"Unknown embedder '{name}'. Registered: {sorted(EMBEDDER_REGISTRY)}"
        )
    spec = EMBEDDER_REGISTRY[name]
    return spec, Embedder(spec.hf_path, device=device)
