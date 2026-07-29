"""Team classification from player crops — who wears which jersey.

The problem: the player detector tells us WHERE everyone is, but not WHICH
team they play for. footlab needs that split (attack_mask / defend_mask)
before any tactical analytics make sense.

The approach (ported from roboflow/sports, explained for learning):

    1. EMBED — every player crop (the image cut out of the bounding box) is
       passed through SigLIP, a vision transformer pretrained on image/text
       pairs. It turns each crop into a 768-dim vector ("embedding") that
       captures visual similarity: same-jersey crops land near each other.

    2. REDUCE — 768 dimensions is too many for clean clustering, so UMAP
       compresses the embeddings to 3 dimensions while preserving the
       neighbourhood structure (points that were close stay close).

    3. CLUSTER — KMeans with k=2 splits the cloud of crops into two groups.
       In a football match, the dominant visual signal across player crops
       is jersey colour, so the two clusters almost always = the two teams.

    4. GOALKEEPERS — keepers wear a third colour, so they fall outside both
       clusters. We assign each keeper to the team whose centroid (mean
       pitch position) is nearest — a keeper stands near his own team.

Usage pattern (mirrors roboflow):
    classifier = TeamClassifier(device="cpu")
    classifier.fit(crops_from_many_frames)   # once per match/video
    team_ids = classifier.predict(crops_from_one_frame)  # 0 or 1 per crop

Known limitations (worth knowing while learning):
    * KMeans clusters are unordered: "team 0" might be the home side in one
      video and the away side in another. The LABEL is arbitrary; only the
      SPLIT is meaningful.
    * If both teams wear similar colours, clustering degrades. Check the
      cluster sizes — a 22/2 split on 24 crops means something went wrong.
    * Referees must be excluded before fitting, or they can form a third
      cluster and confuse KMeans into splitting teams wrongly.

Reference:
    roboflow/sports — sports/common/team.py (MIT license)
    McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and
    Projection." arXiv:1802.03426
"""

from __future__ import annotations

from typing import Iterable, List, TypeVar

import numpy as np

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Iterable[List[V]]:
    """Yield lists of at most `batch_size` items from `sequence`.

    Splitting crops into batches lets the GPU/CPU process N images per
    forward pass instead of one at a time — much faster, same result.
    """
    batch_size = max(batch_size, 1)
    batch: List[V] = []
    for element in sequence:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(element)
    if batch:
        yield batch


class TeamClassifier:
    """Two-team classifier: SigLIP features → UMAP(3D) → KMeans(k=2).

    Lifecycle:
        fit(crops)     — learn the two team clusters from a large, varied
                         set of player crops (e.g. collected across many
                         frames of the same match). Called ONCE per video.
        predict(crops) — assign each new crop to team 0 or 1.

    Attributes after fit:
        cluster_model  — fitted sklearn KMeans (inspect .labels_ on the
                         training projections for a sanity check).
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        import torch  # local import: keeps module importable without torch
        import umap
        from sklearn.cluster import KMeans
        from transformers import AutoProcessor, SiglipVisionModel

        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Turn each BGR player crop into a 768-dim SigLIP embedding.

        We mean-pool over the patch dimension of last_hidden_state, giving
        one vector per crop. Done under torch.no_grad() — inference only,
        no gradients needed, saves memory.
        """
        import supervision as sv
        import torch

        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        data: List[np.ndarray] = []
        with torch.no_grad():
            for batch in create_batches(crops_pil, self.batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """Learn the two team clusters from a set of player crops."""
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Assign each crop to team 0 or 1. Empty input → empty array."""
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


def resolve_goalkeepers_team_id(
    players_xy: np.ndarray,
    players_team_id: np.ndarray,
    goalkeepers_xy: np.ndarray,
) -> np.ndarray:
    """Assign each goalkeeper to the team whose pitch centroid is nearest.

    Args:
        players_xy:      (N, 2) pitch positions of outfield players (meters).
        players_team_id: (N,) team labels for those players (0 or 1).
        goalkeepers_xy:  (M, 2) pitch positions of goalkeepers (meters).

    Returns:
        (M,) team labels for the goalkeepers.

    Why centroid distance: a goalkeeper almost always stands within his own
    team's defensive shape, so his own team's mean position is the closer
    of the two. Simple, no extra model, right in >95% of broadcast frames.
    """
    if len(goalkeepers_xy) == 0:
        return np.array([])
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    assignments = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        assignments.append(0 if dist_0 < dist_1 else 1)
    return np.array(assignments)
