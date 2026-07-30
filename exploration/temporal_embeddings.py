"""Temporal embedding analysis (Phase 2) — how the model's *understanding*
of the scene evolves over time.

Phase 1 answered "what did the model detect per frame". Phase 2 answers a
deeper question: is the STRUCTURE of what the model sees stable over time?

Three instruments:

1. SHARED-PROJECTION MAP
   UMAP and t-SNE are NOT comparable across separate runs — each fit creates
   its own arbitrary coordinate system. So we embed crops from EVERY sampled
   frame, fit ONE projection on the pooled embeddings, then plot each frame's
   points in that shared space. Only then can you watch clusters tighten,
   drift, or smear across time. Rendered as small-multiples (one mini-scatter
   per time window) + a drift-colored master scatter.

2. CLUSTER QUALITY OVER TIME
   Per sampled frame: silhouette score of that frame's 2-cluster partition
   (in the SHARED 3D UMAP space, using the video-level KMeans labels). Dips
   mark moments the teams became less separable — similar-looking kits under
   changed light, heavy occlusion, camera cut. Cross-reference with the
   Phase 1 dashboard: dips should coincide with scene events.

3. CENTROID DRIFT
   Per frame: distance between the two team centroids (separability) and
   each cluster's mean intra-cluster distance (compactness), computed in
   embedding space (cosine distance on L2-normalized vectors = 1 - cosine
   similarity). No KMeans noise — pure geometry of the embedding cloud.

Outputs feed exploration/temporal_plots.py renderers and the CSV in
data/exploration/temporal/<video>/embedding_stats.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detection_pipeline.infer import detect_players_and_ball
from detection_pipeline.teams_pipeline import crop_detections


@dataclass
class EmbeddingFrameStats:
    """Per-frame embedding-structure metrics (one CSV row)."""

    frame_id: int
    time_s: float
    n_crops: int
    silhouette: float              # this frame's partition, shared 3D space
    centroid_distance: float       # between team centroids (cosine dist)
    intra_dist_team0: float        # mean within-cluster cosine distance
    intra_dist_team1: float
    team_sizes: str                # "n0/n1" for quick scanning

    @classmethod
    def fieldnames(cls) -> list[str]:
        return list(cls.__dataclass_fields__.keys())


def _cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance for L2-normalized rows: D = 1 - X @ X.T."""
    sim = x @ x.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def collect_embeddings_over_time(
    source: str,
    player_model: object,
    embedder: object,               # exploration.embedders.Embedder
    class_map: dict[str, int],
    player_conf: float,
    ball_conf: float,
    stride: int = 15,
    max_frames: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk the video, embedding outfield crops per sampled frame.

    Returns:
        embeddings: (M, D) pooled over all frames (L2-normalized by embedder)
        frame_idx:  (M,) which sampled frame each crop came from
        frame_ids:  (F,) the sampled frame numbers
        times:      (F,) seconds per sampled frame
    """
    cap = cv2.VideoCapture(source)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ids = list(range(0, max(n_frames, 1), stride))
    if max_frames is not None:
        ids = ids[:max_frames]
    if verbose:
        print(f"collect_embeddings_over_time: {len(ids)} frames (stride={stride})")

    all_emb, all_frame = [], []
    kept_ids, kept_times = [], []
    for i, fid in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        players, _ = detect_players_and_ball(
            frame, player_model,
            class_map=class_map, player_conf=player_conf, ball_conf=ball_conf,
        )
        outfield = [d for d in players if d.class_name == "player"]
        if len(outfield) < 4:
            continue
        crops = crop_detections(frame, outfield)
        emb = embedder.embed(crops)
        all_emb.append(emb)
        all_frame.append(np.full(len(emb), len(kept_ids)))
        kept_ids.append(fid)
        kept_times.append(fid / fps)
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)} frames embedded...")
    cap.release()

    return (
        np.concatenate(all_emb),
        np.concatenate(all_frame).astype(int),
        np.array(kept_ids),
        np.array(kept_times),
    )


def analyze_embeddings(
    embeddings: np.ndarray,
    frame_idx: np.ndarray,
    frame_ids: np.ndarray,
    times: np.ndarray,
    n_frames_total: int,
) -> tuple[list[EmbeddingFrameStats], np.ndarray, np.ndarray, np.ndarray]:
    """Shared projection + per-frame cluster metrics.

    Returns:
        stats:        per-frame metric rows
        labels:       (M,) video-level KMeans team labels per crop
        proj2_umap:   (M, 2) shared UMAP projection (for maps)
        proj2_tsne:   (M, 2) shared t-SNE projection (for maps)
    """
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import umap

    # --- Shared projections: fit ONCE on pooled embeddings ---
    proj3 = umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings)
    proj2_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    perp = min(30.0, max(2.0, (len(embeddings) - 1) / 3))
    proj2_tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                      learning_rate="auto", random_state=42).fit_transform(embeddings)

    # --- Video-level teams: KMeans on the shared 3D space ---
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(proj3)
    labels = km.labels_

    # --- Per-frame metrics ---
    D = _cosine_distance_matrix(embeddings)
    stats: list[EmbeddingFrameStats] = []
    for f in range(len(frame_ids)):
        mask = frame_idx == f
        if mask.sum() < 4:
            continue
        labs = labels[mask]
        if len(set(labs.tolist())) < 2:
            sil = 0.0
            c0 = c1 = 0.0
            cdist = 0.0
        else:
            sil = float(silhouette_score(proj3[mask], labs))
            emb_f = embeddings[mask]
            idx0 = np.where(labs == 0)[0]
            idx1 = np.where(labs == 1)[0]
            cent0 = emb_f[idx0].mean(axis=0)
            cent1 = emb_f[idx1].mean(axis=0)
            cent0 /= max(np.linalg.norm(cent0), 1e-12)
            cent1 /= max(np.linalg.norm(cent1), 1e-12)
            cdist = float(1.0 - np.clip(cent0 @ cent1, -1, 1))
            sub = D[np.ix_(np.where(mask)[0], np.where(mask)[0])]
            c0 = float(sub[np.ix_(idx0, idx0)][np.triu_indices(len(idx0), 1)].mean()) if len(idx0) > 1 else 0.0
            c1 = float(sub[np.ix_(idx1, idx1)][np.triu_indices(len(idx1), 1)].mean()) if len(idx1) > 1 else 0.0

        stats.append(EmbeddingFrameStats(
            frame_id=int(frame_ids[f]),
            time_s=float(times[f]),
            n_crops=int(mask.sum()),
            silhouette=sil,
            centroid_distance=cdist,
            intra_dist_team0=c0,
            intra_dist_team1=c1,
            team_sizes=f"{int((labs == 0).sum())}/{int((labs == 1).sum())}",
        ))
    return stats, labels, proj2_umap, proj2_tsne


def write_embedding_csv(stats: list[EmbeddingFrameStats], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EmbeddingFrameStats.fieldnames())
        writer.writeheader()
        for s in stats:
            writer.writerow({k: getattr(s, k) for k in EmbeddingFrameStats.fieldnames()})
    return path
