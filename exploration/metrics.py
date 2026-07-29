"""Cluster-quality metrics for label-free model comparison.

We have no ground-truth team labels for arbitrary frames, so embedders are
compared by *how cleanly their embeddings separate into two clusters* —
measured on KMeans(k=2) assignments over UMAP-3D projections (the same
reduction the production pipeline uses, so results transfer).

Metrics (all from sklearn):
    silhouette        -1..1, higher = tighter, better-separated clusters
    davies_bouldin    0..inf, LOWER = better separation vs. spread
    calinski_harabasz 0..inf, higher = denser/better-separated (relative only)

Plus the two-cluster size balance — a 29/2 split on 31 crops is a red flag
regardless of the other scores (usually means one outlier got its own cluster).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterMetrics:
    model: str
    n_samples: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    cluster_sizes: tuple[int, int]

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "n": self.n_samples,
            "silhouette": round(self.silhouette, 3),
            "davies_bouldin": round(self.davies_bouldin, 3),
            "calinski_harabasz": round(self.calinski_harabasz, 1),
            "split": f"{self.cluster_sizes[0]}/{self.cluster_sizes[1]}",
        }


def evaluate_clustering(
    model_name: str,
    embeddings: np.ndarray,
    projections: np.ndarray,
    labels: np.ndarray,
) -> ClusterMetrics:
    """Score one embedder's 2-cluster partition.

    Args:
        model_name:  registry key (for the results table).
        embeddings:  (N, D) original embeddings (unused by metrics directly
                     but kept in signature for future embedding-space scores).
        projections: (N, d) reduced coordinates the labels were computed on.
        labels:      (N,) cluster assignments (exactly 2 clusters expected).
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    sizes = tuple(sorted((int((labels == k).sum()) for k in set(labels.tolist())),
                         reverse=True))
    sizes = (sizes + (0, 0))[:2]  # pad defensively

    return ClusterMetrics(
        model=model_name,
        n_samples=len(labels),
        silhouette=float(silhouette_score(projections, labels)),
        davies_bouldin=float(davies_bouldin_score(projections, labels)),
        calinski_harabasz=float(calinski_harabasz_score(projections, labels)),
        cluster_sizes=sizes,  # type: ignore[arg-type]
    )


def metrics_table_markdown(rows: list[ClusterMetrics]) -> str:
    """Render metrics as a markdown table for notes/reports."""
    header = (
        "| model | n | silhouette ↑ | davies-bouldin ↓ | calinski-harabasz ↑ | split |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]
    for m in rows:
        r = m.as_row()
        lines.append(
            f"| {r['model']} | {r['n']} | {r['silhouette']} | "
            f"{r['davies_bouldin']} | {r['calinski_harabasz']} | {r['split']} |"
        )
    return "\n".join(lines)
