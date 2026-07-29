"""Embedding visualization — t-SNE scatter of player-crop embeddings.

A LEARNING / DATA-AUDIT tool, not a production component.

Why this exists:
    UMAP reduces embeddings to 3D for CLUSTERING (it supports .transform()
    on new points, which predict() needs). t-SNE gives more faithful 2D
    VISUALIZATIONS but cannot transform new data — so we use it only here,
    to look at the data, never in the pipeline.

What it teaches:
    * Healthy case: two tight, well-separated point clouds (one per team),
      with goalkeepers/referees sitting off to the side.
    * Failure modes you can SEE before they hit production:
        - one diffuse cloud  → kits too similar / poor lighting; clustering
                               will split noise, not teams
        - three+ clouds      → referees or bench players contaminating the
                               crop set
        - outliers far away  → occluded / motion-blurred / partial crops;
                               these are the samples to inspect by eye
    * Condition analysis: color points by metadata (frame index, lighting,
      camera side) to discover *when* embeddings degrade — e.g. floodlight
      shadows, low sun, rain. This is data auditing: the model is fixed,
      the DATA is what varies.

Reference:
    van der Maaten & Hinton (2008). "Visualizing Data using t-SNE."
    JMLR 9 — note: t-SNE distances BETWEEN clusters are not meaningful;
    only local neighbourhood structure is trustworthy.
"""

from __future__ import annotations

import numpy as np


def tsne_project(
    features: np.ndarray,
    perplexity: float = 15.0,
    random_state: int = 42,
) -> np.ndarray:
    """Project (N, D) embeddings to (N, 2) with t-SNE.

    Args:
        features:     (N, D) embedding matrix (e.g. N crops × 768 SigLIP dims).
        perplexity:   roughly "expected neighbours per point". Rule of thumb:
                      5–50, and must be < N/3. Lower = tighter local clusters,
                      higher = more global structure. For ~25 crops, 8–15.
        random_state: fixed seed for reproducible figures.

    Returns:
        (N, 2) 2D coordinates for scatter plotting.
    """
    from sklearn.manifold import TSNE

    n = len(features)
    # t-SNE requires perplexity < n_samples; clamp sensibly.
    safe_perplexity = min(perplexity, max(1.0, (n - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def render_tsne_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray | None = None,
    output_png: str = "data/tsne_embeddings.png",
    title: str = "t-SNE — player crop embeddings",
) -> str:
    """Render a 2D scatter of t-SNE coordinates, colored by cluster label.

    Args:
        coords_2d:  (N, 2) from tsne_project.
        labels:     (N,) cluster/team IDs; None = all one colour.
        output_png: where to save the figure.
        title:      plot title.

    Returns:
        The output path (for chaining/logging).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    if labels is None:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=60, alpha=0.8)
    else:
        for lab in sorted(set(labels.tolist())):
            mask = labels == lab
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                s=60, alpha=0.8, label=f"cluster {lab}",
            )
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1 (arbitrary units)")
    ax.set_ylabel("t-SNE dim 2 (arbitrary units)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png
