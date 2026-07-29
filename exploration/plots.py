"""Plotting for exploration studies — 2D grids and interactive 3D.

Two renderers, two purposes:
    * 2D matplotlib grid  — the report artifact. One row per model, UMAP and
      t-SNE side by side, all on the same figure for direct comparison.
    * 3D plotly HTML      — the exploration tool. Rotate, zoom, hover. A third
      dimension sometimes reveals structure 2D flattens into overlap.

Design note: projections are computed in compare_embedders.py and passed in
here — this module only draws, never computes. Keeps the math testable and
the plots honest (same numbers on paper and screen).
"""

from __future__ import annotations

import numpy as np


def render_comparison_grid_2d(
    projections: dict[str, dict[str, np.ndarray]],
    labels_per_model: dict[str, np.ndarray],
    output_png: str,
    title: str = "Embedder comparison — UMAP vs t-SNE (2D)",
) -> str:
    """Grid figure: rows = models, columns = reducers.

    Args:
        projections:      {model_name: {reducer_name: (N,2) coords}}
        labels_per_model: {model_name: (N,) cluster labels}
        output_png:       where to save.

    Returns:
        The output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(projections.keys())
    reducers = list(next(iter(projections.values())).keys())
    fig, axes = plt.subplots(
        len(models), len(reducers),
        figsize=(7 * len(reducers), 5.5 * len(models)),
        squeeze=False,
    )
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

    for r, model in enumerate(models):
        labels = labels_per_model[model]
        for c, reducer in enumerate(reducers):
            ax = axes[r][c]
            coords = projections[model][reducer]
            for lab in sorted(set(labels.tolist())):
                mask = labels == lab
                ax.scatter(coords[mask, 0], coords[mask, 1], s=35, alpha=0.75,
                           color=colors.get(lab, "gray"), label=f"cluster {lab}")
            ax.set_title(f"{model} — {reducer}", fontsize=11)
            if r == 0 and c == len(reducers) - 1:
                ax.legend(fontsize=8, loc="best")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def render_interactive_3d(
    coords_3d: np.ndarray,
    labels: np.ndarray,
    output_html: str,
    title: str,
) -> str:
    """Interactive 3D scatter (plotly → self-contained HTML).

    Rotate/zoom/hover in any browser. Labels color the points; axes are the
    reducer's first three components (units arbitrary — it's a manifold
    projection, not meters).
    """
    import plotly.graph_objects as go

    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    fig = go.Figure()
    for lab in sorted(set(labels.tolist())):
        mask = labels == lab
        fig.add_trace(go.Scatter3d(
            x=coords_3d[mask, 0], y=coords_3d[mask, 1], z=coords_3d[mask, 2],
            mode="markers",
            marker=dict(size=5, color=colors.get(lab, "#888888"), opacity=0.8),
            name=f"cluster {lab}",
        ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    return output_html
