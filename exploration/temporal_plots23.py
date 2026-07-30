"""Renderers for Phase 2 (embedding-over-time) and Phase 3 (spatial) outputs.

All functions take already-computed arrays (from temporal_embeddings /
temporal_spatial) — drawing only, no math, no models.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Phase 2 — embedding maps + metric traces
# ---------------------------------------------------------------------------

def render_shared_projection_map(
    proj2: np.ndarray,
    frame_idx: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    output_png: str,
    reducer_name: str,
    title: str,
    n_windows: int = 6,
) -> str:
    """Master scatter (colored by time) + small-multiples per time window.

    Left: every crop in the shared projection, color = when it appeared
    (viridis: early=dark purple, late=yellow). Cluster drift shows as a
    color gradient ACROSS a cluster; stable clusters show mixed colors.

    Right: one mini-scatter per time window, same axes, so you can watch
    the two team clouds tighten, separate, or smear over the video.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = len(times)
    window = max(1, n_frames // n_windows)
    windows = [(i * window, min((i + 1) * window, n_frames))
               for i in range(n_windows)]

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1)
    sc = ax.scatter(proj2[:, 0], proj2[:, 1], c=times[frame_idx],
                    cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="time (s)")
    for lab in set(labels.tolist()):
        m = labels == lab
        c = proj2[m].mean(axis=0)
        ax.annotate(f"team {lab}", c, fontsize=11, weight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_title(f"{reducer_name} shared projection — color = time")
    ax.set_xticks([]); ax.set_yticks([])

    # small multiples: 2 rows × 3 columns on the right half
    outer = fig.add_gridspec(2, 3, left=0.55, right=0.98, top=0.88, bottom=0.08,
                             wspace=0.15, hspace=0.35)
    for w, (a, b) in enumerate(windows):
        sub = fig.add_subplot(outer[w // 3, w % 3])
        mask = (frame_idx >= a) & (frame_idx < b)
        sub.scatter(proj2[~mask, 0], proj2[~mask, 1], c="lightgray", s=3, alpha=0.3)
        for lab in set(labels.tolist()):
            m = mask & (labels == lab)
            sub.scatter(proj2[m, 0], proj2[m, 1], s=8, alpha=0.8,
                        color=f"C{lab}")
        sub.set_title(f"{times[a]:.0f}–{times[min(b, n_frames - 1)]:.0f}s", fontsize=9)
        sub.set_xticks([]); sub.set_yticks([])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def render_embedding_traces(
    times: np.ndarray,
    silhouette: np.ndarray,
    centroid_dist: np.ndarray,
    intra0: np.ndarray,
    intra1: np.ndarray,
    output_png: str,
    title: str,
) -> str:
    """Two-panel trace: silhouette over time; centroid vs intra distances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(times, silhouette, lw=1.8, color="tab:blue")
    axes[0].axhline(np.mean(silhouette), color="gray", ls="--", lw=1,
                    label=f"mean {np.mean(silhouette):.2f}")
    axes[0].set_ylabel("silhouette")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Cluster separability over time — dips = teams hard to tell apart")

    axes[1].plot(times, centroid_dist, lw=1.8, color="tab:red",
                 label="between-centroid distance")
    axes[1].plot(times, intra0, lw=1.2, alpha=0.8, label="intra team 0")
    axes[1].plot(times, intra1, lw=1.2, alpha=0.8, label="intra team 1")
    axes[1].set_ylabel("cosine distance")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Embedding geometry — healthy: between >> within")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


# ---------------------------------------------------------------------------
# Phase 3 — spatial heatmaps, coverage, jitter
# ---------------------------------------------------------------------------

def render_spatial_panels(
    heatmaps: dict[str, np.ndarray],
    coverage: dict[int, float],
    vertices: dict[int, tuple[float, float]],
    jitter_times: np.ndarray,
    jitter: np.ndarray,
    output_png: str,
    title: str,
    draw_pitch_fn=None,
) -> str:
    """Four-panel: heatmap team0/team1, keypoint coverage, H jitter trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))

    # heatmaps on pitch
    for ax, key, name in ((axes[0][0], "team0", "Team 0 positions"),
                          (axes[0][1], "team1", "Team 1 positions")):
        if draw_pitch_fn is not None:
            draw_pitch_fn(ax)
        hm = heatmaps[key]
        if hm.sum() > 0:
            extent = (-52.5, 52.5, -34, 34)
            ax.imshow(hm, extent=extent, origin="lower", cmap="hot",
                      alpha=0.65, aspect="auto")
        ax.set_title(name)

    # keypoint coverage on pitch
    ax = axes[1][0]
    if draw_pitch_fn is not None:
        draw_pitch_fn(ax)
    for kp_id, (x, y) in vertices.items():
        freq = coverage.get(kp_id, 0.0)
        ax.scatter(x, y, s=60 + 400 * freq, c=[plt.cm.RdYlGn(freq)],
                   edgecolors="black", linewidths=1, zorder=5)
        ax.text(x, y + 1.5, f"{kp_id}\n{freq:.0%}", fontsize=6, ha="center",
                color="white", weight="bold")
    ax.set_title("Keypoint coverage — size/color = detection frequency")

    # jitter trace
    ax = axes[1][1]
    ax.plot(jitter_times, jitter, lw=1.5, color="tab:purple")
    finite = jitter[np.isfinite(jitter)]
    if len(finite):
        ax.axhline(np.median(finite), color="gray", ls="--", lw=1,
                   label=f"median {np.median(finite):.1f}px")
    ax.set_ylabel("corner displacement (px)")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=9)
    ax.set_title("Homography jitter — spikes = instability to smooth")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png
