"""Temporal dashboard — multi-panel PNG + interactive HTML over the CSV.

Reads the per-frame CSV from temporal.py (never re-runs inference) and
renders the observability views:

    Panel 1: detection counts per class vs time (+ full-complement line)
    Panel 2: confidence distribution band (p10–p90) + median vs time
    Panel 3: keypoints detected vs time + homography RMS (successful frames)
    Panel 4: bbox area (mean + p90) vs time — the camera-cut detector
    Panel 5: class balance, stacked area vs time

The interactive HTML (plotly) stacks the same series with a shared x-axis
so you can scrub the timeline and cross-reference dips.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_csv(path: str | Path) -> list[dict[str, float]]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _col(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.array([r[name] for r in rows])


def render_dashboard(
    rows: list[dict[str, float]],
    output_png: str,
    title: str,
) -> str:
    """Five-panel observability dashboard over one run's CSV rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _col(rows, "time_s")
    fig, axes = plt.subplots(5, 1, figsize=(15, 18), sharex=True)

    # 1 — detection counts
    ax = axes[0]
    ax.plot(t, _col(rows, "n_players"), label="players", lw=1.5)
    ax.plot(t, _col(rows, "n_goalkeepers"), label="goalkeepers", lw=1)
    ax.plot(t, _col(rows, "n_referees"), label="referees", lw=1)
    ax.axhline(22, color="gray", ls="--", lw=1, label="full complement (22)")
    ax.set_ylabel("# detections")
    ax.legend(loc="upper right", ncols=4, fontsize=9)
    ax.set_title("Detection counts — dips mark camera cuts, occlusions, lighting events")

    # 2 — confidence band
    ax = axes[1]
    ax.fill_between(t, _col(rows, "conf_p10"), _col(rows, "conf_p90"),
                    alpha=0.3, label="p10–p90")
    ax.plot(t, _col(rows, "conf_median"), lw=1.5, label="median")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Detection confidence — a leftward slide = conditions changed")

    # 3 — keypoints + homography
    ax = axes[2]
    ax.plot(t, _col(rows, "n_keypoints"), lw=1.5, color="tab:purple",
            label="keypoints detected (of 32)")
    ax.set_ylabel("keypoints")
    ax2 = ax.twinx()
    ok = _col(rows, "homography_ok") == 1
    ax2.scatter(t[ok], _col(rows, "homography_rms_m")[ok], s=8,
                color="tab:red", alpha=0.5, label="H RMS (m)")
    ax2.set_ylabel("H RMS (m)", color="tab:red")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Pitch registration — keypoint collapse = close-up/cut frames")

    # 4 — bbox area
    ax = axes[3]
    ax.plot(t, _col(rows, "bbox_area_mean"), lw=1.5, label="mean box area")
    ax.plot(t, _col(rows, "bbox_area_p90"), lw=1, alpha=0.7, label="p90 box area")
    ax.set_ylabel("px²")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Box area — sudden jumps mark camera zoom/cut, not model failure")

    # 5 — class balance stacked
    ax = axes[4]
    ax.stackplot(t, _col(rows, "n_players"), _col(rows, "n_goalkeepers"),
                 _col(rows, "n_referees"),
                 labels=["players", "goalkeepers", "referees"], alpha=0.8)
    ax.plot(t, _col(rows, "ball_detected") * max(1.0, _col(rows, "n_players").max() / 2),
            lw=1, color="black", alpha=0.6, label="ball (scaled)")
    ax.set_ylabel("# objects")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", ncols=4, fontsize=9)
    ax.set_title("Class balance — ball visibility is its own story")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def render_interactive(
    rows: list[dict[str, float]],
    output_html: str,
    title: str,
) -> str:
    """Interactive timeline (plotly) — scrub and cross-reference dips."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = _col(rows, "time_s")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("detections", "confidence (p10–p90 + median)",
                        "keypoints + H RMS", "bbox area"),
    )

    fig.add_trace(go.Scatter(x=t, y=_col(rows, "n_players"), name="players"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(rows, "n_goalkeepers"), name="GK"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(rows, "n_referees"), name="refs"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(rows, "ball_detected"), name="ball (0/1)",
                             line=dict(dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=_col(rows, "conf_p90"), name="conf p90",
                             line=dict(width=0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(rows, "conf_p10"), name="conf p10–p90",
                             fill="tonexty", line=dict(width=0)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(rows, "conf_median"), name="conf median"),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=_col(rows, "n_keypoints"), name="keypoints"),
                  row=3, col=1)
    ok = _col(rows, "homography_ok") == 1
    fig.add_trace(go.Scatter(x=t[ok], y=_col(rows, "homography_rms_m")[ok],
                             name="H RMS (m)", mode="markers",
                             marker=dict(size=4), yaxis="y3"), row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=_col(rows, "bbox_area_mean"), name="mean area"),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(rows, "bbox_area_p90"), name="p90 area"),
                  row=4, col=1)

    fig.update_layout(title=title, height=1000, hovermode="x unified")
    fig.write_html(output_html, include_plotlyjs="cdn")
    return output_html
