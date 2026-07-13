"""Shared plotting for frames (players, ball, velocity arrows) on the pitch.

Kept deliberately thin in Phase 0 — later phases add surface/heatmap and path
overlays here. Everything draws onto a matplotlib Axes so demos can compose
multiple frames into a grid.
"""

from __future__ import annotations

import numpy as np

from . import pitch
from .state import TEAM_ATTACK, FrozenFrame

ATTACK_COLOR = "#d62728"   # red
DEFEND_COLOR = "#1f77b4"   # blue
BALL_COLOR = "#111111"


def plot_frame(frame: FrozenFrame, ax=None, *, title: str | None = None,
               velocity_scale: float = 1.0, annotate: bool = False,
               draw_field: bool = True):
    """Draw a ``FrozenFrame``: pitch, players by team, ball, and velocity arrows.

    Velocity arrows are drawn in data units: ``velocity_scale=1.0`` means an
    arrow shows where the player would be after 1 second at constant velocity,
    which keeps the visual physically meaningful.

    Set ``draw_field=False`` to skip drawing the pitch (when a caller has already
    drawn it, e.g. under a control-surface fill).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    if draw_field:
        pitch.draw_pitch(ax)

    for mask, color, label in (
        (frame.attack_mask, ATTACK_COLOR, "attack"),
        (frame.defend_mask, DEFEND_COLOR, "defend"),
    ):
        pts = frame.positions[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=140, c=color, edgecolors="white",
                   linewidths=1.2, zorder=3, label=label)

    # Velocity arrows (skip near-zero velocities to reduce clutter).
    moving = frame.speeds > 1e-3
    if np.any(moving):
        p = frame.positions[moving]
        v = frame.velocities[moving] * velocity_scale
        ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1],
                  angles="xy", scale_units="xy", scale=1.0,
                  color="#333333", width=0.003, zorder=2)

    # Ball.
    ax.scatter(frame.ball_pos[0], frame.ball_pos[1], s=70, c=BALL_COLOR,
               edgecolors="white", linewidths=1.0, marker="o", zorder=5,
               label="ball")

    # Highlight the carrier with a ring.
    if frame.carrier_position is not None:
        cp = frame.carrier_position
        ax.scatter(cp[0], cp[1], s=320, facecolors="none",
                   edgecolors="#ffcc00", linewidths=2.2, zorder=4)

    if annotate:
        for i, (x, y) in enumerate(frame.positions):
            ax.annotate(str(i), (x, y), fontsize=7, ha="center", va="center",
                        color="white", zorder=6)

    if title:
        ax.set_title(title, fontsize=11)
    return ax


def plot_team_regions(frame: FrozenFrame, team_map: np.ndarray, ax=None, *,
                      title: str | None = None, alpha: float = 0.35,
                      annotate: bool = False):
    """Fill each grid cell by the team controlling it, then overlay the players.

    ``team_map`` is a (ny, nx) array of team ids (0 = attack, 1 = defend), as
    produced from a control-index map by :func:`footlab.geometry.control_to_team`.
    It is drawn with ``origin="lower"`` and ``extent=PITCH_EXTENT`` so it aligns
    with pitch coordinates (see :func:`footlab.pitch.make_grid`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    pitch.draw_pitch(ax)
    cmap = ListedColormap([ATTACK_COLOR, DEFEND_COLOR])
    ax.imshow(team_map, origin="lower", extent=pitch.PITCH_EXTENT,
              cmap=cmap, vmin=0, vmax=1, alpha=alpha, zorder=0,
              interpolation="nearest", aspect="equal")
    plot_frame(frame, ax=ax, title=title, annotate=annotate, draw_field=False)
    return ax


def plot_surface(frame: FrozenFrame, surface: np.ndarray, ax=None, *,
                 title: str | None = None, cmap: str = "RdBu_r",
                 vmin: float = 0.0, vmax: float = 1.0, alpha: float = 0.75,
                 colorbar: bool = True, cbar_label: str = "P(attack control)"):
    """Heatmap a continuous grid ``surface`` (e.g. pitch control) on the pitch.

    ``surface`` is a (ny, nx) array aligned with :func:`footlab.pitch.make_grid`.
    With the default ``RdBu_r`` colormap, red = attacking team, blue = defending
    team, matching the player colors. Players are overlaid on top.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    pitch.draw_pitch(ax)
    im = ax.imshow(surface, origin="lower", extent=pitch.PITCH_EXTENT,
                   cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, zorder=0,
                   interpolation="bilinear", aspect="equal")
    plot_frame(frame, ax=ax, title=title, draw_field=False)
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label(cbar_label, fontsize=9)
    return ax, im


def plot_path(ax, points: np.ndarray, *, color: str = "#00e676",
              label: str | None = None, linewidth: float = 2.5,
              marker_ends: bool = True):
    """Overlay a planned path (K, 2 world coords) with start/goal markers."""
    points = np.asarray(points, float)
    ax.plot(points[:, 0], points[:, 1], "-", color=color, linewidth=linewidth,
            zorder=7, label=label, solid_capstyle="round")
    if marker_ends:
        ax.scatter(points[0, 0], points[0, 1], s=90, marker="o", color=color,
                   edgecolors="black", linewidths=1.2, zorder=8)
        ax.scatter(points[-1, 0], points[-1, 1], s=180, marker="*", color=color,
                   edgecolors="black", linewidths=1.2, zorder=8)
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .simulate import scenario_counter_attack

    plot_frame(scenario_counter_attack(), title="counter attack", annotate=True)
    plt.show()
