"""Pitch geometry: constants, coordinate helpers, and matplotlib rendering.

This module is the single source of truth for the coordinate *convention* the
whole repository relies on. Every other module imports its constants from here
rather than hard-coding numbers, so if the convention ever changes it changes
in exactly one place.

Convention (matches Metrica Sports / Friends-of-Tracking, i.e. Laurie Shaw's
``LaurieOnTracking`` reference implementation, so real tracking data drops in
later without rescaling):

    * Pitch is 105 m long x 68 m wide.
    * Origin (0, 0) is the CENTER SPOT.
    * x runs along the length:  x in [-52.5, +52.5]  (meters)
    * y runs across the width:  y in [-34.0, +34.0]  (meters)
    * The ATTACKING team always plays left -> right, attacking the goal at
      (+52.5, 0) and defending the goal at (-52.5, 0).
    * All positions in meters, all velocities in meters/second.

Pitch marking dimensions use IFAB-standard values.
"""

from __future__ import annotations

import numpy as np

# --- Core dimensions (meters) ------------------------------------------------
PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

HALF_LENGTH: float = PITCH_LENGTH / 2.0  # 52.5
HALF_WIDTH: float = PITCH_WIDTH / 2.0    # 34.0

# --- Standard markings (meters) ----------------------------------------------
CENTER_CIRCLE_RADIUS: float = 9.15
GOAL_WIDTH: float = 7.32            # inner post to inner post
GOAL_DEPTH: float = 2.0             # drawn only, purely cosmetic

PENALTY_AREA_LENGTH: float = 16.5   # depth from the goal line
PENALTY_AREA_WIDTH: float = 40.32   # total width
GOAL_AREA_LENGTH: float = 5.5       # 6-yard box depth
GOAL_AREA_WIDTH: float = 18.32      # 6-yard box width
PENALTY_SPOT_DIST: float = 11.0     # from goal line
PENALTY_ARC_RADIUS: float = 9.15    # same as center circle

# --- Landmark points (attacking team plays toward +x) ------------------------
ATTACKING_GOAL_CENTER: np.ndarray = np.array([HALF_LENGTH, 0.0])   # (+52.5, 0)
DEFENDING_GOAL_CENTER: np.ndarray = np.array([-HALF_LENGTH, 0.0])  # (-52.5, 0)

# (left, right, bottom, top) — pass to ``imshow(..., extent=PITCH_EXTENT,
# origin="lower")`` so a grid array lines up with pitch coordinates.
PITCH_EXTENT: tuple[float, float, float, float] = (
    -HALF_LENGTH, HALF_LENGTH, -HALF_WIDTH, HALF_WIDTH)


# --- Coordinate helpers ------------------------------------------------------
def is_on_pitch(xy: np.ndarray, *, atol: float = 1e-9) -> np.ndarray | bool:
    """Return whether point(s) lie within the pitch rectangle.

    Accepts a single (2,) point or an (N, 2) array; returns a bool or a
    length-N bool array respectively. ``atol`` tolerates floating-point points
    sitting exactly on the touchline.
    """
    xy = np.asarray(xy, dtype=float)
    x = xy[..., 0]
    y = xy[..., 1]
    on = (
        (x >= -HALF_LENGTH - atol)
        & (x <= HALF_LENGTH + atol)
        & (y >= -HALF_WIDTH - atol)
        & (y <= HALF_WIDTH + atol)
    )
    return bool(on) if on.ndim == 0 else on


def clip_to_pitch(xy: np.ndarray) -> np.ndarray:
    """Clip point(s) to the pitch rectangle (used after adding jitter noise)."""
    xy = np.asarray(xy, dtype=float)
    out = xy.copy()
    out[..., 0] = np.clip(out[..., 0], -HALF_LENGTH, HALF_LENGTH)
    out[..., 1] = np.clip(out[..., 1], -HALF_WIDTH, HALF_WIDTH)
    return out


def make_grid(cell_size: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular grid of cell-*center* coordinates covering the whole pitch.

    Shared by every surface-based phase (dominant regions, pitch control, cost
    maps) so they all sample the pitch identically.

    Returns ``(xs, ys, centers)`` where:
        xs:      (nx,) cell-center x coordinates, ascending from -52.5.
        ys:      (ny,) cell-center y coordinates, ascending from -34.0.
        centers: (ny, nx, 2) stacked (x, y) center of every cell.

    ``centers`` is indexed ``[row, col] = [y-index, x-index]`` and ordered so
    that with ``imshow(..., extent=PITCH_EXTENT, origin="lower")`` row 0 is the
    bottom of the pitch — i.e. it renders the right way up.
    """
    nx = int(round(PITCH_LENGTH / cell_size))
    ny = int(round(PITCH_WIDTH / cell_size))
    xs = -HALF_LENGTH + (np.arange(nx) + 0.5) * cell_size
    ys = -HALF_WIDTH + (np.arange(ny) + 0.5) * cell_size
    xx, yy = np.meshgrid(xs, ys)              # both (ny, nx)
    centers = np.stack([xx, yy], axis=-1)     # (ny, nx, 2)
    return xs, ys, centers


# --- Rendering ---------------------------------------------------------------
def draw_pitch(ax=None, *, line_color: str = "black", pitch_color: str = "white",
               linewidth: float = 1.2):
    """Draw a to-scale football pitch onto a matplotlib Axes and return it.

    We implement our own minimal pitch renderer (borrowing marking conventions
    from ``mplsoccer``) to avoid taking on a plotting dependency, per the
    project's "keep dependencies minimal" constraint. If ``ax`` is None a new
    figure/axes is created. The axes is set to equal aspect with the origin at
    the center spot and its spines/ticks hidden.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.set_facecolor(pitch_color)
    ax.set_aspect("equal")
    margin = 3.0
    ax.set_xlim(-HALF_LENGTH - margin, HALF_LENGTH + margin)
    ax.set_ylim(-HALF_WIDTH - margin, HALF_WIDTH + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    lk = dict(color=line_color, linewidth=linewidth, zorder=1)

    # Outer boundary + halfway line.
    ax.plot(
        [-HALF_LENGTH, HALF_LENGTH, HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH],
        [-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH],
        **lk,
    )
    ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], **lk)

    # Center circle + spot.
    ax.add_patch(Arc((0, 0), 2 * CENTER_CIRCLE_RADIUS, 2 * CENTER_CIRCLE_RADIUS,
                     theta1=0, theta2=360, **lk))
    ax.plot(0, 0, marker="o", markersize=2, color=line_color, zorder=1)

    # Draw both penalty ends. ``side = +1`` is the attacking goal (+x).
    for side in (+1, -1):
        goal_x = side * HALF_LENGTH

        # Penalty area.
        pa_x = goal_x - side * PENALTY_AREA_LENGTH
        ax.plot([goal_x, pa_x, pa_x, goal_x],
                [-PENALTY_AREA_WIDTH / 2, -PENALTY_AREA_WIDTH / 2,
                 PENALTY_AREA_WIDTH / 2, PENALTY_AREA_WIDTH / 2], **lk)

        # Goal area (6-yard box).
        ga_x = goal_x - side * GOAL_AREA_LENGTH
        ax.plot([goal_x, ga_x, ga_x, goal_x],
                [-GOAL_AREA_WIDTH / 2, -GOAL_AREA_WIDTH / 2,
                 GOAL_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2], **lk)

        # Goal (cosmetic depth behind the line).
        gm_x = goal_x + side * GOAL_DEPTH
        ax.plot([goal_x, gm_x, gm_x, goal_x],
                [-GOAL_WIDTH / 2, -GOAL_WIDTH / 2,
                 GOAL_WIDTH / 2, GOAL_WIDTH / 2], **lk)

        # Penalty spot.
        spot_x = goal_x - side * PENALTY_SPOT_DIST
        ax.plot(spot_x, 0, marker="o", markersize=2, color=line_color, zorder=1)

        # Penalty arc: the portion of a circle around the penalty spot that
        # bulges OUT of the penalty area. Compute the angle where that circle
        # crosses the top edge of the penalty box, then draw the outward sweep.
        dx = (pa_x - spot_x)  # signed horizontal gap spot -> box edge
        # Guard against numerical issues; |dx| < radius by construction.
        theta = np.degrees(np.arccos(np.clip(dx / PENALTY_ARC_RADIUS, -1, 1)))
        if side == +1:  # arc opens toward -x (pitch center)
            t1, t2 = theta, 360 - theta
        else:           # arc opens toward +x
            t1, t2 = -theta, theta
        ax.add_patch(Arc((spot_x, 0), 2 * PENALTY_ARC_RADIUS,
                         2 * PENALTY_ARC_RADIUS, theta1=t1, theta2=t2, **lk))

    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ax = draw_pitch()
    ax.set_title("footlab pitch — origin at center, attack -> +x")
    plt.show()
