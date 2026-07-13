"""Phase 1 — space geometry: Voronoi control and dominant regions.

Both functions answer "which player owns each patch of pitch?" and return a
``(ny, nx)`` array of the controlling player's index over the grid from
:func:`footlab.pitch.make_grid`.

    * :func:`voronoi_control` — nearest player by straight-line distance. The
      classic Voronoi partition; ignores velocity.
    * :func:`dominant_region` — player who can *arrive soonest* under a simple
      motion model, so momentum bends the boundaries.

Use :func:`control_to_team` to collapse a player-index map into a team-id map for
plotting or area statistics.

References:
    Taki, T. & Hasegawa, J. (2000). "Visualization of dominant region in team
    games and its application to teamwork analysis." Proc. Computer Graphics
    International. Introduces the *dominant region*: the set of pitch points a
    player reaches before any opponent, generalizing the Voronoi cell by using
    arrival time instead of distance.

    Fujimura, A. & Sugihara, K. (2005). "Geometric analysis and quantitative
    evaluation of sport teamwork." Systems and Computers in Japan, 36(6).
    Player motion model with a reaction delay, bounded acceleration and a
    top-speed limit, giving the time to reach a target.
"""

from __future__ import annotations

import numpy as np

from . import pitch
from .state import FrozenFrame

# Motion-model defaults. NOTE these describe *space control*, not the sprint cap
# used to generate velocities (state.MAX_PLAYER_SPEED = 9 m/s). The control speed
# is the effective running speed toward a target and follows Spearman (2018).
REACTION_TIME: float = 0.7      # s, latency before a player redirects
CONTROL_MAX_SPEED: float = 5.0  # m/s, effective running speed toward a target


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of ``a`` (N,2) and ``b`` (M,2).

    Returns an (N, M) array.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a[:, None, :] - b[None, :, :]      # (N, M, 2)
    return np.sqrt(np.einsum("nmk,nmk->nm", diff, diff))


def time_to_arrive(positions: np.ndarray, velocities: np.ndarray,
                   targets: np.ndarray, *, reaction_time: float = REACTION_TIME,
                   max_speed: float = CONTROL_MAX_SPEED) -> np.ndarray:
    """Time for each player to reach each target under the simplified model.

    Model (a simplification of Fujimura & Sugihara 2005): during the reaction
    time the player *coasts* at their current velocity, then runs in a straight
    line at ``max_speed`` to the target:

        t(target) = reaction_time + || target - (p + v * reaction_time) || / max_speed

    The ``v * reaction_time`` head-start is what makes a moving player's region
    stretch ahead of them. The additive ``reaction_time`` is constant across
    players, so it does not affect *who* arrives first (the argmin), but it makes
    the absolute time reusable by Phase 2's time-to-intercept pitch control.

    Simplifications vs. the paper: no explicit acceleration ramp or drag; the
    reaction phase is modeled as pure coasting at the current velocity.

    Parameters:
        positions:  (N, 2) player positions.
        velocities: (N, 2) player velocities.
        targets:    (M, 2) query points.
    Returns:
        (N, M) arrival times in seconds.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    reach = positions + velocities * reaction_time      # (N, 2) post-reaction pos
    dist = _pairwise_dist(reach, np.asarray(targets, dtype=float))
    return reaction_time + dist / max_speed


def _controlling_index(cost_per_player: np.ndarray, shape: tuple[int, int]
                       ) -> np.ndarray:
    """argmin over players (axis 0), reshaped back to the (ny, nx) grid."""
    return np.argmin(cost_per_player, axis=0).reshape(shape).astype(int)


def voronoi_control(frame: FrozenFrame, centers: np.ndarray) -> np.ndarray:
    """Nearest-player (Voronoi) control map — velocity ignored.

    Each cell is assigned to the player with the smallest straight-line distance
    to the cell center. This is exactly the Voronoi partition of the player
    positions, evaluated on the grid.

    Returns a (ny, nx) int array of controlling player indices.
    """
    ny, nx = centers.shape[:2]
    targets = centers.reshape(-1, 2)
    dist = _pairwise_dist(frame.positions, targets)     # (N, M)
    return _controlling_index(dist, (ny, nx))


def dominant_region(frame: FrozenFrame, centers: np.ndarray, *,
                    reaction_time: float = REACTION_TIME,
                    max_speed: float = CONTROL_MAX_SPEED) -> np.ndarray:
    """Dominant-region control map — soonest to arrive under the motion model.

    Each cell is assigned to the player with the minimum :func:`time_to_arrive`.
    Because arrival time depends on velocity, a player's region stretches in the
    direction they are moving and shrinks behind them — the key improvement over
    :func:`voronoi_control`.

    Returns a (ny, nx) int array of controlling player indices.
    """
    ny, nx = centers.shape[:2]
    targets = centers.reshape(-1, 2)
    times = time_to_arrive(frame.positions, frame.velocities, targets,
                           reaction_time=reaction_time, max_speed=max_speed)
    return _controlling_index(times, (ny, nx))


def control_to_team(frame: FrozenFrame, control_index: np.ndarray) -> np.ndarray:
    """Map a player-index control map to a team-id (0/1) map of the same shape."""
    return frame.team_ids[control_index]


def team_area_fraction(frame: FrozenFrame, control_index: np.ndarray, *,
                       team: int = 0) -> float:
    """Fraction of pitch cells controlled by ``team`` (default attack = 0)."""
    team_map = control_to_team(frame, control_index)
    return float(np.mean(team_map == team))


def voronoi_diagram(frame: FrozenFrame):
    """Exact Voronoi diagram of the player positions via ``scipy.spatial``.

    Thin wrapper returning the ``scipy.spatial.Voronoi`` object, whose ridge
    segments are the exact cell boundaries. Used in the demo to overlay the true
    partition on top of the grid-sampled :func:`voronoi_control` map as a visual
    cross-check (they agree up to grid resolution). Requires >= 4 non-degenerate
    points; may raise ``scipy.spatial.QhullError`` on collinear inputs.
    """
    from scipy.spatial import Voronoi

    return Voronoi(frame.positions)


if __name__ == "__main__":
    from .simulate import scenario_counter_attack

    frame = scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    vor = voronoi_control(frame, centers)
    dom = dominant_region(frame, centers)
    changed = np.mean(vor != dom)
    print(f"grid: {centers.shape[0]}x{centers.shape[1]} cells")
    print(f"attack area  Voronoi: {team_area_fraction(frame, vor):.1%}"
          f"   Dominant: {team_area_fraction(frame, dom):.1%}")
    print(f"cells whose owner changed once velocity is considered: {changed:.1%}")
