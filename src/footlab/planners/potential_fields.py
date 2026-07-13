"""Phase 4 — artificial potential fields (APF).

A reactive alternative to the Phase 3 graph search. The ball carrier is treated
as a particle in a force field: the goal attracts it, each defender repels it,
and we extract a path by rolling downhill (gradient descent on the potential).

No global search — just evaluate a force and step. Cheap and reactive, but with
a famous flaw we reproduce on purpose: the particle can get stuck in a **local
minimum** (a spot where attraction and repulsion cancel) and never reach the
goal, even when a path plainly exists.

Reference:
    Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and Mobile
    Robots." International Journal of Robotics Research, 5(1). Introduces the
    artificial potential field: an attractive potential toward the goal plus a
    repulsive potential around each obstacle that is active only within an
    influence distance rho0. Forces are the negative gradients of these
    potentials.

    Attractive force (parabolic well near goal, conic far field to bound the
    far-away magnitude, per Latombe / Koditschek):
        ||x - goal|| <= d0 :  F_att = k_att * (goal - x)
        ||x - goal|| >  d0 :  F_att = k_att * d0 * (goal - x) / ||goal - x||
    Repulsive force from an obstacle at distance rho (active for rho <= rho0):
        F_rep = k_rep * (1/rho - 1/rho0) * (1/rho^2) * (x - obs)/rho

Simplifications: obstacles are points (players), not shapes; we make repulsion
velocity-aware by projecting defenders forward along their velocity by
``project_time`` before treating them as static obstacles for the field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import FrozenFrame


@dataclass(frozen=True)
class PotentialFieldParams:
    """Potential-field parameters.

    k_att:        attractive gain.
    d0:           goal distance (m) below which the well is parabolic, above
                  which it is conic (constant magnitude k_att * d0).
    k_rep:        repulsive gain.
    rho0:         obstacle influence radius (m); beyond it a defender is ignored.
    project_time: seconds to project each defender forward along its velocity
                  before treating it as a static obstacle (velocity-awareness).
    step_size:    gradient-descent step length (m) per iteration.
    max_iters:    iteration cap.
    goal_tol:     distance (m) to the goal counted as "reached".
    min_force:    force magnitude below which we declare a local minimum.
    patience:     stop if the distance-to-goal has not improved for this many
                  steps (catches oscillation around a trap).
    """

    k_att: float = 1.0
    d0: float = 10.0
    k_rep: float = 300.0
    rho0: float = 12.0
    project_time: float = 0.5
    step_size: float = 0.5
    max_iters: int = 800
    goal_tol: float = 1.0
    min_force: float = 1e-4
    patience: int = 40


DEFAULT_PARAMS = PotentialFieldParams()


@dataclass
class FieldPath:
    """Result of rolling downhill through the field.

    points:       (K, 2) world coordinates of the descent path.
    reached_goal: whether the goal was reached within tolerance.
    reason:       'goal' | 'local_minimum' | 'max_iterations'.
    """

    points: np.ndarray
    reached_goal: bool
    reason: str


def projected_defenders(frame: FrozenFrame, project_time: float) -> np.ndarray:
    """Defender positions advanced along their velocity by ``project_time``."""
    return (frame.defender_positions
            + frame.velocities[frame.defend_mask] * project_time)


def attractive_force(pos: np.ndarray, goal: np.ndarray, *, k_att: float,
                     d0: float) -> np.ndarray:
    """Attractive force toward ``goal`` (parabolic near, conic far)."""
    pos = np.asarray(pos, dtype=float)
    goal = np.asarray(goal, dtype=float)
    diff = goal - pos                                  # (..., 2) toward goal
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    far = dist > d0
    with np.errstate(invalid="ignore", divide="ignore"):
        conic = k_att * d0 * diff / dist
    force = np.where(far, conic, k_att * diff)
    # At the goal exactly (dist == 0) the force is zero.
    return np.where(dist == 0, 0.0, force)


def repulsive_force(pos: np.ndarray, obstacles: np.ndarray, *, k_rep: float,
                    rho0: float) -> np.ndarray:
    """Summed repulsive force from point ``obstacles`` (M, 2), active < rho0."""
    pos = np.asarray(pos, dtype=float)
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    total = np.zeros_like(pos)
    for obs in obstacles:
        diff = pos - obs                               # away from the obstacle
        rho = np.linalg.norm(diff, axis=-1, keepdims=True)
        active = (rho <= rho0) & (rho > 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            coeff = k_rep * (1.0 / rho - 1.0 / rho0) / (rho ** 2)
            f = coeff * (diff / rho)
        total += np.where(active, f, 0.0)
    return total


def total_force(pos: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, *,
                params: PotentialFieldParams = DEFAULT_PARAMS) -> np.ndarray:
    """Attractive + repulsive force at ``pos``."""
    return (attractive_force(pos, goal, k_att=params.k_att, d0=params.d0)
            + repulsive_force(pos, obstacles, k_rep=params.k_rep,
                              rho0=params.rho0))


def descend(frame: FrozenFrame, goal: np.ndarray, *,
            params: PotentialFieldParams = DEFAULT_PARAMS) -> FieldPath:
    """Extract a path by gradient descent on the potential from the carrier.

    Steps a fixed arc length in the force direction each iteration. Stops on
    reaching the goal, on a vanishing force, or when the distance-to-goal stops
    improving for ``patience`` steps (a local minimum / oscillation trap).
    """
    goal = np.asarray(goal, dtype=float)
    obstacles = projected_defenders(frame, params.project_time)
    x = frame.carrier_position.astype(float).copy()
    path = [x.copy()]

    best_d = np.inf
    stale = 0
    reason = "max_iterations"
    for _ in range(params.max_iters):
        d = float(np.linalg.norm(x - goal))
        if d < params.goal_tol:
            reason = "goal"
            break
        f = total_force(x, goal, obstacles, params=params)
        fmag = float(np.linalg.norm(f))
        if fmag < params.min_force:
            reason = "local_minimum"
            break
        x = x + params.step_size * f / fmag
        path.append(x.copy())
        if d < best_d - 1e-3:
            best_d = d
            stale = 0
        else:
            stale += 1
            if stale > params.patience:
                reason = "local_minimum"
                break

    return FieldPath(points=np.array(path),
                     reached_goal=(reason == "goal"), reason=reason)


def force_field(points: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, *,
                params: PotentialFieldParams = DEFAULT_PARAMS) -> np.ndarray:
    """Total force at a set of query ``points`` (..., 2) — for quiver plots."""
    return total_force(points, goal, obstacles, params=params)


if __name__ == "__main__":
    from ..state import TEAM_ATTACK, TEAM_DEFEND

    # Colinear obstacle -> classic local-minimum trap.
    frame = FrozenFrame(
        positions=[[-35.0, 0.0], [5.0, 0.0]],
        velocities=[[0.0, 0.0], [0.0, 0.0]],
        team_ids=[TEAM_ATTACK, TEAM_DEFEND],
        ball_pos=[-35.0, 0.0], ball_carrier=0)
    result = descend(frame, np.array([45.0, 0.0]))
    print(f"trap case: reached={result.reached_goal} reason={result.reason} "
          f"stopped at x={result.points[-1, 0]:.1f}")
