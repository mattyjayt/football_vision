"""Phase 2b — positional value surface and the combined planner cost map.

Pitch control (Phase 2) says where the ball is *safe*; it says nothing about
where it is *useful*. This module adds:

    * a positional **value** surface (xT-style): how threatening a location is,
      rising toward the attacking goal and spiking in the box, and
    * the planner **cost map** that fuses safety, value and a defender-proximity
      penalty into a single non-negative number per cell for Phase 3+ to search.

Value reference:
    Singh, K. (2019). "Introducing Expected Threat (xT)."
    https://karun.in/blog/expected-threat.html . xT assigns each pitch location
    a value equal to the probability a possession there leads to a goal soon,
    learned from event data on a coarse grid; it rises toward goal and jumps
    inside the penalty area.

    Simplification: we do NOT fit xT from data. We use a hand-crafted analytic
    surrogate with the same qualitative shape (monotone toward goal,
    central-biased near it, boosted in the box), normalized to [0, 1]. Swap in a
    data-driven xT grid later without touching the cost-map interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from . import pitch
from .pitch_control import DEFAULT_PARAMS, PitchControlParams, pitch_control_surface
from .state import FrozenFrame

# --- Value-surface shape parameters ------------------------------------------
GOAL_VALUE_SCALE: float = 18.0   # m; length scale of the value decay from goal
_W_PROGRESS: float = 0.40        # weight on up-pitch progress
_W_GOAL: float = 0.55            # weight on proximity to goal
_W_BOX: float = 0.15             # weight on the in-the-box bonus
_BOX_SHARPNESS: float = 1.0      # 1/m; edge softness of the box indicator

# --- Defender-proximity penalty ----------------------------------------------
DEFENDER_PENALTY_SIGMA: float = 4.0  # m; width of the Gaussian around a defender


def _value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic xT-surrogate value at pitch coordinate(s), in [0, 1].

    value = 0.40 * progress_up_pitch
          + 0.55 * proximity_to_attacking_goal
          + 0.15 * inside_penalty_area
    then clipped to [0, 1]. Accepts scalars or arrays (broadcast together).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Up-pitch progress: 0 at own goal line, 1 at the attacking goal line.
    progress = np.clip((x + pitch.HALF_LENGTH) / pitch.PITCH_LENGTH, 0.0, 1.0)

    # Proximity to the attacking goal (also central-biased: wide points are
    # farther from the goal center, hence worth less).
    d_goal = np.hypot(x - pitch.ATTACKING_GOAL_CENTER[0],
                      y - pitch.ATTACKING_GOAL_CENTER[1])
    goal_prox = np.exp(-d_goal / GOAL_VALUE_SCALE)

    # Smooth "inside the penalty area" indicator (~1 in the box, ~0 outside).
    pa_x = pitch.HALF_LENGTH - pitch.PENALTY_AREA_LENGTH
    half_w = pitch.PENALTY_AREA_WIDTH / 2.0
    box = (expit(_BOX_SHARPNESS * (x - pa_x))
           * expit(_BOX_SHARPNESS * (half_w - np.abs(y))))

    value = _W_PROGRESS * progress + _W_GOAL * goal_prox + _W_BOX * box
    return np.clip(value, 0.0, 1.0)


def value_at_point(xy: np.ndarray) -> float:
    """Positional value at a single (x, y) point, in [0, 1]."""
    xy = np.asarray(xy, dtype=float)
    return float(_value(xy[0], xy[1]))


def positional_value_surface(centers: np.ndarray) -> np.ndarray:
    """Positional value over a grid from :func:`footlab.pitch.make_grid`.

    Returns a (ny, nx) array in [0, 1]. Independent of the players — it is a
    static property of the pitch (like an xT grid).
    """
    return _value(centers[..., 0], centers[..., 1])


def defender_proximity_penalty(frame: FrozenFrame, centers: np.ndarray, *,
                               sigma: float = DEFENDER_PENALTY_SIGMA,
                               project_time: float = 0.0) -> np.ndarray:
    """Penalty in [0, 1] for being near a defender (max over defenders).

    Each defender contributes a Gaussian bump ``exp(-d^2 / (2 sigma^2))`` that is
    1 at the defender and decays over ``sigma`` meters; the penalty at a cell is
    the largest such bump. ``project_time`` optionally advances defenders along
    their velocity first (0 = use current positions; velocity is already encoded
    in pitch control, so we default to no projection to avoid double-counting).

    Returns a (ny, nx) array; all zeros if there are no defenders.
    """
    ny, nx = centers.shape[:2]
    targets = centers.reshape(-1, 2)
    def_pos = frame.defender_positions
    if def_pos.shape[0] == 0:
        return np.zeros((ny, nx))
    def_vel = frame.velocities[frame.defend_mask]
    future = def_pos + def_vel * project_time                 # (Nd, 2)
    diff = future[:, None, :] - targets[None, :, :]           # (Nd, M, 2)
    d2 = np.einsum("nmk,nmk->nm", diff, diff)                 # (Nd, M)
    pen = np.exp(-d2 / (2.0 * sigma * sigma))
    return pen.max(axis=0).reshape(ny, nx)


@dataclass(frozen=True)
class CostWeights:
    """Weights on the three cost terms; the tactical dial for the planner.

    w_control:  weight on (1 - pitch_control)  — avoid contested/opponent space.
    w_value:    weight on (1 - value)          — prefer threatening locations.
    w_defender: weight on defender_penalty     — keep clear of defenders.

    Larger w_control / w_defender -> cautious, wide routes. Larger w_value ->
    direct, risky routes straight at goal.
    """

    w_control: float = 1.0
    w_value: float = 1.0
    w_defender: float = 1.0


DEFAULT_WEIGHTS = CostWeights()


def cost_map(frame: FrozenFrame, centers: np.ndarray, *,
             control: np.ndarray | None = None,
             weights: CostWeights = DEFAULT_WEIGHTS,
             pc_params: PitchControlParams = DEFAULT_PARAMS,
             penalty_sigma: float = DEFENDER_PENALTY_SIGMA,
             project_time: float = 0.0) -> np.ndarray:
    """Combined planner cost over a grid, shape (ny, nx).

        cost = w_control * (1 - pitch_control)
             + w_value   * (1 - value)
             + w_defender * defender_penalty

    Every term is in [0, 1] and every weight is >= 0, so ``cost`` is
    non-negative (required by Dijkstra / A* in Phase 3). Pass a precomputed
    ``control`` surface to avoid recomputing pitch control.
    """
    if control is None:
        control = pitch_control_surface(frame, centers, params=pc_params)
    value = positional_value_surface(centers)
    penalty = defender_proximity_penalty(frame, centers, sigma=penalty_sigma,
                                         project_time=project_time)
    return (weights.w_control * (1.0 - control)
            + weights.w_value * (1.0 - value)
            + weights.w_defender * penalty)


if __name__ == "__main__":
    from .simulate import scenario_counter_attack

    frame = scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    value = positional_value_surface(centers)
    cost = cost_map(frame, centers)
    print(f"value  range [{value.min():.2f}, {value.max():.2f}]")
    print(f"cost   range [{cost.min():.2f}, {cost.max():.2f}]")
    print(f"value at goal mouth (52,0): {value_at_point([52.0, 0.0]):.2f}")
    print(f"value at halfway   (0,0):  {value_at_point([0.0, 0.0]):.2f}")
    print(f"value at own half (-40,0): {value_at_point([-40.0, 0.0]):.2f}")
