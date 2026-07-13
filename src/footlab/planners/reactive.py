"""Phase 6 — reactive defenders: rollout scoring and replanning.

Every earlier phase assumed a frozen frame: the defenders stand still while we
plan. They don't. This module advances time in small steps, moves the ball
carrier along a planned path, and lets each defender *react* by pursuing an
intercept point under speed/acceleration limits. Two capabilities fall out:

    * :func:`rollout_static`     — score any fixed path against reacting
      defenders (does it reach goal, or get intercepted, and how far it got).
    * :func:`rollout_replanning` — re-run A* every k steps on the *updated* frame
      so the plan adapts as the defense shifts.

Defender motion (Fujimura & Sugihara 2005, simplified): each step a defender
steers toward a lead-pursuit aim point (where the carrier will be in
``lead_time`` seconds), changing velocity under a bounded acceleration and a
top-speed cap.

Reference:
    Fujimura, A. & Sugihara, K. (2005). "Geometric analysis and quantitative
    evaluation of sport teamwork." Player motion with bounded acceleration and a
    speed limit. Pursuit/lead-pursuit is the classic missile-guidance idea (aim
    ahead of a moving target).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..state import FrozenFrame
from ..value_surface import DEFAULT_WEIGHTS, CostWeights, cost_map
from .grid_search import plan_astar, smooth_path


@dataclass(frozen=True)
class ReactiveParams:
    """Rollout / pursuit parameters."""

    dt: float = 0.2                 # s, simulation step
    carrier_speed: float = 6.0      # m/s, ball-carrier dribbling speed
    def_max_speed: float = 6.5      # m/s, defender sprint cap
    def_max_accel: float = 6.0      # m/s^2, defender acceleration cap
    lead_time: float = 0.45         # s, how far ahead defenders aim
    intercept_radius: float = 1.5   # m, tackle distance
    max_time: float = 18.0          # s, simulation cap
    terrain_weight: float = 2.0     # planner terrain weight (replanning)


@dataclass
class RolloutResult:
    """Outcome of a rollout, with full history for animation.

    reached_goal:    carrier arrived at goal without being intercepted.
    intercepted:     a defender closed within the tackle radius.
    progress:        fraction of the initial carrier->goal distance covered.
    sim_time:        seconds simulated.
    times:           (T,) time stamps.
    pos_history:     (T, N, 2) all player positions per step.
    plan_history:    list of (K, 2) the carrier's current planned path per step.
    intercept_point: (2,) where the tackle happened, or None.
    team_ids:        (N,) for plotting.
    carrier_idx:     ball-carrier index.
    """

    reached_goal: bool
    intercepted: bool
    progress: float
    sim_time: float
    times: np.ndarray
    pos_history: np.ndarray
    plan_history: list
    intercept_point: np.ndarray | None
    team_ids: np.ndarray
    carrier_idx: int


# --- path arc-length helpers -------------------------------------------------
def _cumulative_lengths(path: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_at_arc(path: np.ndarray, cum: np.ndarray, arc: float
                  ) -> tuple[np.ndarray, bool]:
    """Position at arc length ``arc`` along ``path``; second value is 'at end'."""
    total = cum[-1]
    if arc >= total or len(path) < 2:
        return path[-1].copy(), True
    idx = int(np.searchsorted(cum, arc) - 1)
    idx = max(0, min(idx, len(path) - 2))
    seg_len = cum[idx + 1] - cum[idx]
    t = (arc - cum[idx]) / seg_len if seg_len > 1e-12 else 0.0
    return path[idx] + t * (path[idx + 1] - path[idx]), False


# --- defender pursuit --------------------------------------------------------
def pursue(def_pos: np.ndarray, def_vel: np.ndarray, aim: np.ndarray, *,
           max_speed: float, max_accel: float, dt: float
           ) -> tuple[np.ndarray, np.ndarray]:
    """Advance defenders one step toward ``aim`` under accel/speed caps.

    Vectorized over defenders: ``def_pos``/``def_vel`` are (Nd, 2), ``aim`` is
    (2,). Returns the new positions and velocities.
    """
    to_aim = aim[None, :] - def_pos                       # (Nd, 2)
    dist = np.linalg.norm(to_aim, axis=1, keepdims=True)
    desired = np.where(dist > 1e-9, to_aim / np.maximum(dist, 1e-9) * max_speed,
                       0.0)
    dv = desired - def_vel
    dvmag = np.linalg.norm(dv, axis=1, keepdims=True)
    scale = np.minimum(1.0, (max_accel * dt) / np.maximum(dvmag, 1e-9))
    vel = def_vel + dv * scale
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    vel = np.where(speed > max_speed, vel / np.maximum(speed, 1e-9) * max_speed,
                   vel)
    return def_pos + vel * dt, vel


def _run(frame: FrozenFrame, goal: np.ndarray, params: ReactiveParams,
         replan: "callable", replan_every: int = 0) -> RolloutResult:
    """Core stepping loop shared by static and replanning rollouts.

    ``replan(pos, vel) -> path`` computes the carrier's path; it is called once
    up front, and again every ``replan_every`` steps (0 = never = fixed path).
    """
    goal = np.asarray(goal, dtype=float)
    pos = frame.positions.astype(float).copy()
    vel = frame.velocities.astype(float).copy()
    carrier = frame.ball_carrier
    didx = np.where(frame.defend_mask)[0]

    # Initial path.
    path = replan(pos, vel)
    cum = _cumulative_lengths(path)
    arc = 0.0

    goal_d0 = float(np.linalg.norm(pos[carrier] - goal))
    times = [0.0]
    pos_hist = [pos.copy()]
    plan_hist = [path.copy()]
    intercepted = False
    reached = False
    intercept_pt = None
    t = 0.0
    n_steps = int(params.max_time / params.dt)

    for step in range(1, n_steps + 1):
        # Optionally recompute the plan from the current world state.
        if replan_every and step % replan_every == 0:
            path = replan(pos, vel)
            cum = _cumulative_lengths(path)
            arc = 0.0  # new path starts at the current carrier position

        prev_c = pos[carrier].copy()
        arc += params.carrier_speed * params.dt
        new_c, at_end = _point_at_arc(path, cum, arc)
        pos[carrier] = new_c
        vel[carrier] = (new_c - prev_c) / params.dt

        if len(didx):
            aim = new_c + vel[carrier] * params.lead_time
            pos[didx], vel[didx] = pursue(pos[didx], vel[didx], aim,
                                          max_speed=params.def_max_speed,
                                          max_accel=params.def_max_accel,
                                          dt=params.dt)
        t += params.dt
        times.append(t)
        pos_hist.append(pos.copy())
        plan_hist.append(path.copy())

        if len(didx) and np.min(np.linalg.norm(pos[didx] - pos[carrier],
                                               axis=1)) <= params.intercept_radius:
            intercepted = True
            intercept_pt = pos[carrier].copy()
            break
        if at_end and float(np.linalg.norm(pos[carrier] - goal)) < 2.0:
            reached = True
            break

    progress = float(np.clip(
        1.0 - np.linalg.norm(pos[carrier] - goal) / max(goal_d0, 1e-9), 0.0, 1.0))
    return RolloutResult(
        reached_goal=reached, intercepted=intercepted, progress=progress,
        sim_time=t, times=np.array(times), pos_history=np.array(pos_hist),
        plan_history=plan_hist, intercept_point=intercept_pt,
        team_ids=frame.team_ids.copy(), carrier_idx=carrier)


def rollout_static(frame: FrozenFrame, path: np.ndarray, goal: np.ndarray, *,
                   params: ReactiveParams = ReactiveParams()) -> RolloutResult:
    """Roll a fixed ``path`` out against reacting defenders."""
    path = np.asarray(path, dtype=float)
    return _run(frame, goal, params, replan=lambda pos, vel: path)


def rollout_replanning(frame: FrozenFrame, centers: np.ndarray,
                       goal: np.ndarray, *, replan_every: int = 5,
                       weights: CostWeights = DEFAULT_WEIGHTS,
                       params: ReactiveParams = ReactiveParams()) -> RolloutResult:
    """Roll out while re-running A* every ``replan_every`` steps on the update.

    Each replan rebuilds a FrozenFrame from the current positions/velocities,
    recomputes the pitch-control cost map, and plans a fresh smoothed A* path
    from the carrier's current position to the goal.
    """
    carrier = frame.ball_carrier
    team_ids = frame.team_ids

    def replan(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        fr = FrozenFrame(pos.copy(), vel.copy(), team_ids, pos[carrier].copy(),
                         carrier)
        cost = cost_map(fr, centers, weights=weights)
        astar = plan_astar(cost, centers, pos[carrier], goal,
                           terrain_weight=params.terrain_weight)
        return smooth_path(astar.points, cost, centers,
                           terrain_weight=params.terrain_weight).points

    return _run(frame, goal, params, replan=replan, replan_every=replan_every)


if __name__ == "__main__":
    from ..state import TEAM_ATTACK, TEAM_DEFEND

    frame = FrozenFrame(
        positions=[[-25.0, 0.0], [5.0, 7.0], [5.0, -7.0]],
        velocities=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        team_ids=[TEAM_ATTACK, TEAM_DEFEND, TEAM_DEFEND],
        ball_pos=[-25.0, 0.0], ball_carrier=0)
    goal = np.array([45.0, 0.0])
    straight = np.array([frame.carrier_position, goal])
    res = rollout_static(frame, straight, goal)
    print(f"static straight: reached={res.reached_goal} "
          f"intercepted={res.intercepted} progress={res.progress:.0%}")
