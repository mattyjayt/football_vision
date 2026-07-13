"""Tests for Phase 6 reactive rollout / replanning."""

from __future__ import annotations

import numpy as np
import pytest

from footlab import pitch
from footlab.planners import reactive as rc
from footlab.planners.reactive import (
    ReactiveParams,
    _point_at_arc,
    pursue,
    rollout_replanning,
    rollout_static,
)
from footlab.planners.reactive import _cumulative_lengths
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame


def _frame(defenders):
    positions = [[-25.0, 0.0]] + [list(map(float, d)) for d in defenders]
    team_ids = [TEAM_ATTACK] + [TEAM_DEFEND] * len(defenders)
    return FrozenFrame(positions, [[0.0, 0.0]] * len(positions), team_ids,
                       [-25.0, 0.0], ball_carrier=0)


# --- arc-length helper -------------------------------------------------------
def test_point_at_arc():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cum = _cumulative_lengths(path)
    p0, end0 = _point_at_arc(path, cum, 0.0)
    p1, _ = _point_at_arc(path, cum, 5.0)
    p2, _ = _point_at_arc(path, cum, 15.0)
    pend, ended = _point_at_arc(path, cum, 100.0)
    assert np.allclose(p0, [0, 0]) and not end0
    assert np.allclose(p1, [5, 0])
    assert np.allclose(p2, [10, 5])
    assert np.allclose(pend, [10, 10]) and ended


# --- defender pursuit --------------------------------------------------------
def test_pursue_accelerates_toward_aim():
    pos = np.array([[0.0, 0.0]])
    vel = np.array([[0.0, 0.0]])
    new_pos, new_vel = pursue(pos, vel, np.array([10.0, 0.0]),
                              max_speed=5.0, max_accel=4.0, dt=0.2)
    assert new_vel[0, 0] == pytest.approx(0.8)   # accel-limited: 4 * 0.2
    assert new_pos[0, 0] == pytest.approx(0.16)
    assert new_vel[0, 1] == pytest.approx(0.0)


def test_pursue_respects_speed_cap():
    pos = np.array([[0.0, 0.0]])
    vel = np.array([[10.0, 0.0]])   # already over the cap
    _, new_vel = pursue(pos, vel, np.array([10.0, 0.0]),
                        max_speed=5.0, max_accel=4.0, dt=0.2)
    assert np.linalg.norm(new_vel[0]) <= 5.0 + 1e-9


# --- rollouts ----------------------------------------------------------------
def test_static_reaches_goal_when_defender_cannot_catch():
    # Defender starts behind the carrier: chasing an equal-speed target with a
    # big head start, it can never close.
    frame = _frame([[-45.0, 8.0]])
    goal = np.array([45.0, 0.0])
    path = np.array([frame.carrier_position, goal])
    res = rollout_static(frame, path, goal, params=ReactiveParams(max_time=20.0))
    assert res.reached_goal
    assert not res.intercepted
    assert res.progress > 0.95


def test_no_defenders_reaches_goal():
    frame = FrozenFrame([[-25.0, 0.0]], [[0.0, 0.0]], [TEAM_ATTACK],
                        [-25.0, 0.0], ball_carrier=0)
    goal = np.array([45.0, 0.0])
    res = rollout_static(frame, np.array([[-25.0, 0.0], goal]), goal)
    assert res.reached_goal


def test_static_gets_intercepted_when_defender_blocks():
    frame = _frame([[5.0, 0.0]])   # defender directly on the straight line
    goal = np.array([45.0, 0.0])
    path = np.array([frame.carrier_position, goal])
    res = rollout_static(frame, path, goal)
    assert res.intercepted
    assert not res.reached_goal
    assert res.progress < 1.0
    assert res.intercept_point is not None


def test_history_shapes_are_consistent():
    frame = _frame([[5.0, 0.0]])
    goal = np.array([45.0, 0.0])
    path = np.array([frame.carrier_position, goal])
    res = rollout_static(frame, path, goal)
    T = len(res.times)
    assert res.pos_history.shape == (T, frame.n_players, 2)
    assert len(res.plan_history) == T


def test_replanning_runs_and_returns_valid_result():
    # Coarse grid keeps the cost-map recompute cheap for the test.
    _, _, centers = pitch.make_grid(cell_size=3.0)
    frame = _frame([[5.0, 3.0], [5.0, -3.0]])
    goal = np.array([45.0, 0.0])
    res = rollout_replanning(frame, centers, goal, replan_every=4,
                             params=ReactiveParams(max_time=10.0))
    assert 0.0 <= res.progress <= 1.0
    assert res.pos_history.shape[1:] == (frame.n_players, 2)
    assert isinstance(res.reached_goal, bool)
