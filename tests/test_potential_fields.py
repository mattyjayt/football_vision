"""Tests for Phase 4 artificial potential fields."""

from __future__ import annotations

import numpy as np
import pytest

from footlab.planners import potential_fields as pf
from footlab.planners.potential_fields import (
    DEFAULT_PARAMS,
    attractive_force,
    descend,
    repulsive_force,
)
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame


def _frame(defenders):
    positions = [[-35.0, 0.0]] + [list(map(float, d)) for d in defenders]
    team_ids = [TEAM_ATTACK] + [TEAM_DEFEND] * len(defenders)
    return FrozenFrame(positions, [[0.0, 0.0]] * len(positions), team_ids,
                       [-35.0, 0.0], ball_carrier=0)


# --- attractive force --------------------------------------------------------
def test_attractive_points_toward_goal():
    goal = np.array([10.0, 0.0])
    f = attractive_force(np.array([0.0, 0.0]), goal, k_att=1.0, d0=5.0)
    assert f[0] > 0 and abs(f[1]) < 1e-9  # points +x toward goal


def test_attractive_far_field_is_constant_magnitude():
    goal = np.array([100.0, 0.0])
    f = attractive_force(np.array([0.0, 0.0]), goal, k_att=2.0, d0=5.0)
    # Far field magnitude is k_att * d0.
    assert np.linalg.norm(f) == pytest.approx(2.0 * 5.0)


def test_attractive_near_field_is_linear():
    goal = np.array([3.0, 0.0])  # within d0 = 5
    f = attractive_force(np.array([0.0, 0.0]), goal, k_att=1.0, d0=5.0)
    assert np.linalg.norm(f) == pytest.approx(3.0)  # k_att * dist


def test_attractive_zero_at_goal():
    goal = np.array([5.0, 5.0])
    f = attractive_force(goal.copy(), goal, k_att=1.0, d0=5.0)
    assert np.allclose(f, 0.0)


# --- repulsive force ---------------------------------------------------------
def test_repulsion_points_away_and_zero_beyond_influence():
    obs = np.array([[10.0, 0.0]])
    # Inside influence (rho0 default 12): pushed away (-x, since pos left of obs).
    f_in = repulsive_force(np.array([5.0, 0.0]), obs, k_rep=300.0, rho0=12.0)
    assert f_in[0] < 0
    # Beyond influence: exactly zero.
    f_out = repulsive_force(np.array([-10.0, 0.0]), obs, k_rep=300.0, rho0=12.0)
    assert np.allclose(f_out, 0.0)


def test_repulsion_grows_as_you_approach():
    obs = np.array([[10.0, 0.0]])
    near = np.linalg.norm(repulsive_force(np.array([8.0, 0.0]), obs,
                                          k_rep=300.0, rho0=12.0))
    far = np.linalg.norm(repulsive_force(np.array([2.0, 0.0]), obs,
                                         k_rep=300.0, rho0=12.0))
    assert near > far  # closer (8 vs 2, i.e. rho 2 vs 8) => stronger


# --- gradient descent --------------------------------------------------------
def test_reaches_goal_in_empty_field():
    frame = _frame([])  # no defenders
    res = descend(frame, np.array([30.0, 0.0]))
    assert res.reached_goal
    assert res.reason == "goal"
    assert np.linalg.norm(res.points[-1] - [30.0, 0.0]) < DEFAULT_PARAMS.goal_tol


def test_curves_around_offset_defender_and_reaches_goal():
    frame = _frame([[0, 3]])
    res = descend(frame, np.array([45.0, 0.0]))
    assert res.reached_goal
    # It had to bend off the straight line to get by.
    assert np.max(np.abs(res.points[:, 1])) > 0.5


def test_colinear_obstacle_traps_in_local_minimum():
    frame = _frame([[5, 0]])  # defender directly between carrier and goal
    res = descend(frame, np.array([45.0, 0.0]))
    assert not res.reached_goal
    assert res.reason == "local_minimum"
    # Stuck in open space, short of the obstacle and the goal.
    assert res.points[-1, 0] < 5.0


def test_concave_cup_is_a_stable_trap():
    frame = _frame([[2, 0], [8, 3], [8, -3]])
    res = descend(frame, np.array([45.0, 0.0]))
    assert not res.reached_goal
    assert res.reason == "local_minimum"


def test_force_field_finite_over_grid():
    frame = _frame([[0, 3], [20, -4]])
    obstacles = pf.projected_defenders(frame, DEFAULT_PARAMS.project_time)
    gx, gy = np.meshgrid(np.linspace(-40, 50, 25), np.linspace(-25, 25, 15))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    F = pf.force_field(pts, np.array([45.0, 0.0]), obstacles)
    assert np.all(np.isfinite(F))
