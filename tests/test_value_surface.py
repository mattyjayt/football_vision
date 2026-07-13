"""Tests for Phase 2b value surface, defender penalty, and cost map."""

from __future__ import annotations

import numpy as np
import pytest

from footlab import pitch, simulate, value_surface
from footlab.value_surface import (
    CostWeights,
    cost_map,
    defender_proximity_penalty,
    positional_value_surface,
    value_at_point,
)


# --- value surface -----------------------------------------------------------
def test_value_in_unit_range():
    _, _, centers = pitch.make_grid(cell_size=2.0)
    value = positional_value_surface(centers)
    assert value.min() >= 0.0
    assert value.max() <= 1.0


def test_value_increases_toward_goal():
    near = value_at_point([50.0, 0.0])
    mid = value_at_point([0.0, 0.0])
    own = value_at_point([-40.0, 0.0])
    assert near > mid > own


def test_value_is_central_biased_near_goal():
    # Straight in front of goal is worth more than a wide angle at the same x.
    central = value_at_point([48.0, 0.0])
    wide = value_at_point([48.0, 30.0])
    assert central > wide


def test_box_bonus():
    # Inside the box beats a point just outside it at the same y.
    inside = value_at_point([40.0, 0.0])       # box edge is x = 36
    outside = value_at_point([30.0, 0.0])
    assert inside > outside


def test_surface_matches_pointwise_value():
    xs, ys, centers = pitch.make_grid(cell_size=5.0)
    surf = positional_value_surface(centers)
    for i in (0, 3, 6):
        for j in (0, 5, 10, 15):
            assert surf[i, j] == pytest.approx(value_at_point(centers[i, j]))


# --- defender penalty --------------------------------------------------------
def test_penalty_peaks_at_a_defender_and_decays():
    frame = simulate.scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    pen = defender_proximity_penalty(frame, centers)
    assert pen.min() >= 0.0
    assert pen.max() <= 1.0
    # A cell centered on a defender should be near the maximum penalty (~1).
    d = frame.defender_positions[0]
    # nearest grid cell to that defender
    xs, ys, _ = pitch.make_grid(cell_size=1.0)
    j = int(np.argmin(np.abs(xs - d[0])))
    i = int(np.argmin(np.abs(ys - d[1])))
    assert pen[i, j] > 0.9
    # Far corner is essentially unpenalized.
    assert pen[0, 0] < 0.1


def test_penalty_zero_without_defenders():
    from footlab.state import TEAM_ATTACK, FrozenFrame

    frame = FrozenFrame(
        positions=[[0.0, 0.0], [5.0, 0.0]],
        velocities=[[0.0, 0.0], [0.0, 0.0]],
        team_ids=[TEAM_ATTACK, TEAM_ATTACK],
        ball_pos=[0.0, 0.0],
        ball_carrier=0,
    )
    _, _, centers = pitch.make_grid(cell_size=5.0)
    pen = defender_proximity_penalty(frame, centers)
    assert np.all(pen == 0.0)


# --- cost map ----------------------------------------------------------------
def test_cost_map_is_nonnegative_and_finite():
    frame = simulate.scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=2.0)
    cost = cost_map(frame, centers)
    assert np.all(np.isfinite(cost))
    assert cost.min() >= 0.0


def test_zero_weights_give_zero_cost():
    frame = simulate.scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=3.0)
    cost = cost_map(frame, centers, weights=CostWeights(0.0, 0.0, 0.0))
    assert np.allclose(cost, 0.0)


def test_defender_weight_raises_cost_near_defenders():
    frame = simulate.scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    control = None  # let it compute once per call is fine at this grid size
    low = cost_map(frame, centers, weights=CostWeights(1.0, 1.0, 0.0))
    high = cost_map(frame, centers, weights=CostWeights(1.0, 1.0, 3.0))
    # Extra defender weight can only add non-negative penalty, never reduce cost.
    assert np.all(high >= low - 1e-9)
    # And it must strictly increase cost somewhere near the defenders.
    assert high.max() > low.max()


def test_value_weight_makes_goal_cheaper():
    frame = simulate.scenario_counter_attack()
    xs, ys, centers = pitch.make_grid(cell_size=1.0)
    control = np.ones(centers.shape[:2])  # neutralize the control term
    cost = cost_map(frame, centers, control=control,
                    weights=CostWeights(0.0, 1.0, 0.0))
    # With only the value term, cost near goal should be lower than at halfway.
    jg = int(np.argmin(np.abs(xs - 50.0)))
    jm = int(np.argmin(np.abs(xs - 0.0)))
    ymid = int(np.argmin(np.abs(ys - 0.0)))
    assert cost[ymid, jg] < cost[ymid, jm]
