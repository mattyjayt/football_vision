"""Tests for Phase 1 space geometry (Voronoi + dominant regions)."""

from __future__ import annotations

import numpy as np
import pytest

from footlab import geometry, pitch
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame


def two_player_frame(v0=(0.0, 0.0), v1=(0.0, 0.0)):
    """Attacker at (-10, 0), defender at (+10, 0), with chosen velocities."""
    return FrozenFrame(
        positions=[[-10.0, 0.0], [10.0, 0.0]],
        velocities=[list(v0), list(v1)],
        team_ids=[TEAM_ATTACK, TEAM_DEFEND],
        ball_pos=[-10.0, 0.0],
        ball_carrier=0,
    )


# --- time_to_arrive ----------------------------------------------------------
def test_time_to_arrive_known_value():
    # One stationary player at the origin, target 5 m away, speed 5 m/s.
    t = geometry.time_to_arrive(
        positions=[[0.0, 0.0]], velocities=[[0.0, 0.0]], targets=[[5.0, 0.0]],
        reaction_time=0.7, max_speed=5.0)
    assert t.shape == (1, 1)
    assert t[0, 0] == pytest.approx(0.7 + 5.0 / 5.0)  # 1.7 s


def test_velocity_gives_a_head_start():
    # Target ahead of the player; moving toward it must arrive sooner than still.
    target = [[10.0, 0.0]]
    still = geometry.time_to_arrive([[0.0, 0.0]], [[0.0, 0.0]], target)
    moving = geometry.time_to_arrive([[0.0, 0.0]], [[5.0, 0.0]], target)
    away = geometry.time_to_arrive([[0.0, 0.0]], [[-5.0, 0.0]], target)
    assert moving[0, 0] < still[0, 0] < away[0, 0]


# --- control maps ------------------------------------------------------------
def test_voronoi_assigns_by_nearest():
    frame = two_player_frame()
    _, _, centers = pitch.make_grid(cell_size=3.0)
    vor = geometry.voronoi_control(frame, centers)
    assert vor.shape == centers.shape[:2]
    # Every cell is a valid player index.
    assert set(np.unique(vor)).issubset({0, 1})
    # Left half of the pitch belongs to the attacker (index 0), right to defender.
    xs, _, _ = pitch.make_grid(cell_size=3.0)
    left_cols = xs < -1.0
    right_cols = xs > 1.0
    assert np.all(vor[:, left_cols] == 0)
    assert np.all(vor[:, right_cols] == 1)


def test_dominant_equals_voronoi_when_velocities_zero():
    frame = two_player_frame(v0=(0, 0), v1=(0, 0))
    _, _, centers = pitch.make_grid(cell_size=2.0)
    vor = geometry.voronoi_control(frame, centers)
    dom = geometry.dominant_region(frame, centers)
    # With no velocity, soonest-to-arrive collapses to nearest, so maps match.
    assert np.array_equal(vor, dom)


def test_velocity_shifts_control_boundary():
    # Defender charging toward the attacker's half should claim extra ground
    # versus the static Voronoi split.
    frame = two_player_frame(v1=(-8.0, 0.0))  # defender moving toward -x
    _, _, centers = pitch.make_grid(cell_size=1.0)
    vor = geometry.voronoi_control(frame, centers)
    dom = geometry.dominant_region(frame, centers)
    assert not np.array_equal(vor, dom)
    # Defender (team 1) controls strictly more area in the dominant map.
    assert (geometry.team_area_fraction(frame, dom, team=TEAM_DEFEND)
            > geometry.team_area_fraction(frame, vor, team=TEAM_DEFEND))


def test_reaction_time_offset_does_not_change_assignment_when_static():
    frame = two_player_frame()
    _, _, centers = pitch.make_grid(cell_size=2.0)
    a = geometry.dominant_region(frame, centers, reaction_time=0.7)
    b = geometry.dominant_region(frame, centers, reaction_time=0.0)
    # A constant reaction delay cancels in the argmin when nobody is moving.
    assert np.array_equal(a, b)


def test_control_to_team_and_area_fraction():
    frame = two_player_frame()
    _, _, centers = pitch.make_grid(cell_size=2.0)
    vor = geometry.voronoi_control(frame, centers)
    team_map = geometry.control_to_team(frame, vor)
    assert team_map.shape == vor.shape
    assert set(np.unique(team_map)).issubset({TEAM_ATTACK, TEAM_DEFEND})
    # Symmetric setup => each team controls about half.
    assert geometry.team_area_fraction(frame, vor) == pytest.approx(0.5, abs=0.02)


def test_full_grid_shapes_and_validity():
    from footlab import simulate

    frame = simulate.scenario_random(seed=0)
    _, _, centers = pitch.make_grid(cell_size=1.0)
    dom = geometry.dominant_region(frame, centers)
    assert dom.shape == (68, 105)
    # Every cell assigned to a real player.
    assert dom.min() >= 0
    assert dom.max() < frame.n_players
