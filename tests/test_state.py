"""Tests for the FrozenFrame state object and velocity helpers."""

from __future__ import annotations

import numpy as np
import pytest

from footlab.state import (
    MAX_PLAYER_SPEED,
    TEAM_ATTACK,
    TEAM_DEFEND,
    FrozenFrame,
    clip_speed,
)


def make_frame():
    return FrozenFrame(
        positions=[[0, 0], [10, 5], [20, -3]],
        velocities=[[3, 0], [2, 1], [-4, 0]],
        team_ids=[TEAM_ATTACK, TEAM_ATTACK, TEAM_DEFEND],
        ball_pos=[0, 0],
        ball_carrier=0,
    )


def test_construction_and_views():
    ff = make_frame()
    assert ff.n_players == 3
    assert ff.attack_mask.tolist() == [True, True, False]
    assert ff.defend_mask.tolist() == [False, False, True]
    assert ff.attacker_positions.shape == (2, 2)
    assert ff.defender_positions.shape == (1, 2)
    assert np.allclose(ff.carrier_position, [0, 0])


def test_speeds():
    ff = make_frame()
    assert np.allclose(ff.speeds, [3.0, np.hypot(2, 1), 4.0])


def test_loose_ball_has_no_carrier_position():
    ff = make_frame()
    ff.ball_carrier = -1
    assert ff.carrier_position is None


def test_arrays_are_coerced_to_float():
    ff = make_frame()
    assert ff.positions.dtype == float
    assert ff.velocities.dtype == float


@pytest.mark.parametrize("bad", [
    dict(velocities=[[1, 1]]),                      # wrong velocity count
    dict(team_ids=[0, 0]),                          # wrong team_ids length
    dict(team_ids=[0, 1, 2]),                       # invalid team id
    dict(ball_carrier=5),                           # carrier out of range
])
def test_validation_rejects_malformed_frames(bad):
    kwargs = dict(
        positions=[[0, 0], [10, 5], [20, -3]],
        velocities=[[3, 0], [2, 1], [-4, 0]],
        team_ids=[TEAM_ATTACK, TEAM_ATTACK, TEAM_DEFEND],
        ball_pos=[0, 0],
        ball_carrier=0,
    )
    kwargs.update(bad)
    with pytest.raises(ValueError):
        FrozenFrame(**kwargs)


def test_copy_is_independent():
    ff = make_frame()
    other = ff.copy()
    other.positions[0, 0] = 99.0
    assert ff.positions[0, 0] == 0.0  # original untouched


def test_clip_speed_caps_magnitude_and_preserves_direction():
    v = np.array([[12.0, 0.0], [0.0, 3.0], [6.0, 8.0]])  # speeds 12, 3, 10
    out = clip_speed(v, MAX_PLAYER_SPEED)
    speeds = np.linalg.norm(out, axis=1)
    assert speeds[0] == pytest.approx(MAX_PLAYER_SPEED)   # capped
    assert speeds[1] == pytest.approx(3.0)                # untouched
    assert speeds[2] == pytest.approx(MAX_PLAYER_SPEED)   # capped
    # Direction preserved for a capped vector.
    assert np.allclose(out[0] / np.linalg.norm(out[0]), [1.0, 0.0])


def test_clip_speed_single_vector():
    out = clip_speed([100.0, 0.0], MAX_PLAYER_SPEED)
    assert out.shape == (2,)
    assert np.linalg.norm(out) == pytest.approx(MAX_PLAYER_SPEED)
