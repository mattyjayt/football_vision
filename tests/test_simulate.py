"""Tests for scenario generators and augmentation helpers.

Asserts the Phase 0 invariants: player counts, everyone on the pitch, speed caps
respected, the carrier is an attacker, scenarios are deterministic, and the
augmentation helpers behave as documented.
"""

from __future__ import annotations

import numpy as np
import pytest

from footlab import pitch, simulate
from footlab.state import MAX_PLAYER_SPEED, TEAM_ATTACK


ALL = [
    simulate.scenario_counter_attack,
    simulate.scenario_low_block,
    simulate.scenario_wing_overload,
    lambda: simulate.scenario_random(seed=0),
]


@pytest.mark.parametrize("make", ALL)
def test_frame_invariants(make):
    ff = make()
    # Everyone (and the ball) is on the pitch.
    assert np.all(pitch.is_on_pitch(ff.positions))
    assert pitch.is_on_pitch(ff.ball_pos)
    # Speed cap respected (allow tiny float slack).
    assert ff.speeds.max() <= MAX_PLAYER_SPEED + 1e-9
    # The carrier exists and is an attacker.
    assert ff.ball_carrier >= 0
    assert ff.team_ids[ff.ball_carrier] == TEAM_ATTACK
    # Ball sits at the carrier's feet.
    assert np.allclose(ff.ball_pos, ff.positions[ff.ball_carrier])


def test_scenario_player_counts():
    assert simulate.scenario_counter_attack().n_players == 6      # 3 + 2 + GK
    assert simulate.scenario_wing_overload().n_players == 5       # 3v2
    lb = simulate.scenario_low_block()
    assert lb.defend_mask.sum() == 10                             # 10 defenders
    assert simulate.scenario_random().n_players == 22             # 11v11


def test_random_is_deterministic():
    a = simulate.scenario_random(seed=42)
    b = simulate.scenario_random(seed=42)
    assert np.allclose(a.positions, b.positions)
    assert np.allclose(a.velocities, b.velocities)
    assert a.ball_carrier == b.ball_carrier
    # Different seed => different frame.
    c = simulate.scenario_random(seed=7)
    assert not np.allclose(a.positions, c.positions)


def test_mirror_lengthwise_preserves_roles_and_flips_y():
    ff = simulate.scenario_wing_overload()
    m = simulate.mirror_lengthwise(ff)
    assert np.allclose(m.positions[:, 1], -ff.positions[:, 1])
    assert np.allclose(m.positions[:, 0], ff.positions[:, 0])
    assert np.array_equal(m.team_ids, ff.team_ids)  # roles unchanged
    assert np.all(pitch.is_on_pitch(m.positions))


def test_mirror_widthwise_flips_x_and_swaps_teams():
    ff = simulate.scenario_wing_overload()
    m = simulate.mirror_widthwise(ff)
    assert np.allclose(m.positions[:, 0], -ff.positions[:, 0])
    # Teams swapped so the attack->+x convention is preserved.
    assert np.array_equal(m.team_ids, 1 - ff.team_ids)
    assert np.all(pitch.is_on_pitch(m.positions))


def test_jitter_keeps_players_on_pitch():
    ff = simulate.scenario_random(seed=0)
    j = simulate.jitter_positions(ff, sigma=3.0, seed=1)
    assert np.all(pitch.is_on_pitch(j.positions))
    assert pitch.is_on_pitch(j.ball_pos)
    # Something actually moved.
    assert not np.allclose(j.positions, ff.positions)


def test_perturb_velocities_respects_speed_cap():
    ff = simulate.scenario_counter_attack()
    p = simulate.perturb_velocities(ff, sigma=5.0, seed=3)
    assert p.speeds.max() <= MAX_PLAYER_SPEED + 1e-9
    assert not np.allclose(p.velocities, ff.velocities)


def test_augmentation_is_nondestructive():
    ff = simulate.scenario_counter_attack()
    before = ff.positions.copy()
    simulate.mirror_lengthwise(ff)
    simulate.jitter_positions(ff, sigma=1.0, seed=0)
    assert np.allclose(ff.positions, before)  # original frame untouched
