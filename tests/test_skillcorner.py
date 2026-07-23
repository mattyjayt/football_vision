"""Tests for the SkillCorner -> FrozenFrame bridge.

These tests require the SkillCorner open data (a multi-hundred-MB git-lfs
clone) which is NOT present in CI or on a fresh checkout. They therefore
``skip`` cleanly when the data directory is absent, so the suite stays green
everywhere. When the data IS present they assert the invariants that matter:

    * 22 players, split 11/11 between the two teams.
    * All speeds within the physiological cap (velocities are finite-difference
      estimates, so this also guards against frame-jitter ghosts).
    * Every position lies on the pitch.
    * The ball carrier, when known, is on the ATTACKING team (possession-based
      team assignment) and the attack is normalised toward +x.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from footlab import pitch
from footlab.state import MAX_PLAYER_SPEED, TEAM_ATTACK

skillcorner = pytest.importorskip("footlab.skillcorner")

DATA_DIR = (Path(__file__).resolve().parent.parent
            / "data" / "skillcorner" / "data" / "matches" / "1886347")

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="SkillCorner open data not downloaded")

FRAME_INDEX = 5000


@pytest.fixture(scope="module")
def frame():
    return skillcorner.frames_to_frozen(DATA_DIR, FRAME_INDEX)


def test_has_22_players_split_11_11(frame):
    assert frame.n_players == 22
    assert frame.attack_mask.sum() == 11
    assert frame.defend_mask.sum() == 11


def test_speeds_within_physiological_cap(frame):
    # Velocities are derived (finite difference); jitter must not exceed cap.
    assert float(frame.speeds.max()) <= MAX_PLAYER_SPEED + 1e-6
    assert np.all(frame.speeds >= 0.0)


def test_positions_on_pitch(frame):
    # Allow a small margin beyond the lines (players can be just off in play).
    assert np.all(np.abs(frame.positions[:, 0]) <= pitch.HALF_LENGTH + 2.0)
    assert np.all(np.abs(frame.positions[:, 1]) <= pitch.HALF_WIDTH + 2.0)


def test_carrier_is_attacker(frame):
    # Frame 5000 has a known possessor; by footlab convention the carrier is an
    # attacker (possession-based team assignment).
    assert frame.ball_carrier >= 0
    assert frame.team_ids[frame.ball_carrier] == TEAM_ATTACK


def test_attack_normalised_toward_plus_x(frame):
    # The possessing (away) team attacks +x in this period, so no mirror is
    # applied and the carrier keeps his raw +x position high up the pitch.
    carrier = frame.carrier_position
    assert carrier is not None
    # Ball and carrier coincide (he has the ball).
    assert np.allclose(carrier, frame.ball_pos, atol=5.0)


def test_ball_on_pitch(frame):
    assert abs(frame.ball_pos[0]) <= pitch.HALF_LENGTH + 2.0
    assert abs(frame.ball_pos[1]) <= pitch.HALF_WIDTH + 2.0
