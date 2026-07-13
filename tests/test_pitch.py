"""Tests for pitch constants, coordinate helpers, and rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

from footlab import pitch


def test_dimensions_and_landmarks():
    assert pitch.PITCH_LENGTH == 105.0
    assert pitch.PITCH_WIDTH == 68.0
    assert pitch.HALF_LENGTH == 52.5
    assert pitch.HALF_WIDTH == 34.0
    # Attacking team plays toward +x.
    assert np.allclose(pitch.ATTACKING_GOAL_CENTER, [52.5, 0.0])
    assert np.allclose(pitch.DEFENDING_GOAL_CENTER, [-52.5, 0.0])


def test_is_on_pitch_scalar():
    assert pitch.is_on_pitch([0.0, 0.0]) is True
    assert pitch.is_on_pitch([52.5, 34.0]) is True          # corner, on line
    assert pitch.is_on_pitch([52.51, 0.0]) is False         # just past goal line
    assert pitch.is_on_pitch([0.0, 40.0]) is False          # off the side


def test_is_on_pitch_vectorized():
    pts = np.array([[0, 0], [60, 0], [0, 40], [-52.5, -34.0]])
    result = pitch.is_on_pitch(pts)
    assert result.dtype == bool
    assert result.tolist() == [True, False, False, True]


def test_clip_to_pitch():
    pts = np.array([[100.0, 0.0], [0.0, -50.0], [10.0, 10.0]])
    clipped = pitch.clip_to_pitch(pts)
    assert np.all(pitch.is_on_pitch(clipped))
    assert clipped[0, 0] == 52.5
    assert clipped[1, 1] == -34.0
    # Interior points are untouched.
    assert np.allclose(clipped[2], [10.0, 10.0])


def test_draw_pitch_returns_axes_without_error():
    import matplotlib.pyplot as plt

    ax = pitch.draw_pitch()
    assert ax.get_aspect() == 1.0  # equal aspect
    # Line artists were actually drawn.
    assert len(ax.lines) > 0
    plt.close("all")
