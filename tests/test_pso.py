"""Tests for Phase 5 PSO over Bezier trajectories."""

from __future__ import annotations

import numpy as np
import pytest

from footlab import pitch
from footlab.planners.grid_search import path_cost
from footlab.planners.pso_path import (
    PSOParams,
    bezier_curve,
    optimize_bezier,
    path_fitness,
)


def make_grid(ny: int, nx: int, cs: float = 1.0):
    xs = np.arange(nx) * cs
    ys = np.arange(ny) * cs
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, np.stack([xx, yy], axis=-1)


# --- Bezier ------------------------------------------------------------------
def test_bezier_hits_endpoints():
    p0, p1, p2, p3 = (np.array(p, float) for p in
                      ([0, 0], [1, 2], [3, -1], [4, 0]))
    curve = bezier_curve(p0, p1, p2, p3, n=25)
    assert np.allclose(curve[0], p0)
    assert np.allclose(curve[-1], p3)


def test_bezier_colinear_controls_give_straight_line():
    # Control points on the segment p0->p3 => the curve is that straight segment.
    p0, p3 = np.array([0.0, 0.0]), np.array([9.0, 0.0])
    p1, p2 = np.array([3.0, 0.0]), np.array([6.0, 0.0])
    curve = bezier_curve(p0, p1, p2, p3, n=20)
    assert np.allclose(curve[:, 1], 0.0)  # never leaves the x-axis


# --- fitness -----------------------------------------------------------------
def test_fitness_is_finite_and_returns_points():
    _, _, centers = make_grid(20, 30)
    cost = np.ones((20, 30))
    flat = np.array([10.0, 5.0, 20.0, 5.0])
    fit, pts = path_fitness(flat, [0.0, 0.0], [29.0, 0.0], cost, centers)
    assert np.isfinite(fit)
    assert pts.shape[0] == PSOParams().n_samples


# --- optimizer ---------------------------------------------------------------
def _uniform_setup():
    _, _, centers = pitch.make_grid(cell_size=3.0)
    cost = np.ones(centers.shape[:2])  # featureless field
    start = np.array([-40.0, 0.0])
    goal = np.array([40.0, 0.0])
    return cost, centers, start, goal


def test_history_is_monotonically_non_increasing():
    cost, centers, start, goal = _uniform_setup()
    res = optimize_bezier(cost, centers, start, goal,
                          params=PSOParams(n_particles=20, n_iters=40))
    assert np.all(np.diff(res.history) <= 1e-9)


def test_deterministic_with_seed():
    cost, centers, start, goal = _uniform_setup()
    p = PSOParams(n_particles=20, n_iters=30, seed=7)
    a = optimize_bezier(cost, centers, start, goal, params=p)
    b = optimize_bezier(cost, centers, start, goal, params=p)
    assert a.fitness == pytest.approx(b.fitness)
    assert np.allclose(a.control_points, b.control_points)


def test_uniform_field_yields_near_straight_path():
    cost, centers, start, goal = _uniform_setup()
    res = optimize_bezier(cost, centers, start, goal,
                          params=PSOParams(n_particles=40, n_iters=80))
    # On a featureless field the cheapest path is the straight line.
    straight = path_cost(np.array([start, goal]), cost, centers)
    assert res.fitness <= straight * 1.10
    # And it should be nearly straight (small max deviation off the x-axis).
    assert np.max(np.abs(res.points[:, 1])) < 3.0


def test_optimizer_improves_over_iterations():
    cost, centers, start, goal = _uniform_setup()
    res = optimize_bezier(cost, centers, start, goal,
                          params=PSOParams(n_particles=25, n_iters=50))
    assert res.history[-1] <= res.history[0]


def test_result_path_stays_on_pitch_on_uniform_field():
    cost, centers, start, goal = _uniform_setup()
    res = optimize_bezier(cost, centers, start, goal,
                          params=PSOParams(n_particles=40, n_iters=80))
    assert np.all(pitch.is_on_pitch(res.points))
