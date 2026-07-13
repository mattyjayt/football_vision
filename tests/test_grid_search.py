"""Tests for Phase 3 grid search (Dijkstra / A* / smoothing)."""

from __future__ import annotations

import numpy as np
import pytest

from footlab.planners import grid_search
from footlab.planners.grid_search import (
    path_cost,
    plan_astar,
    plan_dijkstra,
    smooth_path,
    world_to_cell,
)


def make_grid(ny: int, nx: int, cell_size: float = 1.0):
    """Small standalone centers grid (origin at 0) for controlled tests."""
    xs = np.arange(nx) * cell_size
    ys = np.arange(ny) * cell_size
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, np.stack([xx, yy], axis=-1)


def test_world_to_cell_snaps_to_nearest():
    _, _, centers = make_grid(5, 7)
    assert world_to_cell(centers, [0.0, 0.0]) == (0, 0)
    assert world_to_cell(centers, [6.4, 3.9]) == (4, 6)


def test_astar_matches_dijkstra_cost():
    """The core correctness guarantee: A* returns the optimal (== Dijkstra) cost."""
    rng = np.random.default_rng(0)
    _, _, centers = make_grid(15, 20)
    cost = rng.uniform(0.0, 3.0, size=(15, 20))
    start, goal = [0.0, 0.0], [19.0, 14.0]
    dij = plan_dijkstra(cost, centers, start, goal)
    ast = plan_astar(cost, centers, start, goal)
    assert ast.cost == pytest.approx(dij.cost, rel=1e-9)


def test_astar_expands_no_more_than_dijkstra():
    rng = np.random.default_rng(1)
    _, _, centers = make_grid(20, 30)
    cost = rng.uniform(0.0, 1.0, size=(20, 30))
    dij = plan_dijkstra(cost, centers, [0.0, 0.0], [29.0, 19.0])
    ast = plan_astar(cost, centers, [0.0, 0.0], [29.0, 19.0])
    assert ast.n_expanded <= dij.n_expanded


def test_path_endpoints_and_adjacency():
    _, _, centers = make_grid(12, 12)
    cost = np.ones((12, 12))
    ast = plan_astar(cost, centers, [0.0, 0.0], [11.0, 11.0])
    assert tuple(ast.cells[0]) == (0, 0)
    assert tuple(ast.cells[-1]) == (11, 11)
    # 8-connected: consecutive cells differ by at most 1 in each index.
    steps = np.abs(np.diff(ast.cells, axis=0))
    assert steps.max() <= 1


def test_reported_cost_matches_line_integral_orthogonal():
    # For an orthogonal-only path, bilinear sampling along a grid line is linear
    # between endpoints, so the continuous line integral equals the search's
    # endpoint-average edge cost. (Diagonal steps differ, by design: a diagonal
    # cuts through the other two corners of the 2x2 block.)
    rng = np.random.default_rng(2)
    _, _, centers = make_grid(10, 15)
    cost = rng.uniform(0.0, 2.0, size=(10, 15))
    ast = plan_astar(cost, centers, [0.0, 0.0], [14.0, 9.0], diagonal=False)
    integ = path_cost(ast.points, cost, centers, step=0.25)
    assert integ == pytest.approx(ast.cost, rel=1e-3)


def test_path_prefers_cheap_corridor():
    # Expensive field with one cheap horizontal corridor along row 5.
    ny, nx = 11, 21
    cost = np.full((ny, nx), 5.0)
    cost[5, :] = 0.05
    _, ys, centers = make_grid(ny, nx)
    start = [0.0, ys[5]]
    goal = [float(nx - 1), ys[5]]
    ast = plan_astar(cost, centers, start, goal, terrain_weight=5.0)
    # The path should hug the corridor (row 5), never straying far.
    assert np.all(np.abs(ast.cells[:, 0] - 5) <= 1)


def test_start_equals_goal():
    _, _, centers = make_grid(6, 6)
    cost = np.ones((6, 6))
    ast = plan_astar(cost, centers, [2.0, 2.0], [2.0, 2.0])
    assert len(ast.cells) == 1
    assert ast.cost == pytest.approx(0.0)


def test_smoothing_reduces_points_without_inflating_cost():
    # A cheap L-shaped route that a straight-ish shortcut can improve/preserve.
    ny, nx = 15, 15
    cost = np.ones((ny, nx)) * 0.1
    _, _, centers = make_grid(ny, nx)
    ast = plan_astar(cost, centers, [0.0, 0.0], [14.0, 14.0])
    sm = smooth_path(ast.points, cost, centers, tol=0.05)
    assert len(sm.points) <= len(ast.points)
    assert sm.cost <= ast.cost * 1.05 + 1e-9


def test_smoothing_respects_expensive_terrain():
    # A wall of high cost between start and goal: the straight shortcut must NOT
    # be taken; the smoothed path should still detour (cost stays well below the
    # naive straight line through the wall).
    ny, nx = 21, 21
    cost = np.ones((ny, nx)) * 0.1
    cost[8:13, 10] = 50.0  # vertical wall segment
    _, _, centers = make_grid(ny, nx)
    ast = plan_astar(cost, centers, [0.0, 10.0], [20.0, 10.0], terrain_weight=5.0)
    sm = smooth_path(ast.points, cost, centers, terrain_weight=5.0, tol=0.05)
    straight = path_cost(np.array([[0.0, 10.0], [20.0, 10.0]]), cost, centers,
                         terrain_weight=5.0)
    assert sm.cost < straight  # detour preserved, wall not cut through
