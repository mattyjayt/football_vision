"""Phase 3 — graph search over the cost map: Dijkstra and A*.

The Phase 2b cost grid is treated as an 8-connected weighted graph. Moving
between adjacent cells costs the geometric step length times a per-meter cost
that is 1 (a base distance term) plus the weighted local terrain cost:

    edge(a -> b) = ||b - a|| * (1 + terrain_weight * 0.5 * (cost[a] + cost[b]))

so a path's total cost is a line integral of danger-weighted distance. The base
1 keeps paths from wandering, and — because the minimum per-meter cost is 1 —
straight-line distance to the goal is an admissible A* heuristic.

Reference:
    Hart, P. E., Nilsson, N. J. & Raphael, B. (1968). "A Formal Basis for the
    Heuristic Determination of Minimum Cost Paths." IEEE Trans. Systems Science
    and Cybernetics, 4(2). A* expands nodes in order of f = g + h, where g is
    cost-so-far and h is an admissible (never-overestimating) estimate of the
    cost-to-go; the returned path is optimal. Dijkstra's algorithm is the special
    case h = 0.

Simplifications: 8-connected grid (paths move in 45-degree increments before
smoothing); a single global cost map (defenders are static within a plan).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

# (di, dj, step_length_in_cells) for the 4 orthogonal and 4 diagonal neighbors.
_ORTHO = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
_DIAG = [(-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
         (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]


@dataclass
class Path:
    """A planned path.

    points:     (K, 2) world coordinates (meters) from start to goal.
    cost:       total path cost under the edge-cost model.
    n_expanded: number of nodes expanded (popped) by the search — the A* vs.
                Dijkstra efficiency measure.
    cells:      (K, 2) grid indices (i, j), or None for a smoothed path.
    """

    points: np.ndarray
    cost: float
    n_expanded: int = 0
    cells: np.ndarray | None = None


def _axes(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract the x/y cell-center axes and cell size from a centers grid."""
    xs = centers[0, :, 0]
    ys = centers[:, 0, 1]
    cell_size = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    return xs, ys, cell_size


def world_to_cell(centers: np.ndarray, xy: np.ndarray) -> tuple[int, int]:
    """Nearest grid cell (i, j) to a world point (x, y)."""
    xs, ys, _ = _axes(centers)
    j = int(np.argmin(np.abs(xs - xy[0])))
    i = int(np.argmin(np.abs(ys - xy[1])))
    return i, j


def _search(cost: np.ndarray, centers: np.ndarray, start_xy: np.ndarray,
            goal_xy: np.ndarray, *, use_heuristic: bool, terrain_weight: float,
            diagonal: bool) -> Path:
    """Uniform-cost / A* search. ``use_heuristic=False`` gives Dijkstra."""
    ny, nx = cost.shape
    xs, ys, cell_size = _axes(centers)
    start = world_to_cell(centers, start_xy)
    goal = world_to_cell(centers, goal_xy)
    goal_pt = centers[goal]
    neighbors = _ORTHO + _DIAG if diagonal else _ORTHO

    def heuristic(i: int, j: int) -> float:
        if not use_heuristic:
            return 0.0
        # Admissible: minimum per-meter cost is 1 (terrain >= 0).
        return float(np.hypot(xs[j] - goal_pt[0], ys[i] - goal_pt[1]))

    g = np.full((ny, nx), np.inf)
    g[start] = 0.0
    visited = np.zeros((ny, nx), dtype=bool)
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, float, tuple[int, int]]] = [
        (heuristic(*start), 0.0, start)]
    n_expanded = 0

    while pq:
        _, gcur, (i, j) = heapq.heappop(pq)
        if visited[i, j]:
            continue
        visited[i, j] = True
        n_expanded += 1
        if (i, j) == goal:
            break
        for di, dj, step in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or visited[ni, nj]:
                continue
            step_m = cell_size * step
            edge = step_m * (1.0 + terrain_weight
                             * 0.5 * (cost[i, j] + cost[ni, nj]))
            ng = gcur + edge
            if ng < g[ni, nj]:
                g[ni, nj] = ng
                prev[(ni, nj)] = (i, j)
                heapq.heappush(pq, (ng + heuristic(ni, nj), ng, (ni, nj)))

    # Reconstruct.
    cells = [goal]
    cur = goal
    while cur != start:
        if cur not in prev:  # unreachable (should not happen on a full grid)
            break
        cur = prev[cur]
        cells.append(cur)
    cells.reverse()
    cells_arr = np.array(cells)
    points = centers[cells_arr[:, 0], cells_arr[:, 1]]
    return Path(points=points, cost=float(g[goal]), n_expanded=n_expanded,
                cells=cells_arr)


def plan_dijkstra(cost: np.ndarray, centers: np.ndarray, start_xy: np.ndarray,
                  goal_xy: np.ndarray, *, terrain_weight: float = 1.0,
                  diagonal: bool = True) -> Path:
    """Cheapest path from ``start_xy`` to ``goal_xy`` via Dijkstra (h = 0)."""
    return _search(cost, centers, np.asarray(start_xy, float),
                   np.asarray(goal_xy, float), use_heuristic=False,
                   terrain_weight=terrain_weight, diagonal=diagonal)


def plan_astar(cost: np.ndarray, centers: np.ndarray, start_xy: np.ndarray,
               goal_xy: np.ndarray, *, terrain_weight: float = 1.0,
               diagonal: bool = True) -> Path:
    """Cheapest path via A* with the admissible straight-line distance heuristic.

    Returns the same optimal path as :func:`plan_dijkstra`, usually after
    expanding fewer nodes.
    """
    return _search(cost, centers, np.asarray(start_xy, float),
                   np.asarray(goal_xy, float), use_heuristic=True,
                   terrain_weight=terrain_weight, diagonal=diagonal)


# --- cost integration and smoothing ------------------------------------------
def _sample_cost(cost: np.ndarray, centers: np.ndarray, pts: np.ndarray
                 ) -> np.ndarray:
    """Bilinearly sample the terrain cost grid at world points ``pts`` (M, 2)."""
    xs, ys, _ = _axes(centers)
    ny, nx = cost.shape
    csx = xs[1] - xs[0] if nx > 1 else 1.0
    csy = ys[1] - ys[0] if ny > 1 else 1.0
    fx = np.clip((pts[:, 0] - xs[0]) / csx, 0, nx - 1)
    fy = np.clip((pts[:, 1] - ys[0]) / csy, 0, ny - 1)
    j0 = np.floor(fx).astype(int)
    i0 = np.floor(fy).astype(int)
    j1 = np.minimum(j0 + 1, nx - 1)
    i1 = np.minimum(i0 + 1, ny - 1)
    tx = fx - j0
    ty = fy - i0
    return (cost[i0, j0] * (1 - tx) * (1 - ty)
            + cost[i0, j1] * tx * (1 - ty)
            + cost[i1, j0] * (1 - tx) * ty
            + cost[i1, j1] * tx * ty)


def _segment_cost(p0: np.ndarray, p1: np.ndarray, cost: np.ndarray,
                  centers: np.ndarray, terrain_weight: float, step: float
                  ) -> float:
    """Integrated edge cost along the straight segment p0 -> p1."""
    length = float(np.hypot(*(p1 - p0)))
    if length == 0.0:
        return 0.0
    n = max(2, int(np.ceil(length / step)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    c = _sample_cost(cost, centers, pts)
    seg_len = length / (n - 1)
    avg_c = 0.5 * (c[:-1] + c[1:])
    return float(np.sum(seg_len * (1.0 + terrain_weight * avg_c)))


def path_cost(points: np.ndarray, cost: np.ndarray, centers: np.ndarray, *,
              terrain_weight: float = 1.0, step: float | None = None) -> float:
    """Total edge cost of an arbitrary polyline (line integral over the field)."""
    _, _, cell_size = _axes(centers)
    step = step if step is not None else cell_size / 2.0
    return sum(_segment_cost(points[k], points[k + 1], cost, centers,
                             terrain_weight, step)
               for k in range(len(points) - 1))


def smooth_path(points: np.ndarray, cost: np.ndarray, centers: np.ndarray, *,
                terrain_weight: float = 1.0, tol: float = 0.02,
                window: int = 8, step: float | None = None) -> Path:
    """Shortcut-smooth a staircased grid path, respecting the cost field.

    Greedily replaces a *local* subpath with a straight segment whenever the
    segment's integrated cost does not exceed the subpath's by more than ``tol``
    (relative). This removes the 45-degree staircase artifacts of an 8-connected
    grid without cutting through expensive terrain.

    ``window`` caps how many waypoints a single shortcut may span. This is
    important: an unbounded shortcut could straighten a genuine danger-avoiding
    detour back through the danger (because the continuous line integral used
    here differs slightly from the planner's discrete edge cost). A local window
    smooths jaggedness while preserving the route's overall shape. Returns a new
    :class:`Path` (``cells`` is None).
    """
    _, _, cell_size = _axes(centers)
    step = step if step is not None else cell_size / 2.0
    pts = np.asarray(points, float)
    if len(pts) <= 2:
        return Path(points=pts.copy(),
                    cost=path_cost(pts, cost, centers,
                                   terrain_weight=terrain_weight, step=step))

    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j_max = min(i + window, len(pts) - 1)
        j = j_max
        while j > i + 1:
            direct = _segment_cost(pts[i], pts[j], cost, centers,
                                   terrain_weight, step)
            original = sum(_segment_cost(pts[k], pts[k + 1], cost, centers,
                                         terrain_weight, step)
                           for k in range(i, j))
            if direct <= original * (1.0 + tol):
                break
            j -= 1
        out.append(pts[j])
        i = j
    out_arr = np.array(out)
    return Path(points=out_arr,
                cost=path_cost(out_arr, cost, centers,
                               terrain_weight=terrain_weight, step=step))


if __name__ == "__main__":
    from .. import pitch, simulate, value_surface

    frame = simulate.scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    cost = value_surface.cost_map(frame, centers)
    start = frame.carrier_position
    goal = pitch.ATTACKING_GOAL_CENTER

    dij = plan_dijkstra(cost, centers, start, goal)
    ast = plan_astar(cost, centers, start, goal)
    sm = smooth_path(ast.points, cost, centers)
    print(f"Dijkstra: cost {dij.cost:.2f}, expanded {dij.n_expanded}")
    print(f"A*      : cost {ast.cost:.2f}, expanded {ast.n_expanded}")
    print(f"same cost: {np.isclose(dij.cost, ast.cost)}")
    print(f"smoothed: cost {sm.cost:.2f}, {len(ast.points)} -> {len(sm.points)} points")
