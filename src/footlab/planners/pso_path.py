"""Phase 5 — Particle Swarm Optimization over Bezier trajectory parameters.

A third planner. Where A* (Phase 3) searches a discrete grid and potential fields
(Phase 4) follow a local force, PSO globally searches a tiny *continuous* space of
smooth curve shapes.

The path is a cubic Bezier with the start (carrier) and end (goal) fixed; only the
two interior control points move — four numbers total. PSO tunes those four
numbers to minimize a fitness that reuses the Phase 2b cost map (so it is directly
comparable to A*), plus curvature and off-pitch penalties. The fitness is a black
box (non-differentiable), which is exactly PSO's niche.

References:
    Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization." Proc. IEEE
    International Conference on Neural Networks. Each particle moves under its own
    velocity plus attraction to its personal best and the swarm's global best:
        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x);   x <- x + v
    Shi, Y. & Eberhart, R. (1998). "A Modified Particle Swarm Optimizer." IEEE
    ICEC. Adds the inertia weight w; we linearly decay it from w_start to w_end
    (exploration -> exploitation).

Simplifications: no explicit time/max-speed model, so "feasibility" is a curvature
penalty (a proxy for turn radius) rather than a true kinodynamic constraint. The
cost integral reuses ``grid_search.path_cost`` for an apples-to-apples comparison
with A*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import pitch
from .grid_search import _sample_cost, path_cost


@dataclass(frozen=True)
class PSOParams:
    """PSO and fitness parameters."""

    n_particles: int = 30
    n_iters: int = 60
    w_start: float = 0.9        # inertia at start (exploration)
    w_end: float = 0.4          # inertia at end (exploitation)
    c1: float = 1.5             # cognitive (pull to personal best)
    c2: float = 1.5             # social (pull to global best)
    n_samples: int = 60         # points sampled along the Bezier for fitness
    w_curvature: float = 8.0    # penalty on total turning (radians)
    w_offpitch: float = 50.0    # penalty per meter a sample lies off the pitch
    terrain_weight: float = 1.0  # matches grid_search edge model
    init_spread: float = 15.0   # m; std of control-point initialization
    vel_frac: float = 0.2       # velocity clamp as a fraction of the bounds range
    seed: int = 0


@dataclass
class PSOResult:
    """Result of a PSO run.

    points:         (n_samples, 2) sampled world coordinates of the best curve.
    control_points: (2, 2) the two interior Bezier control points [P1, P2].
    fitness:        best fitness found.
    history:        (n_iters,) global-best fitness per iteration (non-increasing).
    n_evaluations:  number of fitness evaluations performed.
    """

    points: np.ndarray
    control_points: np.ndarray
    fitness: float
    history: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_evaluations: int = 0


def bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                 n: int = 60) -> np.ndarray:
    """Sample a cubic Bezier B(t) at ``n`` evenly spaced t in [0, 1]."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt ** 3 * p0 + 3 * mt ** 2 * t * p1
            + 3 * mt * t ** 2 * p2 + t ** 3 * p3)


def _turning_penalty(points: np.ndarray) -> float:
    """Total absolute turning angle (radians) along the polyline; 0 if straight."""
    seg = np.diff(points, axis=0)
    n = np.linalg.norm(seg, axis=1)
    good = n > 1e-9
    seg = seg[good]
    if len(seg) < 2:
        return 0.0
    u = seg / np.linalg.norm(seg, axis=1, keepdims=True)
    dots = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(dots)))


def _offpitch_penalty(points: np.ndarray) -> float:
    """Sum over samples of how far (m) each lies outside the pitch rectangle."""
    over_x = np.clip(np.abs(points[:, 0]) - pitch.HALF_LENGTH, 0.0, None)
    over_y = np.clip(np.abs(points[:, 1]) - pitch.HALF_WIDTH, 0.0, None)
    return float(np.sum(over_x + over_y))


def _polyline_cost(points: np.ndarray, cost: np.ndarray, centers: np.ndarray,
                   terrain_weight: float) -> float:
    """Vectorized line integral of the edge cost over the sampled polyline.

    Equivalent to :func:`grid_search.path_cost` when the polyline is already
    finely sampled (as a Bezier is), but done in one vectorized pass rather than
    a per-segment Python loop — the difference matters because PSO evaluates this
    thousands of times.
    """
    c = _sample_cost(cost, centers, points)                # (n,) terrain cost
    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    avg_c = 0.5 * (c[:-1] + c[1:])
    return float(np.sum(seg_len * (1.0 + terrain_weight * avg_c)))


def path_fitness(flat: np.ndarray, start: np.ndarray, goal: np.ndarray,
                 cost: np.ndarray, centers: np.ndarray, *,
                 params: PSOParams = PSOParams()) -> tuple[float, np.ndarray]:
    """Fitness of a candidate (P1, P2) encoded as a flat 4-vector.

    Returns ``(fitness, sampled_points)``. Fitness = cost-map line integral +
    curvature penalty + off-pitch penalty (lower is better).
    """
    p1 = flat[:2]
    p2 = flat[2:4]
    pts = bezier_curve(start, p1, p2, goal, params.n_samples)
    c = _polyline_cost(pts, cost, centers, params.terrain_weight)
    fit = (c + params.w_curvature * _turning_penalty(pts)
           + params.w_offpitch * _offpitch_penalty(pts))
    return fit, pts


def optimize_bezier(cost: np.ndarray, centers: np.ndarray, start: np.ndarray,
                    goal: np.ndarray, *, params: PSOParams = PSOParams()
                    ) -> PSOResult:
    """Optimize the two interior Bezier control points with PSO."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    rng = np.random.default_rng(params.seed)
    dim = 4  # (P1x, P1y, P2x, P2y)

    # Bounds: control points anywhere on the pitch.
    low = np.array([-pitch.HALF_LENGTH, -pitch.HALF_WIDTH] * 2)
    high = np.array([pitch.HALF_LENGTH, pitch.HALF_WIDTH] * 2)
    vmax = params.vel_frac * (high - low)

    # Initialize particles near the straight line start->goal, then jitter.
    base = np.concatenate([start + (goal - start) / 3.0,
                           start + 2.0 * (goal - start) / 3.0])
    x = base[None, :] + rng.normal(0.0, params.init_spread,
                                   size=(params.n_particles, dim))
    x = np.clip(x, low, high)
    v = rng.uniform(-vmax, vmax, size=(params.n_particles, dim))

    fit = np.array([path_fitness(xi, start, goal, cost, centers, params=params)[0]
                    for xi in x])
    n_eval = params.n_particles
    pbest = x.copy()
    pbest_fit = fit.copy()
    g = int(np.argmin(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = float(pbest_fit[g])

    history = np.empty(params.n_iters)
    for it in range(params.n_iters):
        w = params.w_start + (params.w_end - params.w_start) * (
            it / max(1, params.n_iters - 1))
        r1 = rng.random((params.n_particles, dim))
        r2 = rng.random((params.n_particles, dim))
        v = (w * v
             + params.c1 * r1 * (pbest - x)
             + params.c2 * r2 * (gbest[None, :] - x))
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, low, high)

        fit = np.array([path_fitness(xi, start, goal, cost, centers,
                                     params=params)[0] for xi in x])
        n_eval += params.n_particles
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gbest_fit:
            gbest = pbest[g].copy()
            gbest_fit = float(pbest_fit[g])
        history[it] = gbest_fit

    _, pts = path_fitness(gbest, start, goal, cost, centers, params=params)
    return PSOResult(points=pts, control_points=gbest.reshape(2, 2),
                     fitness=gbest_fit, history=history, n_evaluations=n_eval)


if __name__ == "__main__":
    from .. import simulate, value_surface
    from .grid_search import plan_astar

    frame = simulate.scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    cost = value_surface.cost_map(frame, centers)
    start, goal = frame.carrier_position, pitch.ATTACKING_GOAL_CENTER

    res = optimize_bezier(cost, centers, start, goal)
    astar = plan_astar(cost, centers, start, goal)
    astar_fit = path_cost(astar.points, cost, centers)
    print(f"PSO  fitness {res.fitness:.1f} ({res.n_evaluations} evals)")
    print(f"A*   cost    {astar_fit:.1f}")
    print(f"history monotone non-increasing: "
          f"{bool(np.all(np.diff(res.history) <= 1e-9))}")
