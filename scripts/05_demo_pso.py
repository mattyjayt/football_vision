"""Phase 5 demo: PSO over Bezier trajectories, benchmarked against A*.

Run standalone:

    uv run python scripts/05_demo_pso.py

Figure 1: A* (grid) vs. PSO (smooth Bezier) paths on each scenario, over the cost
map. Prints a table comparing pure cost-map cost, smoothness (total turning), and
runtime.
Figure 2: PSO convergence for different swarm sizes (the sensitivity study).
"""

from __future__ import annotations

import time
from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footlab import pitch, simulate, value_surface, viz
from footlab.planners import grid_search
from footlab.planners.pso_path import (
    PSOParams,
    _turning_penalty,
    optimize_bezier,
)
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame

GOAL = pitch.ATTACKING_GOAL_CENTER


def defensive_wall_frame() -> FrozenFrame:
    defenders = [[5, -16], [5, -8], [5, 0], [5, 8], [10, -12], [10, -4], [10, 4]]
    positions = [[-25.0, 0.0]] + defenders
    team_ids = [TEAM_ATTACK] + [TEAM_DEFEND] * len(defenders)
    return FrozenFrame(positions, [[0.0, 0.0]] * len(positions), team_ids,
                       [-25.0, 0.0], ball_carrier=0)


def main() -> None:
    _, _, centers = pitch.make_grid(cell_size=1.0)
    scenarios = {
        "Counter attack": simulate.scenario_counter_attack(),
        "Low block": simulate.scenario_low_block(),
        "Wing overload": simulate.scenario_wing_overload(),
        "Defensive wall": defensive_wall_frame(),
    }

    print(f"{'scenario':16s} | {'A* cost':>8s} {'A* turn':>8s} {'A* t(s)':>8s} "
          f"| {'PSO cost':>8s} {'PSO turn':>9s} {'PSO t(s)':>8s}")
    print("-" * 82)

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    axes = axes.ravel()
    for ax, (name, frame) in zip(axes, scenarios.items()):
        cost = value_surface.cost_map(frame, centers)
        start = frame.carrier_position

        t0 = time.perf_counter()
        astar = grid_search.plan_astar(cost, centers, start, GOAL)
        astar = grid_search.smooth_path(astar.points, cost, centers)
        t_astar = time.perf_counter() - t0

        t0 = time.perf_counter()
        pso = optimize_bezier(cost, centers, start, GOAL)
        t_pso = time.perf_counter() - t0

        # Apples-to-apples: pure cost-map integral of each path, and smoothness.
        astar_cost = grid_search.path_cost(astar.points, cost, centers)
        pso_cost = grid_search.path_cost(pso.points, cost, centers)
        astar_turn = _turning_penalty(astar.points)
        pso_turn = _turning_penalty(pso.points)
        print(f"{name:16s} | {astar_cost:8.1f} {astar_turn:8.2f} {t_astar:8.3f} "
              f"| {pso_cost:8.1f} {pso_turn:9.2f} {t_pso:8.3f}")

        viz.plot_surface(frame, cost, ax=ax, title=f"{name}: A* vs. PSO",
                         cmap="magma_r", vmin=float(cost.min()),
                         vmax=float(cost.max()), cbar_label="cost (lower=better)")
        viz.plot_path(ax, astar.points, color="#26c6da", label="A* (grid)")
        viz.plot_path(ax, pso.points, color="#00e676", label="PSO (Bezier)",
                      marker_ends=False)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_dir = FsPath(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / "05_pso_vs_astar.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")

    # --- Figure 2: convergence vs. swarm size --------------------------------
    frame = scenarios["Defensive wall"]
    cost = value_surface.cost_map(frame, centers)
    start = frame.carrier_position
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    print("\nConvergence vs. swarm size (defensive wall)")
    print("-" * 50)
    for n in (10, 30, 60):
        res = optimize_bezier(cost, centers, start, GOAL,
                              params=PSOParams(n_particles=n, n_iters=80))
        ax2.plot(res.history, label=f"{n} particles ({res.n_evaluations} evals)")
        print(f"{n:3d} particles | final fitness {res.fitness:7.1f} "
              f"| {res.n_evaluations} evals")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("global-best fitness")
    ax2.set_title("PSO convergence vs. swarm size")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = out_dir / "05_pso_convergence.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")

    print("-" * 50)
    print(f"Saved figures -> {out1}\n                {out2}")


if __name__ == "__main__":
    main()
