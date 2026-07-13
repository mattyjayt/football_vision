"""Phase 3 demo (project MVP): optimal ball-carrier paths via A* over the cost map.

Run standalone:

    uv run python scripts/03_demo_grid_search.py

Figure 1: the smoothed A* path from the carrier to goal on each scenario, drawn
over the combined cost map.
Figure 2: the payoff — the SAME frame planned with "cautious" vs. "direct/risky"
cost weights, giving a safe wide route vs. a route straight through traffic.

Also prints Dijkstra-vs-A* optimality/efficiency and smoothing stats.
"""

from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footlab import pitch, simulate, value_surface, viz
from footlab.planners import grid_search
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame


def defensive_wall_frame() -> FrozenFrame:
    """Illustrative frame: a defensive wall between carrier and goal, gap on +y.

    Not one of the canonical scenarios — purpose-built so the safe-vs-risky
    weight trade-off produces visibly different routes (a sustained barrier is
    needed; a lone defender is never worth a big detour).
    """
    defenders = [[5, -16], [5, -8], [5, 0], [5, 8],
                 [10, -12], [10, -4], [10, 4]]
    positions = [[-25.0, 0.0]] + defenders
    team_ids = [TEAM_ATTACK] + [TEAM_DEFEND] * len(defenders)
    return FrozenFrame(positions, [[0.0, 0.0]] * len(positions), team_ids,
                       [-25.0, 0.0], ball_carrier=0)


def main() -> None:
    _, _, centers = pitch.make_grid(cell_size=1.0)
    goal = pitch.ATTACKING_GOAL_CENTER

    scenarios = {
        "Counter attack": simulate.scenario_counter_attack(),
        "Low block": simulate.scenario_low_block(),
        "Wing overload": simulate.scenario_wing_overload(),
    }

    # --- Figure 1: MVP path per scenario -------------------------------------
    print("A* carrier -> goal (balanced weights)")
    print("-" * 72)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 6.4 * len(scenarios)))
    for ax, (name, frame) in zip(axes, scenarios.items()):
        cost = value_surface.cost_map(frame, centers)
        start = frame.carrier_position

        dij = grid_search.plan_dijkstra(cost, centers, start, goal)
        ast = grid_search.plan_astar(cost, centers, start, goal)
        sm = grid_search.smooth_path(ast.points, cost, centers)
        straight = grid_search.path_cost(
            np.array([start, goal]), cost, centers)

        print(f"{name:16s} | A*=Dijkstra cost {np.isclose(ast.cost, dij.cost)} "
              f"({ast.cost:6.1f}) | expanded A* {ast.n_expanded:4d} vs "
              f"Dijkstra {dij.n_expanded:4d} | smoothed {sm.cost:6.1f} "
              f"({len(ast.points)}->{len(sm.points)} pts) | straight {straight:6.1f}")

        viz.plot_surface(frame, cost, ax=ax, title=f"{name} — A* path to goal",
                         cmap="magma_r", vmin=float(cost.min()),
                         vmax=float(cost.max()), cbar_label="cost (lower=better)")
        viz.plot_path(ax, ast.points, color="#26c6da", label="A* (raw)",
                      linewidth=1.6, marker_ends=False)
        viz.plot_path(ax, sm.points, color="#00e676", label="A* (smoothed)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_dir = FsPath(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / "03_grid_search_paths.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")

    # --- Figure 2: safe vs. risky routes on the same frame -------------------
    # A sustained defensive wall (a lone defender is never worth a detour).
    frame = defensive_wall_frame()
    start = frame.carrier_position
    tw = 2.0  # let terrain matter more than raw distance for this illustration
    routes = {
        "Cautious (2, 0.5, 3)": (value_surface.CostWeights(2.0, 0.5, 3.0),
                                 "#00b0ff"),
        "Direct / risky (1, 2, 0.4)": (value_surface.CostWeights(1.0, 2.0, 0.4),
                                       "#ff5252"),
    }
    display_cost = value_surface.cost_map(frame, centers)
    fig2, ax2 = plt.subplots(figsize=(12, 7.2))
    viz.plot_surface(frame, display_cost, ax=ax2,
                     title="Defensive wall: same frame, safe wide vs. direct route",
                     cmap="magma_r", vmin=float(display_cost.min()),
                     vmax=float(display_cost.max()),
                     cbar_label="cost (balanced, for reference)")
    print("\nSafe vs. risky route (defensive wall)")
    print("-" * 72)
    for label, (weights, color) in routes.items():
        cost = value_surface.cost_map(frame, centers, weights=weights)
        ast = grid_search.plan_astar(cost, centers, start, goal, terrain_weight=tw)
        sm = grid_search.smooth_path(ast.points, cost, centers, terrain_weight=tw)
        max_off = float(np.max(np.abs(sm.points[:, 1])))
        print(f"{label:30s} | max |y| off-center {max_off:5.1f} m "
              f"| {len(sm.points)} pts")
        viz.plot_path(ax2, sm.points, color=color, label=label)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig2.tight_layout()
    out2 = out_dir / "03_safe_vs_risky.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")

    print("-" * 72)
    print(f"Saved figures -> {out1}\n                {out2}")


if __name__ == "__main__":
    main()
