"""Phase 4 demo: artificial potential fields, including the local-minimum trap.

Run standalone:

    uv run python scripts/04_demo_potential_fields.py

Three panels:
  A. Working case  — the force field (quiver) guides the carrier around offset
     defenders to the goal.
  B. Local-minimum trap — a concave "cup" of defenders stalls gradient descent
     in open space; the goal is never reached (the classic APF failure).
  C. Same trapping frame solved by A* (Phase 3) — global search has no local
     minima, so it finds a path. The contrast is the lesson.
"""

from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footlab import pitch, value_surface, viz
from footlab.planners import grid_search
from footlab.planners import potential_fields as pf
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame

GOAL = np.array([45.0, 0.0])


def _frame(defenders: list[list[float]]) -> FrozenFrame:
    positions = [[-35.0, 0.0]] + [list(map(float, d)) for d in defenders]
    team_ids = [TEAM_ATTACK] + [TEAM_DEFEND] * len(defenders)
    return FrozenFrame(positions, [[0.0, 0.0]] * len(positions), team_ids,
                       [-35.0, 0.0], ball_carrier=0)


def _draw_field(ax, frame: FrozenFrame) -> None:
    """Overlay a normalized quiver of the total force field."""
    obstacles = pf.projected_defenders(frame, pf.DEFAULT_PARAMS.project_time)
    gx, gy = np.meshgrid(np.arange(-40, 50, 3.0), np.arange(-24, 25, 3.0))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    F = pf.force_field(pts, GOAL, obstacles)
    mag = np.linalg.norm(F, axis=-1)
    unit = F / np.clip(mag, 1e-9, None)[:, None]
    ax.quiver(pts[:, 0], pts[:, 1], unit[:, 0], unit[:, 1],
              np.clip(mag, 0, 20), cmap="viridis", zorder=1,
              scale=40, width=0.003, alpha=0.9)


def main() -> None:
    working = _frame([[0, 3], [20, -4]])
    trap = _frame([[2, 0], [8, 3], [8, -3]])

    fig, axes = plt.subplots(1, 3, figsize=(30, 7.2))

    # --- Panel A: working case ----------------------------------------------
    ax = axes[0]
    pitch.draw_pitch(ax)
    _draw_field(ax, working)
    res = pf.descend(working, GOAL)
    viz.plot_frame(working, ax=ax, draw_field=False,
                   title="A. Field guides carrier around defenders to goal")
    viz.plot_path(ax, res.points, color="#00e676", label="APF descent")
    ax.scatter(*GOAL, s=260, marker="*", color="#00e676", edgecolors="black",
               zorder=9, label="goal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    print(f"A. working: reached={res.reached_goal} ({res.reason})")

    # --- Panel B: local-minimum trap ----------------------------------------
    ax = axes[1]
    pitch.draw_pitch(ax)
    _draw_field(ax, trap)
    res_trap = pf.descend(trap, GOAL)
    viz.plot_frame(trap, ax=ax, draw_field=False,
                   title="B. Local-minimum trap: descent stalls in open space")
    viz.plot_path(ax, res_trap.points, color="#ff5252", label="APF descent")
    stuck = res_trap.points[-1]
    ax.scatter(*stuck, s=320, marker="X", color="#ff1744", edgecolors="black",
               linewidths=1.5, zorder=10)
    ax.annotate("TRAPPED\n(goal never reached)", xy=stuck,
                xytext=(stuck[0] - 24, stuck[1] + 14), fontsize=10, color="#b71c1c",
                weight="bold",
                arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.5))
    ax.scatter(*GOAL, s=260, marker="*", color="#333333", edgecolors="black",
               zorder=9, label="goal (unreached)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    print(f"B. trap:    reached={res_trap.reached_goal} ({res_trap.reason}) "
          f"stopped at x={stuck[0]:.1f}")

    # --- Panel C: A* solves the same trapping frame -------------------------
    ax = axes[2]
    _, _, centers = pitch.make_grid(cell_size=1.0)
    cost = value_surface.cost_map(trap, centers)
    astar = grid_search.plan_astar(cost, centers, trap.carrier_position, GOAL,
                                   terrain_weight=2.0)
    smooth = grid_search.smooth_path(astar.points, cost, centers,
                                     terrain_weight=2.0)
    viz.plot_surface(trap, cost, ax=ax,
                     title="C. A* (global search) solves the same frame",
                     cmap="magma_r", vmin=float(cost.min()),
                     vmax=float(cost.max()), cbar_label="cost (lower=better)")
    viz.plot_path(ax, smooth.points, color="#00e676", label="A* path")
    ax.scatter(*GOAL, s=260, marker="*", color="#00e676", edgecolors="black",
               zorder=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    print(f"C. A* same frame: reached goal (cost {astar.cost:.1f})")

    fig.tight_layout()
    out_dir = FsPath(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "04_potential_fields.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
