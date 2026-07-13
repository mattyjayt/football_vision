"""Phase 6 demo: reactive defenders — a static plan failing vs. replanning.

Run standalone:

    uv run python scripts/06_demo_reactive.py

Same opening frame, two carriers:
  * one commits to the initial A* plan (static) — the defenders react, cut it off,
    and intercept;
  * one re-runs A* every few steps (replanning) — it re-routes around the
    shifting defenders and reaches goal.

Saves an animated GIF (the headline deliverable) and a static summary PNG showing
the trajectories taken and where the tackle happened.
"""

from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from footlab import pitch, value_surface, viz
from footlab.planners import reactive as rc
from footlab.planners.grid_search import plan_astar, smooth_path
from footlab.state import TEAM_ATTACK, TEAM_DEFEND, FrozenFrame

GOAL = np.array([45.0, 0.0])
WEIGHTS = value_surface.CostWeights(w_control=1.0, w_value=0.5, w_defender=4.0)
PARAMS = rc.ReactiveParams(carrier_speed=6.5, def_max_speed=6.5,
                           def_max_accel=4.0, terrain_weight=3.0)
ATTACK_C, DEFEND_C = "#d62728", "#1f77b4"


def scenario() -> FrozenFrame:
    """Carrier behind a two-defender gate guarding the route to goal."""
    positions = [[-25.0, 0.0], [5.0, 3.0], [5.0, -3.0]]
    team_ids = [TEAM_ATTACK, TEAM_DEFEND, TEAM_DEFEND]
    return FrozenFrame(positions, [[0.0, 0.0]] * 3, team_ids,
                       [-25.0, 0.0], ball_carrier=0)


def _draw_state(ax, res: rc.RolloutResult, idx: int) -> None:
    """Draw the world (players, ball, current plan) at history index ``idx``."""
    pitch.draw_pitch(ax)
    pos = res.pos_history[idx]
    att = res.team_ids == TEAM_ATTACK
    dfn = res.team_ids == TEAM_DEFEND
    # Faint trails up to now.
    trail = res.pos_history[:idx + 1]
    ax.plot(trail[:, res.carrier_idx, 0], trail[:, res.carrier_idx, 1],
            color=ATTACK_C, alpha=0.35, lw=1.5, zorder=2)
    for k in np.where(dfn)[0]:
        ax.plot(trail[:, k, 0], trail[:, k, 1], color=DEFEND_C, alpha=0.3,
                lw=1.2, zorder=2)
    # Current plan.
    plan = res.plan_history[idx]
    ax.plot(plan[:, 0], plan[:, 1], "--", color="#00e676", lw=2.0, zorder=3,
            label="current plan")
    # Players + ball.
    ax.scatter(pos[att, 0], pos[att, 1], s=150, c=ATTACK_C, edgecolors="white",
               linewidths=1.2, zorder=5)
    ax.scatter(pos[dfn, 0], pos[dfn, 1], s=150, c=DEFEND_C, edgecolors="white",
               linewidths=1.2, zorder=5)
    cp = pos[res.carrier_idx]
    ax.scatter(cp[0], cp[1], s=320, facecolors="none", edgecolors="#ffcc00",
               linewidths=2.2, zorder=6)
    ax.scatter(*GOAL, s=240, marker="*", color="#00e676", edgecolors="black",
               zorder=6)
    # Interception marker once it has happened.
    if res.intercepted and idx == len(res.times) - 1:
        ax.scatter(*res.intercept_point, s=380, marker="X", color="#ff1744",
                   edgecolors="black", linewidths=1.5, zorder=8)


def _status(res: rc.RolloutResult, idx: int, label: str) -> str:
    if res.intercepted and idx == len(res.times) - 1:
        return f"{label}: INTERCEPTED at {res.progress:.0%} progress"
    if res.reached_goal and idx == len(res.times) - 1:
        return f"{label}: reached goal ✓"
    return f"{label}  (t = {res.times[idx]:.1f}s)"


def main() -> None:
    frame = scenario()
    _, _, centers = pitch.make_grid(cell_size=1.0)

    # The plan both carriers start with.
    cost0 = value_surface.cost_map(frame, centers, weights=WEIGHTS)
    astar0 = plan_astar(cost0, centers, frame.carrier_position, GOAL,
                        terrain_weight=PARAMS.terrain_weight)
    plan0 = smooth_path(astar0.points, cost0, centers,
                        terrain_weight=PARAMS.terrain_weight).points

    static = rc.rollout_static(frame, plan0, GOAL, params=PARAMS)
    replan = rc.rollout_replanning(frame, centers, GOAL, replan_every=4,
                                   weights=WEIGHTS, params=PARAMS)

    print("Reactive rollout (same opening frame)")
    print("-" * 56)
    print(f"Static plan : reached={static.reached_goal} "
          f"intercepted={static.intercepted} progress={static.progress:.0%} "
          f"time={static.sim_time:.1f}s")
    print(f"Replanning  : reached={replan.reached_goal} "
          f"intercepted={replan.intercepted} progress={replan.progress:.0%} "
          f"time={replan.sim_time:.1f}s")

    out_dir = FsPath(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)

    # --- Summary PNG (final trajectories) ------------------------------------
    figs, axs = plt.subplots(1, 2, figsize=(21, 7))
    _draw_state(axs[0], static, len(static.times) - 1)
    axs[0].set_title(_status(static, len(static.times) - 1, "Static plan"),
                     fontsize=12)
    _draw_state(axs[1], replan, len(replan.times) - 1)
    axs[1].set_title(_status(replan, len(replan.times) - 1, "Replanning"),
                     fontsize=12)
    figs.tight_layout()
    png = out_dir / "06_reactive_summary.png"
    figs.savefig(png, dpi=120, bbox_inches="tight")

    # --- Animated GIF --------------------------------------------------------
    n = max(len(static.times), len(replan.times)) + 6  # +hold at end
    fig, axes = plt.subplots(1, 2, figsize=(21, 7))

    def update(t: int):
        for ax, res, label in ((axes[0], static, "Static plan"),
                               (axes[1], replan, "Replanning")):
            ax.clear()
            idx = min(t, len(res.times) - 1)
            _draw_state(ax, res, idx)
            ax.set_title(_status(res, idx, label), fontsize=12)
        return []

    anim = FuncAnimation(fig, update, frames=n, interval=140, blit=False)
    gif = out_dir / "06_reactive.gif"
    anim.save(gif, writer=PillowWriter(fps=7))
    plt.close("all")

    print("-" * 56)
    print(f"Saved -> {gif}\n         {png}")


if __name__ == "__main__":
    main()
