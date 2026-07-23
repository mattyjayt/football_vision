"""Complete end-to-end demo: every footlab stage run on ONE real match frame.

Run standalone (requires the SkillCorner open data in ``data/skillcorner``):

    uv run python scripts/complete_demo.py [frame_index]

Where the per-phase demos (01..06) each show ONE component on a SIMULATED
frame, this script runs the WHOLE stack on a single REAL frame (broadcast
tracking from a 2024/25 A-League match) and writes one figure per stage to
``figures/complete/``:

    01_voronoi_dominant.png  space ownership: Voronoi vs. dominant regions
    02_pitch_control.png     Spearman pitch-control surface
    03_cost_map.png          value + defender cost, fused
    04_astar.png             A* optimal carrier path over the cost map
    05_potential_fields.png  APF field + gradient-descent path (may trap!)
    06_pso.png               PSO-optimised Bezier path vs. A*
    07_reactive_summary.png  static plan vs. replanning (final state)
    07_reactive.gif          the same two rollouts animated

Toggle stages without touching code via the ``STAGES`` dict below (the GIF is
off by default — it is the slow one). Every stage consumes the same
``FrozenFrame`` produced by ``footlab.skillcorner``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from footlab import geometry, pitch, skillcorner, value_surface, viz
from footlab.pitch_control import pitch_control_surface
from footlab.planners import grid_search
from footlab.planners import potential_fields as pf
from footlab.planners import pso_path
from footlab.planners import reactive as rc
from footlab.state import TEAM_ATTACK, TEAM_DEFEND

DATA_DIR = (FsPath(__file__).resolve().parent.parent
            / "data" / "skillcorner" / "data" / "matches" / "1886347")
OUT_DIR = FsPath(__file__).resolve().parent.parent / "figures" / "complete"

GOAL = pitch.ATTACKING_GOAL_CENTER
WEIGHTS = value_surface.CostWeights(w_control=1.0, w_value=0.5, w_defender=4.0)
REACTIVE = rc.ReactiveParams(carrier_speed=6.5, def_max_speed=6.5,
                             def_max_accel=4.0, terrain_weight=3.0)
ATTACK_C, DEFEND_C = "#d62728", "#1f77b4"

# --- Stage toggles -----------------------------------------------------------
# Set a stage to False to skip it (faster iteration); no code edits needed.
STAGES = {
    "voronoi": True,
    "pitch_control": True,
    "cost_map": True,
    "astar": True,
    "potential_fields": True,
    "pso": True,
    "reactive": True,
    "reactive_gif": False,   # slow (~tens of seconds); enable when wanted
}


# --- Shared context ----------------------------------------------------------
class Ctx:
    """Precomputed frame + surfaces shared by every stage."""

    def __init__(self, frame_index: int):
        if not DATA_DIR.exists():
            raise SystemExit(
                "SkillCorner data not found. Get it with:\n"
                "  git clone https://github.com/SkillCorner/opendata.git "
                "data/skillcorner  (needs git-lfs)")
        print(f"Loading real frame {frame_index} ...")
        self.frame = skillcorner.frames_to_frozen(DATA_DIR, frame_index)
        self.frame_index = frame_index
        self.start = (self.frame.carrier_position
                      if self.frame.carrier_position is not None
                      else self.frame.ball_pos)
        _, _, self.centers = pitch.make_grid(cell_size=1.0)
        # Populated lazily by stages (declared here so Pyright knows them).
        self.control: np.ndarray | None = None
        self.cost: np.ndarray | None = None
        self.astar_path: np.ndarray | None = None
        print(f"players {self.frame.n_players} "
              f"({self.frame.attack_mask.sum()} att / "
              f"{self.frame.defend_mask.sum()} def) | "
              f"start {np.round(self.start, 1)} | "
              f"ball {np.round(self.frame.ball_pos, 1)}")

    def save(self, fig, name: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / name
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out}")


# --- Stages ------------------------------------------------------------------
def stage_voronoi(ctx: Ctx) -> None:
    print("[1/7] Voronoi vs. dominant regions")
    team_voronoi = geometry.control_to_team(
        ctx.frame, geometry.voronoi_control(ctx.frame, ctx.centers))
    team_dominant = geometry.control_to_team(
        ctx.frame, geometry.dominant_region(ctx.frame, ctx.centers))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    viz.plot_team_regions(ctx.frame, team_voronoi, ax=axes[0],
                          title="Voronoi (nearest player)")
    viz.plot_team_regions(ctx.frame, team_dominant, ax=axes[1],
                          title="Dominant regions (arrival time)")
    fig.tight_layout()
    ctx.save(fig, "01_voronoi_dominant.png")


def stage_pitch_control(ctx: Ctx) -> None:
    print("[2/7] Pitch control surface")
    ctx.control = pitch_control_surface(ctx.frame, ctx.centers)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    viz.plot_surface(ctx.frame, ctx.control, ax=ax,
                     title="Pitch control (attacking team)",
                     cmap="RdYlGn", vmin=0.0, vmax=1.0,
                     cbar_label="P(attack controls ball)")
    fig.tight_layout()
    ctx.save(fig, "02_pitch_control.png")


def stage_cost_map(ctx: Ctx) -> None:
    print("[3/7] Combined cost map")
    ctx.cost = value_surface.cost_map(ctx.frame, ctx.centers,
                                      control=ctx.control, weights=WEIGHTS)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    viz.plot_surface(ctx.frame, ctx.cost, ax=ax, title="Cost map (fused)",
                     cmap="magma_r", vmin=float(ctx.cost.min()),
                     vmax=float(ctx.cost.max()), cbar_label="cost (lower=better)")
    fig.tight_layout()
    ctx.save(fig, "03_cost_map.png")


def stage_astar(ctx: Ctx) -> None:
    print("[4/7] A* grid search")
    cost = ctx.cost if ctx.cost is not None else value_surface.cost_map(
        ctx.frame, ctx.centers, weights=WEIGHTS)
    ast = grid_search.plan_astar(cost, ctx.centers, ctx.start, GOAL,
                                 terrain_weight=REACTIVE.terrain_weight)
    ctx.astar_path = grid_search.smooth_path(ast.points, cost, ctx.centers,
                                             terrain_weight=REACTIVE.terrain_weight).points
    print(f"  A*: {len(ast.points)} pts, cost {ast.cost:.1f}, "
          f"expanded {ast.n_expanded} nodes")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    viz.plot_surface(ctx.frame, cost, ax=ax, title="A* optimal path to goal",
                     cmap="magma_r", vmin=float(cost.min()),
                     vmax=float(cost.max()), cbar_label="cost")
    viz.plot_path(ax, ctx.astar_path, color="#00e676", label="A* (smoothed)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    ctx.save(fig, "04_astar.png")


def stage_potential_fields(ctx: Ctx) -> None:
    print("[5/7] Artificial potential fields")
    res = pf.descend(ctx.frame, GOAL)
    pts = np.asarray(res.points)
    print(f"  APF: {len(pts)} steps, reason={res.reason}")
    # Force field on a coarse grid + the descended path.
    _, _, gcenters = pitch.make_grid(cell_size=3.0)
    obstacles = pf.projected_defenders(ctx.frame, pf.DEFAULT_PARAMS.project_time)
    F = pf.force_field(gcenters, GOAL, obstacles)   # (..., 2) force vectors
    fig, ax = plt.subplots(figsize=(10, 6.2))
    pitch.draw_pitch(ax)
    ax.quiver(gcenters[..., 0], gcenters[..., 1], F[..., 0], F[..., 1],
              color="#90a4ae", scale=None, width=0.002)
    viz.plot_frame(ctx.frame, ax=ax,
                   title=f"Potential field + descent ({res.reason})")
    ax.plot(pts[:, 0], pts[:, 1], color="#00e676", lw=2.2, label="APF path")
    ax.scatter(*GOAL, s=240, marker="*", color="#00e676", edgecolors="black",
               zorder=6)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    ctx.save(fig, "05_potential_fields.png")


def stage_pso(ctx: Ctx) -> None:
    print("[6/7] PSO Bezier trajectory vs. A*")
    cost = ctx.cost if ctx.cost is not None else value_surface.cost_map(
        ctx.frame, ctx.centers, weights=WEIGHTS)
    res = pso_path.optimize_bezier(cost, ctx.centers, ctx.start, GOAL)
    print(f"  PSO: fitness {res.fitness:.1f} after {res.n_evaluations} evals")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    viz.plot_surface(ctx.frame, cost, ax=ax, title="PSO Bezier vs. A*",
                     cmap="magma_r", vmin=float(cost.min()),
                     vmax=float(cost.max()), cbar_label="cost")
    viz.plot_path(ax, res.points, color="#00b0ff", label="PSO Bezier")
    if ctx.astar_path is not None:
        viz.plot_path(ax, ctx.astar_path, color="#00e676",
                      label="A* (smoothed)", linewidth=1.6)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    ctx.save(fig, "06_pso.png")


def _draw_state(ax, res, idx):
    pitch.draw_pitch(ax)
    pos = res.pos_history[idx]
    att = res.team_ids == TEAM_ATTACK
    dfn = res.team_ids == TEAM_DEFEND
    trail = res.pos_history[:idx + 1]
    ax.plot(trail[:, res.carrier_idx, 0], trail[:, res.carrier_idx, 1],
            color=ATTACK_C, alpha=0.35, lw=1.5, zorder=2)
    for k in np.where(dfn)[0]:
        ax.plot(trail[:, k, 0], trail[:, k, 1], color=DEFEND_C, alpha=0.3,
                lw=1.2, zorder=2)
    plan = res.plan_history[idx]
    ax.plot(plan[:, 0], plan[:, 1], "--", color="#00e676", lw=2.0, zorder=3)
    ax.scatter(pos[att, 0], pos[att, 1], s=150, c=ATTACK_C, edgecolors="white",
               linewidths=1.2, zorder=5)
    ax.scatter(pos[dfn, 0], pos[dfn, 1], s=150, c=DEFEND_C, edgecolors="white",
               linewidths=1.2, zorder=5)
    cp = pos[res.carrier_idx]
    ax.scatter(cp[0], cp[1], s=320, facecolors="none", edgecolors="#ffcc00",
               linewidths=2.2, zorder=6)
    ax.scatter(*GOAL, s=240, marker="*", color="#00e676", edgecolors="black",
               zorder=6)
    if res.intercepted and idx == len(res.times) - 1:
        ax.scatter(*res.intercept_point, s=380, marker="X", color="#ff1744",
                   edgecolors="black", linewidths=1.5, zorder=8)


def _status(res, idx, label):
    if res.intercepted and idx == len(res.times) - 1:
        return f"{label}: INTERCEPTED at {res.progress:.0%}"
    if res.reached_goal and idx == len(res.times) - 1:
        return f"{label}: reached goal"
    return f"{label}  (t = {res.times[idx]:.1f}s)"


def stage_reactive(ctx: Ctx, make_gif: bool) -> None:
    print("[7/7] Reactive rollout: static plan vs. replanning")
    cost0 = value_surface.cost_map(ctx.frame, ctx.centers, weights=WEIGHTS)
    ast0 = grid_search.plan_astar(cost0, ctx.centers, ctx.start, GOAL,
                                  terrain_weight=REACTIVE.terrain_weight)
    plan0 = grid_search.smooth_path(ast0.points, cost0, ctx.centers,
                                    terrain_weight=REACTIVE.terrain_weight).points
    static = rc.rollout_static(ctx.frame, plan0, GOAL, params=REACTIVE)
    replan = rc.rollout_replanning(ctx.frame, ctx.centers, GOAL,
                                   replan_every=4, weights=WEIGHTS,
                                   params=REACTIVE)
    print(f"  static : reached={static.reached_goal} "
          f"intercepted={static.intercepted} progress={static.progress:.0%}")
    print(f"  replan : reached={replan.reached_goal} "
          f"intercepted={replan.intercepted} progress={replan.progress:.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(21, 7))
    _draw_state(axes[0], static, len(static.times) - 1)
    axes[0].set_title(_status(static, len(static.times) - 1, "Static plan"))
    _draw_state(axes[1], replan, len(replan.times) - 1)
    axes[1].set_title(_status(replan, len(replan.times) - 1, "Replanning"))
    fig.tight_layout()
    ctx.save(fig, "07_reactive_summary.png")

    if make_gif:
        print("  rendering GIF (slow) ...")
        n = max(len(static.times), len(replan.times)) + 6
        fig2, axes2 = plt.subplots(1, 2, figsize=(21, 7))

        def update(t):
            for ax, res, label in ((axes2[0], static, "Static plan"),
                                   (axes2[1], replan, "Replanning")):
                ax.clear()
                idx = min(t, len(res.times) - 1)
                _draw_state(ax, res, idx)
                ax.set_title(_status(res, idx, label))
            return []

        anim = FuncAnimation(fig2, update, frames=n, interval=140, blit=False)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        gif = OUT_DIR / "07_reactive.gif"
        anim.save(gif, writer=PillowWriter(fps=7))
        plt.close(fig2)
        print(f"  saved -> {gif}")


def main() -> None:
    frame_index = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    t0 = time.time()
    ctx = Ctx(frame_index)

    if STAGES["voronoi"]:
        stage_voronoi(ctx)
    if STAGES["pitch_control"]:
        stage_pitch_control(ctx)
    if STAGES["cost_map"]:
        stage_cost_map(ctx)
    if STAGES["astar"]:
        stage_astar(ctx)
    if STAGES["potential_fields"]:
        stage_potential_fields(ctx)
    if STAGES["pso"]:
        stage_pso(ctx)
    if STAGES["reactive"]:
        stage_reactive(ctx, make_gif=STAGES["reactive_gif"])

    plt.close("all")
    print(f"\nDone in {time.time() - t0:.1f}s. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
