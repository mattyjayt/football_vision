"""SkillCorner demo: footlab's tactical engine run on a REAL professional match.

Run standalone (requires the SkillCorner open data, see notes below):

    uv run python scripts/07_demo_skillcorner.py [frame_index]

This is the payoff of the whole ``FrozenFrame`` design: the exact same
pipeline that runs on simulated frames — pitch control -> cost map -> A* path
to goal — now runs, unchanged, on broadcast tracking from a real 2024/25
A-League match (Melbourne Victory vs Auckland FC, SkillCorner open data).

Three panels:
    1. The raw frozen frame (players, velocities, ball, carrier).
    2. Spearman pitch control over the real frame.
    3. The combined cost map with the A* optimal carrier path to goal.

Data: ``git clone https://github.com/SkillCorner/opendata.git`` into
``data/skillcorner`` (gitignored; needs git-lfs for the tracking .jsonl).
If the data is absent the demo exits with a clear message rather than crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footlab import pitch, skillcorner, value_surface, viz
from footlab.pitch_control import pitch_control_surface
from footlab.planners import grid_search

DATA_DIR = (FsPath(__file__).resolve().parent.parent
            / "data" / "skillcorner" / "data" / "matches" / "1886347")


def main() -> None:
    if not DATA_DIR.exists():
        print("SkillCorner data not found. Get it with:\n"
              "  cd <repo> && git clone "
              "https://github.com/SkillCorner/opendata.git data/skillcorner\n"
              "(requires git-lfs for the tracking .jsonl files)")
        sys.exit(1)

    frame_index = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"Loading real frame {frame_index} from {DATA_DIR} ...")
    frame = skillcorner.frames_to_frozen(DATA_DIR, frame_index)

    start = frame.carrier_position
    if start is None:
        start = frame.ball_pos
        print("(ball loose in this frame — planning from the ball position)")
    goal = pitch.ATTACKING_GOAL_CENTER

    print(f"players {frame.n_players} "
          f"({frame.attack_mask.sum()} att / {frame.defend_mask.sum()} def) | "
          f"carrier idx {frame.ball_carrier} | start {np.round(start, 1)} | "
          f"ball {np.round(frame.ball_pos, 1)}")

    _, _, centers = pitch.make_grid(cell_size=1.0)
    control = pitch_control_surface(frame, centers)
    cost = value_surface.cost_map(frame, centers, control=control)
    ast = grid_search.plan_astar(cost, centers, start, goal)
    sm = grid_search.smooth_path(ast.points, cost, centers)

    print(f"A* path: {len(ast.points)} waypoints, cost {ast.cost:.1f}, "
          f"expanded {ast.n_expanded} nodes | smoothed {sm.cost:.1f} "
          f"({len(sm.points)} pts)")

    # --- Three-panel figure ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

    viz.plot_frame(frame, ax=axes[0],
                   title=f"Real frame {frame_index} — Melbourne V vs Auckland")
    viz.plot_surface(frame, control, ax=axes[1],
                     title="Pitch control (attacking team)",
                     cmap="RdYlGn", vmin=0.0, vmax=1.0,
                     cbar_label="P(attack controls ball)")
    viz.plot_surface(frame, cost, ax=axes[2],
                     title="Cost map + A* optimal carrier path",
                     cmap="magma_r", vmin=float(cost.min()),
                     vmax=float(cost.max()), cbar_label="cost (lower=better)")
    viz.plot_path(axes[2], sm.points, color="#00e676", label="A* (smoothed)")
    axes[2].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_dir = FsPath(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "07_skillcorner_real_frame.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
