"""Phase 1 demo: Voronoi control vs. dominant regions, side by side.

Run standalone:

    uv run python scripts/01_demo_voronoi.py

For two scenarios it draws, left-to-right, the nearest-player Voronoi partition
and the velocity-aware dominant region, filling each cell by the controlling
team. The exact scipy Voronoi boundaries are overlaid on the left panel as a
cross-check. Prints how the attacking team's share of the pitch changes once
velocity is taken into account.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footlab import geometry, pitch, simulate, viz


def _overlay_voronoi_edges(ax, frame) -> None:
    """Overlay exact scipy Voronoi boundaries (finite ridge segments only).

    We draw the finite ridges by hand rather than using ``voronoi_plot_2d`` so
    the unbounded ridges (which shoot off to infinity) don't clutter the panel.
    The remaining solid segments should trace the grid partition exactly.
    """
    try:
        vor = geometry.voronoi_diagram(frame)
    except Exception as exc:  # e.g. QhullError on degenerate inputs
        print(f"  (skipped exact Voronoi overlay: {exc})")
        return
    for v0, v1 in vor.ridge_vertices:
        if v0 == -1 or v1 == -1:
            continue  # unbounded ridge
        seg = vor.vertices[[v0, v1]]
        ax.plot(seg[:, 0], seg[:, 1], color="#333333", lw=0.9, alpha=0.7,
                zorder=1.5, clip_on=True)


def main() -> None:
    scenarios = {
        "Counter attack": simulate.scenario_counter_attack(),
        "Wing overload": simulate.scenario_wing_overload(),
    }
    _, _, centers = pitch.make_grid(cell_size=1.0)

    print("Attack's share of the pitch")
    print("-" * 60)

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(19, 6.2 * len(scenarios)))
    axes = np.atleast_2d(axes)
    for row, (name, frame) in enumerate(scenarios.items()):
        vor = geometry.voronoi_control(frame, centers)
        dom = geometry.dominant_region(frame, centers)

        atk_v = geometry.team_area_fraction(frame, vor)
        atk_d = geometry.team_area_fraction(frame, dom)
        changed = float(np.mean(vor != dom))
        print(f"{name:16s} | Voronoi {atk_v:5.1%} -> Dominant {atk_d:5.1%} "
              f"| owner changed on {changed:4.1%} of cells")

        viz.plot_team_regions(frame, geometry.control_to_team(frame, vor),
                              ax=axes[row, 0],
                              title=f"{name} — Voronoi (nearest player)")
        _overlay_voronoi_edges(axes[row, 0], frame)
        viz.plot_team_regions(frame, geometry.control_to_team(frame, dom),
                              ax=axes[row, 1],
                              title=f"{name} — Dominant region (velocity-aware)")

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "01_voronoi_vs_dominant.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("-" * 60)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
