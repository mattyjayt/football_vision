"""Phase 2 demo: Spearman pitch control surfaces.

Run standalone:

    uv run python scripts/02_demo_pitch_control.py

Computes the potential pitch control field for several scenarios and draws each
as a heatmap (red = attacking control, blue = defending control) with players
and velocity arrows on top. Prints the attacking team's mean control and the
control at the ball carrier's feet.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footlab import pitch, pitch_control, simulate, viz


def main() -> None:
    scenarios = {
        "Counter attack": simulate.scenario_counter_attack(),
        "Low block": simulate.scenario_low_block(),
        "Wing overload": simulate.scenario_wing_overload(),
        "Random 11v11": simulate.scenario_random(seed=0),
    }
    _, _, centers = pitch.make_grid(cell_size=1.0)

    print("Pitch control (attacking team)")
    print("-" * 64)

    n = len(scenarios)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 9.5, nrows * 6.2))
    axes = axes.ravel()

    for ax, (name, frame) in zip(axes, scenarios.items()):
        t0 = time.perf_counter()
        surf = pitch_control.pitch_control_surface(frame, centers)
        dt = time.perf_counter() - t0
        p_att, p_def = pitch_control.pitch_control_at_target(
            frame, frame.carrier_position)
        print(f"{name:16s} | mean att {surf.mean():5.1%} "
              f"| at carrier att {p_att:.2f}/def {p_def:.2f} "
              f"| {surf.shape[1]}x{surf.shape[0]} cells in {dt:4.2f}s")
        viz.plot_surface(frame, surf, ax=ax, title=f"{name} — pitch control")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "02_pitch_control.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("-" * 64)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
