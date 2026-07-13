"""Phase 2b demo: pitch control, value, defender penalty, and combined cost.

Run standalone:

    uv run python scripts/02b_demo_cost_map.py

For two scenarios it draws the four layers side by side, then shows how changing
the cost weights reshapes the combined cost (safe/cautious vs. direct/risky).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from footlab import pitch, pitch_control, simulate, value_surface, viz


def main() -> None:
    scenarios = {
        "Counter attack": simulate.scenario_counter_attack(),
        "Low block": simulate.scenario_low_block(),
    }
    _, _, centers = pitch.make_grid(cell_size=1.0)

    # --- Figure 1: the four layers for each scenario -------------------------
    fig, axes = plt.subplots(len(scenarios), 4, figsize=(34, 6.0 * len(scenarios)))
    for row, (name, frame) in enumerate(scenarios.items()):
        control = pitch_control.pitch_control_surface(frame, centers)
        value = value_surface.positional_value_surface(centers)
        penalty = value_surface.defender_proximity_penalty(frame, centers)
        cost = value_surface.cost_map(frame, centers, control=control)

        viz.plot_surface(frame, control, ax=axes[row, 0],
                         title=f"{name} — pitch control", cmap="RdBu_r",
                         cbar_label="P(attack control)")
        viz.plot_surface(frame, value, ax=axes[row, 1],
                         title=f"{name} — value (xT surrogate)", cmap="viridis",
                         cbar_label="positional value")
        viz.plot_surface(frame, penalty, ax=axes[row, 2],
                         title=f"{name} — defender penalty", cmap="inferno",
                         cbar_label="proximity penalty")
        viz.plot_surface(frame, cost, ax=axes[row, 3],
                         title=f"{name} — combined cost", cmap="magma_r",
                         vmin=float(cost.min()), vmax=float(cost.max()),
                         cbar_label="cost (lower = better)")
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / "02b_cost_layers.png"
    fig.savefig(out1, dpi=110, bbox_inches="tight")

    # --- Figure 2: weight sensitivity on the counter-attack cost -------------
    frame = scenarios["Counter attack"]
    control = pitch_control.pitch_control_surface(frame, centers)
    weightings = {
        "Balanced (1,1,1)": value_surface.CostWeights(1.0, 1.0, 1.0),
        "Cautious (2,0.5,2)": value_surface.CostWeights(2.0, 0.5, 2.0),
        "Direct/risky (0.5,2,0.5)": value_surface.CostWeights(0.5, 2.0, 0.5),
    }
    fig2, axes2 = plt.subplots(1, 3, figsize=(28, 6.2))
    print("Cost-map weight sensitivity (counter attack)")
    print("-" * 60)
    for ax, (label, w) in zip(axes2, weightings.items()):
        cost = value_surface.cost_map(frame, centers, control=control, weights=w)
        print(f"{label:26s} | cost range [{cost.min():.2f}, {cost.max():.2f}] "
              f"| mean {cost.mean():.2f}")
        viz.plot_surface(frame, cost, ax=ax, title=label, cmap="magma_r",
                         vmin=float(cost.min()), vmax=float(cost.max()),
                         cbar_label="cost (lower = better)")
    fig2.tight_layout()
    out2 = out_dir / "02b_cost_weights.png"
    fig2.savefig(out2, dpi=110, bbox_inches="tight")

    print("-" * 60)
    print(f"Saved figures -> {out1}\n                {out2}")


if __name__ == "__main__":
    main()
