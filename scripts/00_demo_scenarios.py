"""Phase 0 demo: render every scenario + an augmentation example.

Run standalone:

    uv run python scripts/00_demo_scenarios.py

Saves a figure to ``figures/00_scenarios.png`` (the directory is gitignored) and
prints a one-line summary of each frame so you can eyeball the invariants.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; we save a PNG rather than pop a window
import matplotlib.pyplot as plt

from footlab import simulate, viz


def main() -> None:
    frames = {
        "Counter attack (3v2 + GK)": simulate.scenario_counter_attack(),
        "Low block (10 defenders)": simulate.scenario_low_block(),
        "Wing overload (3v2)": simulate.scenario_wing_overload(),
        "Random 11v11 (seed=0)": simulate.scenario_random(seed=0),
    }

    # Show augmentation on the random frame: jitter positions + perturb velocity,
    # then mirror lengthwise. This is the noise a real pipeline injects.
    rnd = frames["Random 11v11 (seed=0)"]
    noisy = simulate.perturb_velocities(
        simulate.jitter_positions(rnd, sigma=0.6, seed=1), sigma=0.6, seed=2)
    frames["Random + jitter/perturb noise"] = noisy
    frames["Random, mirrored lengthwise"] = simulate.mirror_lengthwise(rnd)

    print("Scenario summary")
    print("-" * 68)
    for name, ff in frames.items():
        print(f"{name:34s} | {ff.n_players:2d} players "
              f"({ff.attack_mask.sum():2d} atk / {ff.defend_mask.sum():2d} def) "
              f"| vmax {ff.speeds.max():4.1f} m/s | carrier {ff.ball_carrier}")

    n = len(frames)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 9.5, nrows * 6.2))
    axes = axes.ravel()
    for ax, (name, ff) in zip(axes, frames.items()):
        viz.plot_frame(ff, ax=ax, title=name)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "00_scenarios.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("-" * 68)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
