"""Demo: RADAR_SINGLE end-to-end with visualization.

Runs the detection pipeline on one frame of a video, writes the JSONL,
loads it back into a FrozenFrame via footlab.io_radar, and renders the
radar view on footlab's pitch for visual verification of the homography.

Usage:
    uv run python scripts/10_demo_radar_single.py
    uv run python scripts/10_demo_radar_single.py --source data/video.mp4 --frame 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add repo root to path so we can import both packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footlab.pitch import PITCH_LENGTH, PITCH_WIDTH, draw_pitch
from footlab.io_radar import load_radar_jsonl
from footlab.state import TEAM_ATTACK, TEAM_DEFEND

from detection_pipeline.__main__ import process_single_frame
from detection_pipeline.config import PIPELINE
from detection_pipeline.frames import write_jsonl


def render_radar_view(jsonl_path: str, output_png: str) -> None:
    """Load the JSONL back into a FrozenFrame and render on footlab's pitch."""
    frame = load_radar_jsonl(jsonl_path)

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    draw_pitch(ax)

    # Plot players
    att_mask = frame.attack_mask
    def_mask = frame.defend_mask

    if att_mask.any():
        ax.scatter(frame.positions[att_mask, 0], frame.positions[att_mask, 1],
                   c="red", s=120, zorder=5, label="Attack", edgecolors="white", linewidths=1)
    if def_mask.any():
        ax.scatter(frame.positions[def_mask, 0], frame.positions[def_mask, 1],
                   c="blue", s=120, zorder=5, label="Defense", edgecolors="white", linewidths=1)

    # Plot ball
    ax.scatter(*frame.ball_pos, c="yellow", s=60, zorder=6, marker="o",
               edgecolors="black", linewidths=1, label="Ball")

    # Highlight carrier
    if frame.ball_carrier >= 0 and frame.carrier_position is not None:
        cp = frame.carrier_position
        ax.scatter(*cp, c="none", s=300, zorder=5, edgecolors="lime",
                   linewidths=2.5, label="Carrier")

    # Velocity arrows (all zero for single frame, but show the structure)
    for i in range(frame.n_players):
        v = frame.velocities[i]
        if np.linalg.norm(v) > 0.01:
            ax.arrow(frame.positions[i, 0], frame.positions[i, 1],
                     v[0] * 0.5, v[1] * 0.5,
                     head_width=0.8, head_length=0.4, fc="white", ec="white", alpha=0.7)

    ax.set_title(f"RADAR_SINGLE — {Path(jsonl_path).name}\n"
                 f"{frame.n_players} players, carrier={frame.ball_carrier}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Radar view saved: {output_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="RADAR_SINGLE end-to-end demo")
    parser.add_argument("--source", type=str, default="data/roboflow_video.mp4")
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--out", type=str, default="data/radar_frame_120.jsonl")
    args = parser.parse_args()

    print("=" * 60)
    print("RADAR_SINGLE — Detection Pipeline Demo")
    print("=" * 60)

    # Step 1: Run the pipeline
    print(f"\n[1/3] Processing frame {args.frame} from {args.source}...")
    radar_frame = process_single_frame(args.source, args.frame, PIPELINE)
    write_jsonl([radar_frame], args.out)
    print(f"  Written: {args.out}")

    # Step 2: Load back into FrozenFrame
    print(f"\n[2/3] Loading into FrozenFrame via io_radar...")
    frozen = load_radar_jsonl(args.out)
    print(f"  {frozen.n_players} players, carrier={frozen.ball_carrier}")
    print(f"  Ball at: {frozen.ball_pos}")

    # Step 3: Render radar view
    print(f"\n[3/3] Rendering radar view...")
    png_path = args.out.replace(".jsonl", "_radar_view.png")
    render_radar_view(args.out, png_path)

    print(f"\n{'=' * 60}")
    print("Done. Check the radar view PNG to verify the homography.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
