"""Demo: RADAR_SINGLE end-to-end with side-by-side visualization.

Runs the detection pipeline on one frame of a video, writes the JSONL,
loads it back into a FrozenFrame via footlab.io_radar, and renders a
side-by-side comparison: original frame with keypoints + radar view.

Usage:
    uv run python scripts/10_demo_radar_single.py
    uv run python scripts/10_demo_radar_single.py --source data/video.mp4 --frame 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
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
from detection_pipeline.config import PIPELINE, KEYPOINT_VERTICES_M
from detection_pipeline.frames import write_jsonl, read_jsonl
from detection_pipeline.infer import load_keypoint_model, detect_keypoints


def render_side_by_side(
    source_video: str,
    frame_id: int,
    jsonl_path: str,
    output_png: str,
) -> None:
    """Render original frame with keypoints + radar view side by side."""

    # Load the radar frame data
    radar_frames = read_jsonl(jsonl_path)
    radar_frame = radar_frames[0]

    # Read the original video frame
    cap = cv2.VideoCapture(source_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id} from {source_video}")

    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- LEFT: Original frame with keypoints ---
    ax1.imshow(frame_rgb)
    ax1.set_title(f"Original Frame {frame_id} — Keypoints Used for Homography",
                  fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Draw keypoints on the original frame
    det_cfg = PIPELINE["detector"]
    if det_cfg.get("keypoint_model_type", "roboflow") == "local":
        keypoint_model = load_keypoint_model(det_cfg["keypoint_model_path"], model_type="local")
    else:
        keypoint_model = load_keypoint_model(det_cfg["keypoint_model_id"])
    kps = detect_keypoints(frame, keypoint_model,
                          min_conf=PIPELINE["detector"]["keypoint_conf"])

    for kp_id, pix_xy in kps.points.items():
        if kp_id in KEYPOINT_VERTICES_M:
            pitch_xy = KEYPOINT_VERTICES_M[kp_id]
            conf = kps.confidences[kp_id]
            # Draw only mid+ confidence keypoints to keep the image clean
            if conf < 0.5:
                continue
            color = "lime" if conf > 0.7 else "yellow"

            ax1.scatter(pix_xy[0], pix_xy[1], c=color, s=150, marker="o",
                       edgecolors="black", linewidths=2, zorder=5)
            ax1.text(pix_xy[0] + 10, pix_xy[1] - 10,
                    f"{kp_id}\n({pitch_xy[0]:.0f},{pitch_xy[1]:.0f})",
                    fontsize=8, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # Add legend for keypoint colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=10, label="High conf (>0.7)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markersize=10, label="Med conf (0.5-0.7)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # --- RIGHT: Pitch with projected keypoints (no players yet) ---
    draw_pitch(ax2)

    # Project each detected image-space keypoint through H and plot where it
    # lands on the pitch vs its true landmark position — a direct visual of
    # the homography's accuracy.
    H = np.array(radar_frame.homography) if radar_frame.homography is not None else None
    for kp_id, pix_xy in kps.points.items():
        if kp_id not in KEYPOINT_VERTICES_M or kps.confidences[kp_id] < 0.5:
            continue
        true_xy = KEYPOINT_VERTICES_M[kp_id]
        # True landmark position
        ax2.scatter(true_xy[0], true_xy[1], c="white", s=60, marker="o",
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax2.text(true_xy[0] + 1, true_xy[1] + 1, str(kp_id), fontsize=8,
                color="white", weight="bold", zorder=6)
        # Projected position (where H thinks the detected point is)
        if H is not None:
            pt = np.array(pix_xy, dtype=np.float64).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(pt, H).reshape(2)
            ax2.scatter(proj[0], proj[1], c="red", s=60, marker="x",
                       linewidths=2, zorder=6)

    legend2 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="True landmark"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="red",
               markersize=8, label="Projected (via H)", linestyle="None"),
    ]
    ax2.legend(handles=legend2, loc="upper right", fontsize=9)
    ax2.set_title("Pitch — Keypoint Projection Check", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    # Add homography info
    if radar_frame.homography is not None:
        rms_text = f"Homography: {len(radar_frame.keypoints_used)} keypoints used"
        if hasattr(radar_frame, 'rms_error'):
            rms_text += f", RMS error: {radar_frame.rms_error:.2f}m"
        fig.text(0.5, 0.02, rms_text, ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Side-by-side view saved: {output_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="RADAR_SINGLE end-to-end demo")
    parser.add_argument("--source", type=str, default="data/roboflow_video.mp4")
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Output filename reflects the actual frame unless explicitly overridden
    if args.out is None:
        args.out = f"data/radar_frame_{args.frame}.jsonl"

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

    # Step 3: Render side-by-side comparison
    print(f"\n[3/3] Rendering side-by-side comparison...")
    png_path = args.out.replace(".jsonl", "_comparison.png")
    render_side_by_side(args.source, args.frame, args.out, png_path)

    print(f"\n{'=' * 60}")
    print("Done. Check the comparison PNG to verify the homography.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
