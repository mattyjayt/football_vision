"""Test script: isolate pitch keypoint detection and projection.

Shows original frame with detected keypoints + pitch with projected keypoints
ONLY (no players). This isolates the homography from player detection issues.

Usage:
    uv run python scripts/test_keypoints.py --source data/08fd33_0.mp4 --frame 50
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

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footlab.pitch import draw_pitch
from detection_pipeline.config import PIPELINE, KEYPOINT_VERTICES_M
from detection_pipeline.infer import load_keypoint_model, detect_keypoints
from detection_pipeline.transform import compute_homography, project_points


def test_keypoints(source: str, frame_id: int, output_png: str):
    """Test keypoint detection and projection in isolation."""

    # Read frame
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id} from {source}")

    # Detect keypoints
    keypoint_model = load_keypoint_model(PIPELINE["detector"]["keypoint_model_id"])
    kps = detect_keypoints(frame, keypoint_model,
                          min_conf=PIPELINE["detector"]["keypoint_conf"])

    print(f"Detected {kps.n_keypoints} keypoints:")
    for kp_id in sorted(kps.points.keys()):
        if kp_id in KEYPOINT_VERTICES_M:
            pitch_xy = KEYPOINT_VERTICES_M[kp_id]
            conf = kps.confidences[kp_id]
            print(f"  ID {kp_id:2d}: pixel ({kps.points[kp_id][0]:4.0f}, {kps.points[kp_id][1]:4.0f}) "
                  f"conf {conf:.2f} → pitch ({pitch_xy[0]:6.1f}, {pitch_xy[1]:6.1f})")

    # Compute homography
    H, rms_error = compute_homography(
        kps,
        min_keypoints=PIPELINE["homography"]["min_keypoints"],
        min_confidence=PIPELINE["homography"].get("min_keypoint_conf", 0.0),
    )

    if H is None:
        print("\nERROR: Could not compute homography")
        return

    print(f"\nHomography RMS error: {rms_error:.3f}m")

    # Project keypoints to pitch coordinates
    projected_keypoints = {}
    for kp_id, pix_xy in kps.points.items():
        if kp_id in KEYPOINT_VERTICES_M:
            pitch_xy = project_points(pix_xy, H)
            expected_xy = np.array(KEYPOINT_VERTICES_M[kp_id])
            error = np.linalg.norm(pitch_xy - expected_xy)
            projected_keypoints[kp_id] = {
                'pixel': pix_xy,
                'projected': pitch_xy,
                'expected': expected_xy,
                'error': error,
                'conf': kps.confidences[kp_id],
            }

    # Print projection errors
    print("\nProjection errors:")
    for kp_id in sorted(projected_keypoints.keys()):
        kp = projected_keypoints[kp_id]
        print(f"  ID {kp_id:2d}: projected ({kp['projected'][0]:6.1f}, {kp['projected'][1]:6.1f}) "
              f"expected ({kp['expected'][0]:6.1f}, {kp['expected'][1]:6.1f}) "
              f"error {kp['error']:5.1f}m")

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- LEFT: Original frame with keypoints ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax1.imshow(frame_rgb)
    ax1.set_title(f"Original Frame {frame_id} — {kps.n_keypoints} Keypoints Detected",
                  fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Draw keypoints on original frame
    for kp_id, pix_xy in kps.points.items():
        if kp_id in KEYPOINT_VERTICES_M:
            conf = kps.confidences[kp_id]
            # Color by confidence
            if conf > 0.7:
                color = "lime"
            elif conf > 0.4:
                color = "yellow"
            else:
                color = "red"

            ax1.scatter(pix_xy[0], pix_xy[1], c=color, s=150, marker="o",
                       edgecolors="black", linewidths=2, zorder=5)
            ax1.text(pix_xy[0] + 10, pix_xy[1] - 10, str(kp_id),
                    fontsize=10, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # --- RIGHT: Pitch with projected keypoints ---
    draw_pitch(ax2)
    ax2.set_title(f"Pitch — Projected Keypoints (RMS error: {rms_error:.3f}m)",
                  fontsize=13, fontweight="bold")

    # Draw expected keypoint positions (ground truth)
    for kp_id, expected_xy in KEYPOINT_VERTICES_M.items():
        ax2.scatter(expected_xy[0], expected_xy[1], c="blue", s=100, marker="x",
                   linewidths=2, alpha=0.3, zorder=3)

    # Draw projected keypoints
    for kp_id, kp in projected_keypoints.items():
        # Color by projection error
        if kp['error'] < 1.0:
            color = "lime"
        elif kp['error'] < 3.0:
            color = "yellow"
        else:
            color = "red"

        ax2.scatter(kp['projected'][0], kp['projected'][1], c=color, s=150, marker="o",
                   edgecolors="black", linewidths=2, zorder=5)
        ax2.text(kp['projected'][0] + 1, kp['projected'][1] + 1, str(kp_id),
                fontsize=10, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=10, label="Error < 1m"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markersize=10, label="Error 1-3m"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Error > 3m"),
        Line2D([0], [0], marker="x", color="blue", markersize=10,
               label="Expected position", linestyle="None"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test pitch keypoint detection")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--out", type=str, default="data/test_keypoints.png")
    args = parser.parse_args()

    test_keypoints(args.source, args.frame, args.out)


if __name__ == "__main__":
    main()
