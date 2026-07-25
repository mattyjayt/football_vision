"""CLI entrypoint for the detection pipeline.

Usage:
    # Single-frame ground-truth test
    uv run python -m detection_pipeline \
        --source data/roboflow_video.mp4 \
        --mode RADAR_SINGLE --frame 120 \
        --out data/radar_frame_120.jsonl

    # Full video (not yet implemented)
    uv run python -m detection_pipeline \
        --source data/roboflow_video.mp4 \
        --mode RADAR_VIDEO \
        --out data/radar_video.jsonl
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .config import PIPELINE, PitchSpec
from .frames import RadarFrame, RadarPlayer, write_jsonl
from .infer import (
    detect_keypoints,
    detect_players_and_ball,
    load_keypoint_model,
    load_player_model,
)
from .track import create_tracker
from .transform import compute_homography, project_points


class Mode(Enum):
    RADAR_SINGLE = "RADAR_SINGLE"
    RADAR_VIDEO = "RADAR_VIDEO"


def process_single_frame(
    source: str,
    frame_id: int,
    config: dict,
) -> RadarFrame:
    """Run the full pipeline on ONE frame of a video.

    This is the ground-truth stitch test: prove that video → inference →
    homography → radar-view JSONL works end-to-end for a single frame
    before scaling to full video.
    """
    pitch_spec = PitchSpec(
        length_m=config["output"]["pitch_length_m"],
        width_m=config["output"]["pitch_width_m"],
    )

    # --- Load models ---
    det_cfg = config["detector"]
    player_model = load_player_model(det_cfg["player_model_id"])
    keypoint_model = load_keypoint_model(det_cfg["keypoint_model_id"])

    # --- Read frame ---
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id} from {source}")

    # --- Inference ---
    players, ball = detect_players_and_ball(
        frame, player_model,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"],
        ball_conf=det_cfg["ball_conf"],
    )
    keypoints = detect_keypoints(
        frame, keypoint_model,
        min_conf=det_cfg["keypoint_conf"],
    )

    print(f"Frame {frame_id}: {len(players)} players detected, "
          f"ball={'yes' if ball else 'no'}, "
          f"{keypoints.n_keypoints} keypoints")

    # --- Homography ---
    hom_cfg = config["homography"]
    H, rms_error = compute_homography(
        keypoints, min_keypoints=hom_cfg["min_keypoints"]
    )

    if H is None:
        print(f"  WARNING: insufficient keypoints ({keypoints.n_keypoints} < "
              f"{hom_cfg['min_keypoints']}), homography skipped")
        homography_list = None
        keypoints_used = []
    elif rms_error > hom_cfg["rms_reproj_threshold"]:
        print(f"  WARNING: homography RMS error {rms_error:.2f}m exceeds "
              f"threshold {hom_cfg['rms_reproj_threshold']}m, rejecting")
        homography_list = None
        keypoints_used = []
        H = None
    else:
        print(f"  Homography OK (RMS reproj error: {rms_error:.3f}m)")
        homography_list = H.tolist()
        keypoints_used = sorted(keypoints.points.keys())

    # --- Project to pitch coordinates ---
    tracked_players: list[RadarPlayer] = []
    ball_pitch = None

    if H is not None:
        tracker = create_tracker("single_frame")
        tracked = tracker.update(players, H)

        for tp in tracked:
            tracked_players.append(RadarPlayer(
                track_id=tp.track_id,
                class_name=tp.class_name,
                team=None,  # team classification deferred
                pitch_xy_m=[float(tp.pitch_xy[0]), float(tp.pitch_xy[1])],
                pitch_vxy_ms=[0.0, 0.0],  # single frame = zero velocity
            ))

        if ball is not None:
            ball_m = project_points(ball.center, H)
            ball_pitch = {"pitch_xy_m": [float(ball_m[0]), float(ball_m[1])]}

    # --- Carrier assignment ---
    carrier_id = None
    if ball_pitch is not None and tracked_players:
        carrier_cfg = config["carrier"]
        ball_xy = np.array(ball_pitch["pitch_xy_m"])
        min_dist = float("inf")
        for p in tracked_players:
            dist = np.linalg.norm(np.array(p.pitch_xy_m) - ball_xy)
            if dist < min_dist:
                min_dist = dist
                carrier_id = p.track_id
        if min_dist > carrier_cfg["max_distance_m"]:
            carrier_id = None  # ball is loose / in the air

    return RadarFrame(
        frame_id=frame_id,
        source=Path(source).name,
        pitch={
            "length_m": pitch_spec.length_m,
            "width_m": pitch_spec.width_m,
            "origin": "center",
            "attack": config["output"]["attack_direction"],
        },
        homography=homography_list,
        keypoints_used=keypoints_used,
        players=tracked_players,
        ball=ball_pitch,
        carrier_track_id=carrier_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detection pipeline: video → radar-view JSONL"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source video file")
    parser.add_argument("--mode", type=Mode, default=Mode.RADAR_SINGLE,
                        choices=list(Mode),
                        help="RADAR_SINGLE: one frame. RADAR_VIDEO: all frames (not yet implemented)")
    parser.add_argument("--frame", type=int, default=120,
                        help="Frame index for RADAR_SINGLE mode")
    parser.add_argument("--out", type=str, default="data/radar_output.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    if args.mode == Mode.RADAR_VIDEO:
        raise NotImplementedError(
            "RADAR_VIDEO is stubbed. Design notes:\n"
            "  - Use SORTTracker (pitch-plane tracking)\n"
            "  - Finite-difference velocities across frames\n"
            "  - Temporal smoothing of homography (median over window)\n"
            "  - Team classification via SigLIP embeddings\n"
            "  See detection_pipeline/config.py for tracker swap."
        )

    # RADAR_SINGLE
    radar_frame = process_single_frame(args.source, args.frame, PIPELINE)
    write_jsonl([radar_frame], args.out)
    print(f"\nWritten: {args.out}")
    print(f"  Players: {len(radar_frame.players)}")
    print(f"  Ball: {'yes' if radar_frame.ball else 'no'}")
    print(f"  Carrier: {radar_frame.carrier_track_id}")
    print(f"  Homography: {'OK' if radar_frame.homography else 'SKIPPED'}")


if __name__ == "__main__":
    main()
