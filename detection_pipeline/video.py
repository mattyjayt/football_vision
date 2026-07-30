"""RADAR_VIDEO mode — walk a whole video with persistent tracking.

The difference from RADAR_SINGLE: a tracker (ByteTrack) persists IDs across
frames, and velocities become REAL (finite differences of pitch positions
per track), not zeros. Frames where the homography is rejected emit no
positions (a gap, not a fabrication) — footlab tolerates missing data.

Flow per frame:
    detect players/ball → detect keypoints → homography (validated)
    → ByteTrack (pixel space) → project feet to pitch → finite-diff v
    → optional team assignment (fitted once up front) → RadarFrame

Output: one JSONL line per successfully processed frame. Frames rejected at
the homography gate are logged to the console but skipped in the output —
the loader never sees a warped world.
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from pathlib import Path

import cv2
import numpy as np

from .config import PitchSpec
from .frames import RadarFrame, RadarPlayer
from .infer import (
    detect_keypoints,
    detect_players_and_ball,
    load_keypoint_model,
    load_player_model,
)
from .track import create_tracker
from .transform import compute_homography


def process_video(
    source: str,
    config: dict,
    team_classifier: object | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    verbose: bool = True,
) -> list[RadarFrame]:
    """Run the tracking pipeline over a video.

    Args:
        stride: process every Nth frame (1 = every frame; velocities are
            computed against the PREVIOUS PROCESSED frame, so the tracker's
            internal dt uses stride/fps — passed via tracker kwargs).
        max_frames: cap on processed frames (None = whole video).
    """
    det_cfg = config["detector"]
    hom_cfg = config["homography"]
    trk_cfg = config.get("tracker", {})
    pitch_spec = PitchSpec(
        length_m=config["output"]["pitch_length_m"],
        width_m=config["output"]["pitch_width_m"],
    )

    cap = cv2.VideoCapture(source)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    player_model = load_player_model(det_cfg["player_model_path"], model_type="local")
    kp_type = det_cfg.get("keypoint_model_type", "roboflow")
    keypoint_model = (
        load_keypoint_model(det_cfg["keypoint_model_path"], model_type="local")
        if kp_type == "local"
        else load_keypoint_model(det_cfg["keypoint_model_id"])
    )

    tracker = create_tracker(
        trk_cfg.get("type", "bytetrack"),
        fps=fps / stride,
        lost_track_buffer=trk_cfg.get("max_age", 30),
        minimum_consecutive_frames=trk_cfg.get("min_hits", 3),
    )

    frame_ids = list(range(0, max(n_frames, 1), stride))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if verbose:
        print(f"RADAR_VIDEO: {len(frame_ids)} frames to process "
              f"(video {n_frames} @ {fps:.0f}fps, stride={stride})")

    frames_out: list[RadarFrame] = []
    n_h_rejected = 0

    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        players, ball = detect_players_and_ball(
            frame, player_model,
            class_map=det_cfg["class_map"],
            player_conf=det_cfg["player_conf"], ball_conf=det_cfg["ball_conf"],
        )
        kps = detect_keypoints(frame, keypoint_model,
                               min_conf=det_cfg["keypoint_conf"])
        H, rms = compute_homography(
            kps,
            min_keypoints=hom_cfg["min_keypoints"],
            min_confidence=hom_cfg.get("min_keypoint_conf", 0.0),
        )
        h_ok = H is not None and rms <= hom_cfg.get("rms_reproj_threshold", 2.0)

        if not h_ok:
            # Track STILL updates (keeps IDs alive through the gap) but no
            # positions are emitted for this frame.
            n_h_rejected += 1
            if verbose and n_h_rejected <= 5:
                print(f"  frame {fid}: homography rejected "
                      f"(kpts={kps.n_keypoints}, rms={rms:.2f}) — gap emitted")
            continue

        tracked = tracker.update(players, H)

        # Team assignment (optional, detection order == tracked order for
        # single-frame consistency; tracked subset matched by IoU inside).
        team_labels: list[str | None] = [None] * len(tracked)
        attacking_team = None
        if team_classifier is not None and tracked:
            from .teams_pipeline import assign_teams, crop_detections
            det_pitch = np.array([tp.pitch_xy for tp in tracked])
            crops = crop_detections(frame, players[: len(tracked)])
            assignment = assign_teams(
                crops, players[: len(tracked)], det_pitch, team_classifier,
            )
            team_labels = [
                None if lab == -1 else str(lab) for lab in assignment.labels
            ]
            team_labels = [
                "referee" if tp.class_name == "referee" else tl
                for tp, tl in zip(tracked, team_labels)
            ]
            attacking_team = assignment.attacking_team

        tracked_players = [
            RadarPlayer(
                track_id=tp.track_id,
                class_name=tp.class_name,
                team=tl,
                pitch_xy_m=[float(tp.pitch_xy[0]), float(tp.pitch_xy[1])],
                pitch_vxy_ms=[float(tp.pitch_vxy[0]), float(tp.pitch_vxy[1])],
            )
            for tp, tl in zip(tracked, team_labels)
        ]

        ball_pitch = None
        if ball is not None:
            from .transform import project_points
            bm = project_points(ball.center.reshape(1, 2), H)[0]
            ball_pitch = {"pitch_xy_m": [float(bm[0]), float(bm[1])]}

        carrier_id = None
        if ball_pitch is not None and tracked_players:
            max_d = config["carrier"]["max_distance_m"]
            bxy = np.array(ball_pitch["pitch_xy_m"])
            dists = [(np.linalg.norm(np.array(p.pitch_xy_m) - bxy), p.track_id)
                     for p in tracked_players]
            d, carrier_id = min(dists)
            if d > max_d:
                carrier_id = None

        frames_out.append(RadarFrame(
            frame_id=fid,
            source=Path(source).name,
            pitch={
                "length_m": pitch_spec.length_m,
                "width_m": pitch_spec.width_m,
                "origin": "center",
                "attack": config["output"]["attack_direction"],
            },
            homography=H.tolist(),
            keypoints_used=sorted(kps.points.keys()),
            players=tracked_players,
            ball=ball_pitch,
            carrier_track_id=carrier_id,
            attacking_team=attacking_team,
        ))
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frame_ids)} processed "
                  f"({len(frames_out)} emitted, {n_h_rejected} gaps)")

    cap.release()
    if verbose:
        print(f"RADAR_VIDEO done: {len(frames_out)} frames emitted, "
              f"{n_h_rejected} homography gaps")
    return frames_out
