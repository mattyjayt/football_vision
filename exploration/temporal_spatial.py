"""Spatial analysis (Phase 3) — where things happen on the pitch, and how
stable the geometry is over time.

Three instruments:

1. POSITION HEATMAPS
   2D histogram of projected player positions in pitch coordinates, per team,
   over a frame window. Reveals football structure (formations, pressing)
   AND homography failure: positions smeared beyond the touchlines or
   clustered impossibly mean H was bad in that window. We also split frames
   by homography-RMS quartile so you can literally see "positions when H was
   good" vs "positions when H was bad".

2. KEYPOINT COVERAGE MAP
   Which of the 32 pitch landmarks the model detects, drawn on the pitch,
   intensity = detection frequency across the video. Exposes the camera's
   blind spots (e.g. far-side corners on a single-side broadcast rig) —
   direct evidence of what footage this setup can and cannot register.

3. HOMOGRAPHY JITTER TRACE
   Per frame: how much does H move the four pitch corners (in pixels) vs the
   previous frame? A smooth camera pan yields a smooth trace; spikes mark
   frame-to-frame instability that tracking/smoothing must absorb. This is
   the number that will tell us whether H temporal smoothing (Path B step 4)
   is luxury or necessity.
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detection_pipeline.config import KEYPOINT_VERTICES_M, PitchSpec
from detection_pipeline.infer import detect_keypoints, detect_players_and_ball
from detection_pipeline.transform import compute_homography, project_points


@dataclass
class SpatialFrameData:
    """Per-frame spatial measurements kept for the heatmaps/jitter."""

    frame_id: int
    time_s: float
    positions: np.ndarray          # (N, 2) pitch meters, empty if H failed
    teams: np.ndarray              # (N,) team labels (from classifier) or zeros
    homography_rms: float          # 0 if skipped
    keypoints_seen: list[int]      # landmark IDs detected this frame
    corner_pixels: np.ndarray      # (4, 2) pixel positions of the 4 pitch
                                   # corners under H⁻¹ (for jitter), NaN if
                                   # no H


def collect_spatial_over_time(
    source: str,
    player_model: object,
    keypoint_model: object,
    class_map: dict[str, int],
    player_conf: float,
    ball_conf: float,
    keypoint_conf: float,
    homography_cfg: dict,
    team_classifier: object | None = None,
    stride: int = 15,
    max_frames: int | None = None,
    verbose: bool = True,
) -> tuple[list[SpatialFrameData], float]:
    """Walk the video collecting positions, keypoint sightings, and H jitter.

    Returns (frames, fps).
    """
    cap = cv2.VideoCapture(source)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ids = list(range(0, max(n_frames, 1), stride))
    if max_frames is not None:
        ids = ids[:max_frames]
    if verbose:
        print(f"collect_spatial_over_time: {len(ids)} frames (stride={stride})")

    # The four pitch corners in pitch coords (true landmarks 0, 5, 24, 29).
    corners_pitch = np.array([
        KEYPOINT_VERTICES_M[0], KEYPOINT_VERTICES_M[5],
        KEYPOINT_VERTICES_M[24], KEYPOINT_VERTICES_M[29],
    ])

    frames: list[SpatialFrameData] = []
    for i, fid in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        players, _ = detect_players_and_ball(
            frame, player_model,
            class_map=class_map, player_conf=player_conf, ball_conf=ball_conf,
        )
        kps = detect_keypoints(frame, keypoint_model, min_conf=keypoint_conf)
        H, rms = compute_homography(
            kps,
            min_keypoints=homography_cfg["min_keypoints"],
            min_confidence=homography_cfg.get("min_keypoint_conf", 0.0),
        )
        rms_reject = homography_cfg.get("rms_reproj_threshold", 2.0)
        h_ok = H is not None and rms <= rms_reject

        positions = np.empty((0, 2))
        teams = np.zeros(0, dtype=int)
        corner_px = np.full((4, 2), np.nan)
        if h_ok:
            feet = np.array([
                [(d.xyxy[0] + d.xyxy[2]) / 2, d.xyxy[3]] for d in players
            ]) if players else np.empty((0, 2))
            if len(feet):
                positions = project_points(feet, H)
                if team_classifier is not None and players:
                    from detection_pipeline.teams_pipeline import (
                        assign_teams, crop_detections,
                    )
                    crops = crop_detections(frame, players)
                    assignment = assign_teams(crops, players, positions, team_classifier)
                    teams = assignment.labels
            # Corner pixels via inverse H (pitch → pixel).
            H_inv = np.linalg.inv(H)
            corner_px = project_points(corners_pitch, H_inv)

        frames.append(SpatialFrameData(
            frame_id=fid,
            time_s=fid / fps,
            positions=positions,
            teams=teams,
            homography_rms=float(rms) if h_ok else 0.0,
            keypoints_seen=sorted(kps.points.keys()),
            corner_pixels=corner_px,
        ))
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)} frames processed...")
    cap.release()
    return frames, fps


def position_heatmaps(
    frames: list[SpatialFrameData],
    spec: PitchSpec,
    bin_m: float = 2.0,
) -> dict[str, np.ndarray]:
    """2D histograms of player positions, split by team and H-quality.

    Returns {"team0", "team1", "good_h", "bad_h"} → (ny, nx) arrays in meters.
    """
    x_edges = np.arange(-spec.half_length, spec.half_length + bin_m, bin_m)
    y_edges = np.arange(-spec.half_width, spec.half_width + bin_m, bin_m)

    def _acc(sel: list[np.ndarray]) -> np.ndarray:
        pts = np.vstack(sel) if sel else np.empty((0, 2))
        H2d, _, _ = np.histogram2d(pts[:, 1] if len(pts) else [],
                                   pts[:, 0] if len(pts) else [],
                                   bins=[y_edges, x_edges])
        return H2d

    ok_frames = [f for f in frames if len(f.positions)]
    rms_vals = np.array([f.homography_rms for f in ok_frames])
    median_rms = float(np.median(rms_vals)) if len(rms_vals) else 0.0

    t0, t1, good, bad = [], [], [], []
    for f in ok_frames:
        if f.homography_rms <= median_rms:
            good.append(f.positions)
        else:
            bad.append(f.positions)
        if len(f.teams) == len(f.positions):
            t0.append(f.positions[f.teams == 0])
            t1.append(f.positions[f.teams == 1])
    return {
        "team0": _acc(t0), "team1": _acc(t1),
        "good_h": _acc(good), "bad_h": _acc(bad),
    }


def keypoint_coverage(frames: list[SpatialFrameData]) -> dict[int, float]:
    """Detection frequency per landmark ID across all sampled frames."""
    counts: dict[int, int] = {}
    total = 0
    for f in frames:
        total += 1
        for kp in f.keypoints_seen:
            counts[kp] = counts.get(kp, 0) + 1
    return {kp: counts.get(kp, 0) / max(total, 1) for kp in KEYPOINT_VERTICES_M}


def homography_jitter(frames: list[SpatialFrameData]) -> np.ndarray:
    """Mean pixel displacement of the 4 pitch corners vs previous frame.

    (N,) array, NaN where either frame lacks H. First frame = NaN.
    """
    n = len(frames)
    jitter = np.full(n, np.nan)
    for i in range(1, n):
        a, b = frames[i - 1].corner_pixels, frames[i].corner_pixels
        if not (np.isnan(a).any() or np.isnan(b).any()):
            jitter[i] = float(np.linalg.norm(b - a, axis=1).mean())
    return jitter
