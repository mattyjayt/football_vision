"""Team fitting & assignment across a video — the pipeline-facing layer.

teams.py holds the *algorithms* (SigLIP/UMAP/KMeans, centroid heuristic).
This module holds the *workflow*:

    1. fit_teams_from_video()  — sample frames across the video with a stride,
       detect players in each, crop them, and fit ONE TeamClassifier on the
       pooled crops. More crops → cleaner clusters; a single frame (~25
       crops) works but a few hundred crops is where misclassifications die.

    2. assign_teams()          — given one frame's detections + their crops,
       predict a team label per outfield player and resolve goalkeepers by
       the nearest-centroid heuristic (teams.py).

    3. resolve_attack_side()   — KMeans labels are ARBITRARY: "team 0" says
       nothing about which side that team attacks. footlab's contract is
       attack → +x. We resolve it from the goalkeeper pitch positions: the
       team whose GK stands at negative x defends the LEFT goal and
       therefore attacks +x. Fallback when no GK is visible: the team's
       defensive-third centroid (deepest x of its players).

Why crops come back with detections: the classifier must see the SAME pixels
the detector saw — re-cropping inside assign keeps the two in lockstep.
"""

from __future__ import annotations

import os

# See scripts/test_teams_live.py — numba/UMAP parallelism segfaults on macOS
# once torch has loaded its OpenMP runtime. Must be set before numeric libs.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from dataclasses import dataclass

import cv2
import numpy as np

from .infer import Detection, detect_players_and_ball
from .teams import TeamClassifier, resolve_goalkeepers_team_id


@dataclass
class TeamAssignment:
    """Team labels for one frame's detections, in detection order."""

    labels: np.ndarray          # (N,) int: 0 or 1 per detection
    attacking_team: int | None  # which label attacks +x, if resolvable


def crop_detections(frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
    """Cut each detection's bounding box out of the frame (BGR crops)."""
    crops = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        crops.append(crop if crop.size else np.zeros((2, 2, 3), dtype=np.uint8))
    return crops


def fit_teams_from_video(
    source: str,
    player_model: object,
    class_map: dict[str, int],
    player_conf: float,
    ball_conf: float,
    stride: int = 60,
    max_frames: int = 25,
    device: str = "cpu",
    verbose: bool = True,
) -> TeamClassifier:
    """Sample frames across the video and fit one TeamClassifier.

    Args:
        source:       video path.
        player_model: loaded detection model (local ultralytics or roboflow).
        class_map:    detector class mapping from config.
        player_conf / ball_conf: detection thresholds from config.
        stride:       process every Nth frame (60 ≈ every 2 s at 30 fps).
        max_frames:   cap on sampled frames (bounds fit time).
        device:       SigLIP device ("cpu" is fine; ~1–2 s per batch).

    Returns:
        A fitted TeamClassifier ready for assign_teams().
    """
    cap = cv2.VideoCapture(source)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = list(range(0, max(n_frames, 1), stride))[:max_frames]

    crops: list[np.ndarray] = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        players, _ = detect_players_and_ball(
            frame, player_model,
            class_map=class_map, player_conf=player_conf, ball_conf=ball_conf,
        )
        outfield = [d for d in players if d.class_name == "player"]
        crops.extend(crop_detections(frame, outfield))
    cap.release()

    if verbose:
        print(f"fit_teams_from_video: {len(crops)} outfield crops "
              f"from {len(frame_ids)} sampled frames")

    if len(crops) < 8:
        raise RuntimeError(
            f"Only {len(crops)} crops collected — not enough to cluster two "
            f"teams. Check the detector and thresholds."
        )

    clf = TeamClassifier(device=device)
    clf.fit(crops)
    if verbose:
        sizes = {k: int((clf.cluster_model.labels_ == k).sum())
                 for k in set(clf.cluster_model.labels_.tolist())}
        print(f"  fitted on {len(crops)} crops — cluster sizes: {sizes}")
    return clf


def assign_teams(
    crops: list[np.ndarray],
    detections: list[Detection],
    pitch_xy: np.ndarray,
    clf: TeamClassifier,
) -> TeamAssignment:
    """Predict team labels for one frame.

    Args:
        crops:      one crop per detection (same order).
        detections: the frame's detections.
        pitch_xy:   (N, 2) pitch positions of the detections (meters) — used
                    for the goalkeeper centroid heuristic and attack-side
                    resolution.
        clf:        a fitted TeamClassifier.

    Returns:
        TeamAssignment with a label per detection and the attacking side.
    """
    n = len(detections)
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return TeamAssignment(labels=labels, attacking_team=None)

    is_player = np.array([d.class_name == "player" for d in detections])
    is_gk = np.array([d.class_name == "goalkeeper" for d in detections])

    if is_player.any():
        player_crops = [c for c, p in zip(crops, is_player) if p]
        player_labels = clf.predict(player_crops)
        labels[is_player] = player_labels

        if is_gk.any():
            gk_labels = resolve_goalkeepers_team_id(
                players_xy=pitch_xy[is_player],
                players_team_id=player_labels,
                goalkeepers_xy=pitch_xy[is_gk],
            )
            labels[is_gk] = gk_labels

    # Referees: no team. Mark with -1 so downstream can exclude them.
    is_ref = np.array([d.class_name == "referee" for d in detections])
    labels[is_ref] = -1

    attacking = resolve_attack_side(labels, pitch_xy, detections)
    return TeamAssignment(labels=labels, attacking_team=attacking)


def resolve_attack_side(
    labels: np.ndarray,
    pitch_xy: np.ndarray,
    detections: list[Detection],
) -> int | None:
    """Which cluster attacks +x? Resolved from the goalkeeper's x position.

    A GK stands in front of the goal he DEFENDS. If team 0's GK is at
    x < 0 (left goal in our center-origin frame), team 0 defends left and
    attacks +x (right). Fallback: each team's defensive-third centroid.
    """
    gk_x = {lab: [] for lab in (0, 1)}
    for lab, xy, det in zip(labels, pitch_xy, detections):
        if det.class_name == "goalkeeper" and lab in (0, 1):
            gk_x[lab].append(xy[0])

    for lab in (0, 1):
        if gk_x[lab]:
            return lab if np.mean(gk_x[lab]) < 0 else 1 - lab

    # Fallback: deepest (most negative) mean x among each team's players —
    # that side of the pitch is where their defense lives.
    team_min_x: dict[int, float] = {}
    for lab in (0, 1):
        xs = pitch_xy[(labels == lab), 0]
        if len(xs):
            team_min_x[lab] = float(np.percentile(xs, 10))  # defensive depth
    if len(team_min_x) == 2:
        # The team whose defensive line is on the left attacks right.
        left_team = min(team_min_x, key=team_min_x.get)
        return left_team
    return None
