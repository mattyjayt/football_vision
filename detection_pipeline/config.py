"""Pipeline configuration — every tunable in one place.

This is the ONLY file you should need to edit when swapping components
(models, trackers, thresholds). No magic values buried in code.

The keypoint vertex map translates the Roboflow ``football-field-detection``
model's numeric keypoint IDs into (x, y) meter coordinates on OUR canonical
pitch (105×68 m, origin at center spot, attack toward +x). The Roboflow model
returns ~30 numbered keypoints; we map the 8 most stable ones (corners, box
corners, center circle top/bottom) which are almost always visible in
broadcast footage.

Reference:
    Roboflow (2024). "football-field-detection" model (football-field-detection-f07vi).
    Keypoint IDs follow the model's internal ordering. See:
    https://github.com/roboflow/sports/tree/main/examples/soccer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Pitch specification (mirrors footlab.pitch constants)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PitchSpec:
    """Canonical pitch dimensions in meters. Matches footlab.pitch exactly."""

    length_m: float = 105.0
    width_m: float = 68.0

    @property
    def half_length(self) -> float:
        return self.length_m / 2.0

    @property
    def half_width(self) -> float:
        return self.width_m / 2.0

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) for matplotlib imshow extent."""
        return (-self.half_length, self.half_length,
                -self.half_width, self.half_width)


# ---------------------------------------------------------------------------
# Keypoint → pitch-coordinate mapping
# Keypoint → pitch-coordinate mapping
#
# The Roboflow model uses 0-indexed keypoints (0–31) corresponding to the
# SoccerPitchConfiguration vertices. We map ALL 32 keypoints and let the
# RANSAC homography solver handle outliers.
#
# Their coordinate system (from sports/configs/soccer.py):
#   - x: 0 (left goal line) → 12000 (right goal line) [cm]
#   - y: 0 (top touchline) → 7000 (bottom touchline) [cm]
#
# Our coordinate system:
#   - x: -52.5 (defending goal) → +52.5 (attacking goal) [m]
#   - y: -34.0 (bottom touchline) → +34.0 (top touchline) [m]
#
# Translation: x_ours = (x_theirs_cm / 100) - 52.5
#              y_ours = (y_theirs_cm / 100) - 34.0

KEYPOINT_VERTICES_M: dict[int, tuple[float, float]] = {
    # Left goal line (defending side) — vertices 0–5
    0: (-52.5, -34.0),      # bottom-left corner
    1: (-52.5, -20.16),     # penalty box bottom-left
    2: (-52.5, -9.16),      # goal box bottom-left
    3: (-52.5, +9.16),      # goal box top-left
    4: (-52.5, +20.16),     # penalty box top-left
    5: (-52.5, +34.0),      # top-left corner

    # Left penalty box — vertices 6–12
    6: (-46.95, -9.16),     # goal box bottom-right
    7: (-46.95, +9.16),     # goal box top-right
    8: (-41.5, 0.0),        # penalty spot
    9: (-36.0, -20.16),     # penalty box bottom-right
    10: (-36.0, -9.16),     # penalty box bottom-left (inner)
    11: (-36.0, +9.16),     # penalty box top-left (inner)
    12: (-36.0, +20.16),    # penalty box top-right

    # Halfway line — vertices 13–16
    13: (0.0, -34.0),       # halfway line bottom
    14: (0.0, -9.15),       # center circle bottom
    15: (0.0, +9.15),       # center circle top
    16: (0.0, +34.0),       # halfway line top

    # Right penalty box (attacking side) — vertices 17–23
    17: (+36.0, -20.16),    # penalty box bottom-left
    18: (+36.0, -9.16),     # goal box bottom-left (inner)
    19: (+36.0, +9.16),     # goal box top-left (inner)
    20: (+36.0, +20.16),    # penalty box top-left
    21: (+41.5, 0.0),       # penalty spot
    22: (+46.95, -9.16),    # goal box bottom-right
    23: (+46.95, +9.16),    # goal box top-right

    # Right goal line — vertices 24–29
    24: (+52.5, -34.0),     # bottom-right corner
    25: (+52.5, -20.16),    # penalty box bottom-right
    26: (+52.5, -9.16),     # goal box bottom-right
    27: (+52.5, +9.16),     # goal box top-right
    28: (+52.5, +20.16),    # penalty box top-right
    29: (+52.5, +34.0),     # top-right corner

    # Center circle (left and right edges) — vertices 30–31
    30: (-9.15, 0.0),       # center circle left
    31: (+9.15, 0.0),       # center circle right
}


# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------

PIPELINE: dict[str, Any] = {
    # --- Models ---
    "detector": {
        "type": "local",  # "roboflow" or "local"
        # Local model paths (used when type="local")
        "player_model_path": "models/player_detector.pt",
        # Roboflow model IDs (used when type="roboflow")
        "player_model_id": "football-players-detection-3zvbc/9",
        "player_conf": 0.35,
        "ball_conf": 0.25,
        # Pitch keypoint detection (RF-DETR X-Large, 100% mAP@50)
        # Local Roboflow-exported weights (32 keypoints, matches KEYPOINT_VERTICES_M)
        "keypoint_model_type": "local",
        "keypoint_model_path": "models/football-pitch-detection.pt",
        "keypoint_model_id": "football-field-detection-f07vi/17",
        "keypoint_conf": 0.0,
        # Which Roboflow class IDs correspond to what (model-specific)
        "class_map": {
            "player": 2,
            "goalkeeper": 1,
            "referee": 3,
            "ball": 0,
        },
    },
    # --- Tracker ---
    "tracker": {
        # "sort" or "oc_sort" (via roboflow trackers lib) — swap by string.
        "type": "sort",
        "iou_threshold": 0.30,
        "max_age": 30,       # frames to keep a lost track alive
        "min_hits": 3,       # detections before a track is confirmed
    },
    # --- Homography ---
    "homography": {
        "min_keypoints": 6,          # minimum to compute a valid homography
        "rms_reproj_threshold": 2.0, # meters; if RMS reproj error > this, reject
        "min_keypoint_conf": 0.3,    # balance: drop garbage low-conf points, keep coverage
    },
    # --- Carrier assignment ---
    "carrier": {
        "max_distance_m": 1.5,  # player must be within this of ball to be carrier
    },
    # --- Output ---
    "output": {
        "pitch_length_m": 105.0,
        "pitch_width_m": 68.0,
        "attack_direction": "+x",   # single-frame assumption
    },
}
