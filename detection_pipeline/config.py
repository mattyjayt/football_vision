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
# ---------------------------------------------------------------------------

# Keypoint IDs used by the Roboflow football-field-detection model.
# These are the most stable points: the four corners of the pitch, the
# penalty-area corners, and the center circle's top/bottom. In practice the
# model detects most of these in typical broadcast framing.
KEYPOINT_VERTICES_M: dict[int, tuple[float, float]] = {
    # Pitch corners
    0: (-52.5, -34.0),   # bottom-left corner (from attacker's view)
    1: (+52.5, -34.0),   # bottom-right corner
    2: (+52.5, +34.0),   # top-right corner
    3: (-52.5, +34.0),   # top-left corner
    # Penalty area corners (left / defending side)
    4: (-52.5 + 16.5, -20.16),   # left penalty box bottom-right
    5: (-52.5 + 16.5, +20.16),   # left penalty box top-right
    6: (-52.5, -20.16),          # left penalty box bottom-left (on goal line)
    7: (-52.5, +20.16),          # left penalty box top-left (on goal line)
    # Penalty area corners (right / attacking side)
    8: (+52.5 - 16.5, -20.16),   # right penalty box bottom-left
    9: (+52.5 - 16.5, +20.16),   # right penalty box top-left
    10: (+52.5, -20.16),         # right penalty box bottom-right (on goal line)
    11: (+52.5, +20.16),         # right penalty box top-right (on goal line)
    # Center circle top and bottom
    12: (0.0, -9.15),   # center circle bottom
    13: (0.0, +9.15),   # center circle top
    # Penalty spots
    14: (-52.5 + 11.0, 0.0),   # left penalty spot
    15: (+52.5 - 11.0, 0.0),   # right penalty spot
}


# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------

PIPELINE: dict[str, Any] = {
    # --- Models (Roboflow Inference API, pretrained) ---
    "detector": {
        "type": "roboflow",
        # Player + ball detection (one model, multi-class)
        "player_model_id": "football-players-detection-3zvbc/9",
        "player_conf": 0.35,
        "ball_conf": 0.25,
        # Pitch keypoint detection
        "keypoint_model_id": "football-field-detection-f07vi/14",
        "keypoint_conf": 0.30,
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
        "min_keypoints": 4,          # minimum to compute a valid homography
        "rms_reproj_threshold": 1.5, # meters; if RMS reproj error > this, reject
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
