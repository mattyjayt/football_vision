"""Inference wrappers — call Roboflow-hosted models, return structured detections.

Thin layer over the Roboflow Inference HTTP API. Each function takes a video
frame (numpy array) and returns a plain Python dict / list of detections,
decoupled from the Roboflow response format so we can swap in our own models
later without touching downstream code.

Requires ``ROBOFLOW_API_KEY`` in the environment.

Reference:
    Roboflow (2024). Inference API. https://inference.roboflow.com/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

try:
    from inference import get_model
except ImportError:
    get_model = None  # type: ignore[assignment]


@dataclass
class Detection:
    """One detected object in pixel space."""

    xyxy: np.ndarray        # (4,) [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> np.ndarray:
        """(2,) center point of the bounding box."""
        return np.array([(self.xyxy[0] + self.xyxy[2]) / 2,
                         (self.xyxy[1] + self.xyxy[3]) / 2])

    @property
    def foot_point(self) -> np.ndarray:
        """(2,) bottom-center of the bounding box — ground contact point."""
        return np.array([(self.xyxy[0] + self.xyxy[2]) / 2,
                         self.xyxy[3]])


@dataclass
class KeypointSet:
    """Detected pitch keypoints in pixel space."""

    points: dict[int, np.ndarray]  # keypoint_id -> (2,) pixel coordinates
    confidences: dict[int, float]  # keypoint_id -> confidence

    def filter_confident(self, min_conf: float) -> "KeypointSet":
        """Return a new KeypointSet with only points above threshold."""
        pts = {k: v for k, v in self.points.items()
               if self.confidences.get(k, 0.0) >= min_conf}
        cfs = {k: self.confidences[k] for k in pts}
        return KeypointSet(points=pts, confidences=cfs)

    @property
    def n_keypoints(self) -> int:
        return len(self.points)


def _check_inference_available() -> None:
    if get_model is None:
        raise ImportError(
            "roboflow 'inference' package not installed. "
            "Run: uv add inference"
        )
    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise EnvironmentError(
            "ROBOFLOW_API_KEY not set. "
            "Export it or add it to your environment before running inference."
        )


def load_player_model(model_id: str) -> object:
    """Load the player+ball detection model (cached by roboflow inference)."""
    _check_inference_available()
    return get_model(model_id=model_id)


def load_keypoint_model(model_id: str) -> object:
    """Load the pitch keypoint detection model."""
    _check_inference_available()
    return get_model(model_id=model_id)


def detect_players_and_ball(
    frame: np.ndarray,
    model: object,
    class_map: dict[str, int],
    player_conf: float = 0.35,
    ball_conf: float = 0.25,
) -> tuple[list[Detection], Detection | None]:
    """Run player+ball detection on one frame.

    Returns:
        players: list of Detection objects for players, goalkeepers, referees.
        ball:    single Detection for the ball, or None if not found.
    """
    results = model.infer(frame)[0]  # type: ignore[attr-defined]

    players: list[Detection] = []
    ball: Detection | None = None

    player_classes = {class_map["player"], class_map["goalkeeper"], class_map["referee"]}
    ball_class = class_map["ball"]

    for pred in results.predictions:  # type: ignore[attr-defined]
        cls_id = int(pred.class_id)  # type: ignore[attr-defined]
        conf = float(pred.confidence)  # type: ignore[attr-defined]
        xyxy = np.array([pred.x - pred.width / 2,   # type: ignore[attr-defined]
                         pred.y - pred.height / 2,  # type: ignore[attr-defined]
                         pred.x + pred.width / 2,   # type: ignore[attr-defined]
                         pred.y + pred.height / 2])  # type: ignore[attr-defined]

        if cls_id in player_classes and conf >= player_conf:
            cls_name = {v: k for k, v in class_map.items()}.get(cls_id, "unknown")
            players.append(Detection(xyxy=xyxy, confidence=conf,
                                     class_id=cls_id, class_name=cls_name))
        elif cls_id == ball_class and conf >= ball_conf:
            det = Detection(xyxy=xyxy, confidence=conf,
                            class_id=cls_id, class_name="ball")
            if ball is None or det.confidence > ball.confidence:
                ball = det

    return players, ball


def detect_keypoints(
    frame: np.ndarray,
    model: object,
    min_conf: float = 0.30,
) -> KeypointSet:
    """Run pitch keypoint detection on one frame.

    Returns:
        KeypointSet with pixel-space keypoint positions and confidences.
    """
    results = model.infer(frame)[0]  # type: ignore[attr-defined]

    points: dict[int, np.ndarray] = {}
    confidences: dict[int, float] = {}

    for pred in results.predictions:  # type: ignore[attr-defined]
        # Roboflow keypoint models return keypoints in pred.keypoints
        if hasattr(pred, "keypoints") and pred.keypoints is not None:  # type: ignore[attr-defined]
            for kp_idx, kp in enumerate(pred.keypoints):  # type: ignore[attr-defined]
                conf = float(kp.confidence)
                if conf >= min_conf:
                    points[kp_idx] = np.array([kp.x, kp.y])
                    confidences[kp_idx] = conf

    return KeypointSet(points=points, confidences=confidences)
