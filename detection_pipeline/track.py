"""Player tracking in pitch coordinates.

Wraps SORT (and later OC-SORT via roboflow ``trackers``) to maintain persistent
track IDs across frames. The tracker operates on pitch-plane coordinates
(meters), not pixel coordinates, so track identity survives camera movement.

For RADAR_SINGLE (single frame), tracking is trivial — each detection gets a
unique ID. For RADAR_VIDEO, the tracker maintains temporal consistency.

The tracker is swappable via ``config.PIPELINE["tracker"]["type"]``.

Reference:
    Bewley, A. et al. (2016). "Simple Online and Realtime Tracking." IEEE ICIP.
    — SORT: Kalman filter + Hungarian assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .infer import Detection


@dataclass
class TrackedPlayer:
    """A player with a persistent track ID and pitch-plane position."""

    track_id: int
    pitch_xy: np.ndarray   # (2,) meters
    pitch_vxy: np.ndarray  # (2,) m/s (zero for single-frame)
    class_name: str        # "player", "goalkeeper", "referee"
    confidence: float


class SingleFrameTracker:
    """Trivial tracker for single-frame mode — assigns sequential IDs.

    No temporal state; each detection gets a fresh track ID. This is the
    ground-basis test: prove the full pipeline works for one frame before
    worrying about track persistence.
    """

    def __init__(self) -> None:
        self._next_id = 0

    def update(self, detections: list[Detection], H: np.ndarray) -> list[TrackedPlayer]:
        from .transform import project_points

        tracked = []
        for det in detections:
            foot_pix = det.foot_point.reshape(1, 2)
            foot_m = project_points(foot_pix, H)[0]
            tracked.append(TrackedPlayer(
                track_id=self._next_id,
                pitch_xy=foot_m,
                pitch_vxy=np.zeros(2),
                class_name=det.class_name,
                confidence=det.confidence,
            ))
            self._next_id += 1
        return tracked


class SORTTracker:
    """SORT tracker operating in pitch coordinates.

    Uses the roboflow ``trackers`` library's SORT implementation, but feeds
    it pitch-plane foot-point positions instead of pixel bounding boxes.
    This makes track identity robust to camera pan/zoom.

    For RADAR_VIDEO only — not exercised in RADAR_SINGLE mode.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        try:
            from trackers import SORTTracker as _SORT
        except ImportError:
            raise ImportError(
                "roboflow 'trackers' package not installed. "
                "Run: uv add trackers"
            )
        self._tracker = _SORT(
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )

    def update(self, detections: list[Detection], H: np.ndarray) -> list[TrackedPlayer]:
        from .transform import project_points

        if not detections:
            return []

        # Project foot points to pitch space
        foot_points_pix = np.array([d.foot_point for d in detections])
        foot_points_m = project_points(foot_points_pix, H)

        # Build a detection array for the tracker: [x, y, confidence]
        det_array = np.column_stack([foot_points_m, [d.confidence for d in detections]])
        tracked_dets = self._tracker.update(det_array)

        tracked = []
        for i, td in enumerate(tracked_dets):
            tracked.append(TrackedPlayer(
                track_id=int(td[3]) if len(td) > 3 else i,  # tracker-assigned ID
                pitch_xy=np.array([td[0], td[1]]),
                pitch_vxy=np.zeros(2),  # TODO: finite-diff from previous frame
                class_name=detections[i].class_name if i < len(detections) else "player",
                confidence=detections[i].confidence if i < len(detections) else 0.0,
            ))
        return tracked


def create_tracker(tracker_type: str, **kwargs):
    """Factory: create a tracker by name.

    Supported: "single_frame", "sort", "oc_sort" (future).
    """
    if tracker_type == "single_frame":
        return SingleFrameTracker()
    elif tracker_type == "sort":
        return SORTTracker(**kwargs)
    elif tracker_type == "oc_sort":
        raise NotImplementedError(
            "OC-SORT not yet wired. Requires: uv add trackers, then map "
            "trackers.OCSORTTracker into this factory."
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type!r}")
