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


class ByteTrackTracker:
    """ByteTrack on pixel-space detections (supervision's sv.ByteTrack).

    WHY PIXEL SPACE, not pitch space: the tracker matches detections by box
    IoU and motion in the image — the domain the detector's noise lives in.
    Projecting to pitch first would entangle homography error with tracking
    error. So: track pixels → project tracked feet to pitch → velocities by
    finite difference on the PITCH positions per track ID.

    ByteTrack's edge over SORT: it keeps low-confidence detections in a
    second association round instead of dropping them — fewer lost tracks
    when a player is briefly occluded (exactly our camera-cut dips from the
    temporal analysis).

    Reference:
        Zhang, Y. et al. (2022). "ByteTrack: Multi-Object Tracking by
        Associating Every Detection Box." ECCV.
    """

    def __init__(self, fps: float = 25.0,
                 track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 minimum_consecutive_frames: int = 3):
        import supervision as sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        self._dt = 1.0 / fps
        # Per-track pitch position history for finite-difference velocities:
        # track_id → (prev_xy_m, prev_vxy_ms)
        self._prev: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def update(self, detections: list[Detection], H: np.ndarray) -> list[TrackedPlayer]:
        import supervision as sv
        from .transform import project_points

        if detections:
            xyxy = np.array([d.xyxy for d in detections])
            conf = np.array([d.confidence for d in detections])
            cls = np.array([d.class_id for d in detections])
            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        else:
            sv_dets = sv.Detections.empty()

        tracked_dets = self._tracker.update_with_detections(sv_dets)

        tracked: list[TrackedPlayer] = []
        if tracked_dets.tracker_id is None or len(tracked_dets) == 0:
            return tracked

        # Project feet to pitch for all tracked boxes at once.
        feet = np.array([
            [(b[0] + b[2]) / 2, b[3]] for b in tracked_dets.xyxy
        ])
        feet_m = project_points(feet, H)

        # Map tracker output back to our Detection objects for class names.
        # sv preserves order of confirmed tracks; match by IoU on boxes.
        det_boxes = np.array([d.xyxy for d in detections]) if detections else np.empty((0, 4))

        for i, (tid, box) in enumerate(zip(tracked_dets.tracker_id, tracked_dets.xyxy)):
            tid = int(tid)
            xy_m = feet_m[i]

            # Finite-difference velocity with light smoothing
            # v = α·v_prev + (1−α)·(Δx/Δt); α=0.4 damps projection jitter
            # while staying responsive (humans accelerate ~3 m/s²).
            prev = self._prev.get(tid)
            if prev is None:
                v = np.zeros(2)
            else:
                v_raw = (xy_m - prev[0]) / self._dt
                v = 0.4 * prev[1] + 0.6 * v_raw
            self._prev[tid] = (xy_m, v)

            # Class name: best-IoU match among current detections.
            cls_name = "player"
            conf = 0.0
            if len(det_boxes):
                ious = _iou_batch(box, det_boxes)
                j = int(np.argmax(ious))
                if ious[j] > 0.1:
                    cls_name = detections[j].class_name
                    conf = detections[j].confidence

            tracked.append(TrackedPlayer(
                track_id=tid,
                pitch_xy=xy_m,
                pitch_vxy=v,
                class_name=cls_name,
                confidence=conf,
            ))
        return tracked


def _iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box [x1,y1,x2,y2] against N boxes. (N,)"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return inter / np.maximum(union, 1e-9)


def create_tracker(tracker_type: str, **kwargs):
    """Factory: create a tracker by name.

    Supported: "single_frame", "bytetrack", "sort", "oc_sort" (future).
    """
    if tracker_type == "single_frame":
        return SingleFrameTracker()
    elif tracker_type == "bytetrack":
        return ByteTrackTracker(**kwargs)
    elif tracker_type == "sort":
        return SORTTracker(**kwargs)
    elif tracker_type == "oc_sort":
        raise NotImplementedError(
            "OC-SORT not yet wired. Requires: uv add trackers, then map "
            "trackers.OCSORTTracker into this factory."
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type!r}")
