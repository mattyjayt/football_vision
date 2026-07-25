"""Detection pipeline — video frames → radar-view JSONL for footlab.

A feeder system, NOT part of footlab's core. Takes raw broadcast video,
runs pretrained perception models (Roboflow-hosted for now, our own later),
projects player positions onto the canonical 105×68 m pitch, and writes
one JSONL line per frame that ``footlab.io_radar`` loads into a FrozenFrame.

Design principles:
    * Modular — every component (detector, tracker, transformer, writer)
      is swappable via ``config.py``. Change a string, not the code.
    * Single responsibility — this package produces data. footlab consumes it.
    * No heavy deps beyond what footlab already has (supervision, ultralytics).
      No ``sports`` package; the homography transform is ~25 lines of cv2.

Modes:
    RADAR_SINGLE — process ONE frame end-to-end (ground-truth stitch test).
    RADAR_VIDEO  — process every frame (stubbed; deferred until RADAR_SINGLE
                   is verified on real data).
"""

from .config import PIPELINE, KEYPOINT_VERTICES_M, PitchSpec
from .frames import RadarFrame, write_jsonl

__all__ = [
    "PIPELINE",
    "KEYPOINT_VERTICES_M",
    "PitchSpec",
    "RadarFrame",
    "write_jsonl",
]
