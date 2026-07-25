"""RadarFrame dataclass and JSONL writer.

The JSONL schema is the CONTRACT between this pipeline and footlab.
One line per frame, self-contained, loadable by ``footlab.io_radar``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RadarPlayer:
    """One player in pitch coordinates."""

    track_id: int
    class_name: str           # "player", "goalkeeper", "referee"
    team: str | None          # None until team classification is built
    pitch_xy_m: list[float]   # [x, y] meters
    pitch_vxy_ms: list[float] # [vx, vy] m/s


@dataclass
class RadarFrame:
    """One processed frame — the unit of output.

    Serializes to a single JSONL line. Every field footlab's
    ``io_radar.load_radar_jsonl()`` needs is here.
    """

    frame_id: int
    source: str                          # video filename or path
    pitch: dict                          # {"length_m": 105.0, "width_m": 68.0, ...}
    homography: list[list[float]] | None # 3×3 matrix, or None if skipped
    keypoints_used: list[int]            # keypoint IDs that fed the homography
    players: list[RadarPlayer]
    ball: dict | None                    # {"pitch_xy_m": [x, y]} or None
    carrier_track_id: int | None         # track_id of ball carrier, or None

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSON line."""
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_jsonl_line(cls, line: str) -> "RadarFrame":
        """Deserialize from a single JSON line."""
        d = json.loads(line)
        d["players"] = [RadarPlayer(**p) for p in d["players"]]
        return cls(**d)


def write_jsonl(frames: list[RadarFrame], path: str | Path) -> None:
    """Write a list of RadarFrames to a JSONL file (one per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame in frames:
            f.write(frame.to_jsonl_line() + "\n")


def read_jsonl(path: str | Path) -> list[RadarFrame]:
    """Read a JSONL file back into RadarFrames."""
    path = Path(path)
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(RadarFrame.from_jsonl_line(line))
    return frames
