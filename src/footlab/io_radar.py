"""Load radar-view JSONL (from detection_pipeline) into FrozenFrame.

This adapter mirrors ``skillcorner.py`` in structure: it reads the JSONL
produced by ``detection_pipeline`` and produces a ``FrozenFrame`` that every
footlab phase can consume unchanged.

The JSONL schema is the contract (see ``detection_pipeline/frames.py``):
one line per frame with players, ball, homography, and carrier in
pitch-plane coordinates (105×68 m, origin at center, attack → +x).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .state import FrozenFrame, TEAM_ATTACK, TEAM_DEFEND


def load_radar_jsonl(
    path: str | Path,
    frame_id: int | None = None,
) -> FrozenFrame:
    """Load one frame from a radar JSONL file into a FrozenFrame.

    Args:
        path: path to the .jsonl file written by detection_pipeline.
        frame_id: which frame to load. If the file has only one line
                  (RADAR_SINGLE output), this can be None.

    Returns:
        FrozenFrame ready for any footlab phase.

    Raises:
        ValueError: if the frame has no homography (projection was skipped).
    """
    path = Path(path)
    lines = path.read_text().strip().split("\n")

    target_line = None
    for line in lines:
        d = json.loads(line)
        if frame_id is None or d["frame_id"] == frame_id:
            target_line = d
            break

    if target_line is None:
        raise ValueError(f"Frame {frame_id} not found in {path}")

    if target_line["homography"] is None:
        raise ValueError(
            f"Frame {target_line['frame_id']} has no homography — "
            f"the detection pipeline skipped projection for this frame "
            f"(insufficient keypoints). Cannot build a FrozenFrame."
        )

    players = target_line["players"]
    if not players:
        raise ValueError(f"Frame {target_line['frame_id']} has no players")

    positions = np.array([p["pitch_xy_m"] for p in players], dtype=float)
    velocities = np.array([p["pitch_vxy_ms"] for p in players], dtype=float)

    # Team assignment: deferred. For now, put all players on TEAM_ATTACK
    # (the planner will see them all as attackers). This is honest — we
    # don't know teams yet. Team classification via SigLIP comes later.
    team_ids = np.full(len(players), TEAM_ATTACK, dtype=int)

    # Player IDs = track IDs from the pipeline
    player_ids = np.array([p["track_id"] for p in players])

    # Ball
    if target_line["ball"] is not None:
        ball_pos = np.array(target_line["ball"]["pitch_xy_m"], dtype=float)
    else:
        ball_pos = np.array([0.0, 0.0])  # center spot fallback

    # Carrier: map track_id back to array index
    carrier_idx = -1
    if target_line["carrier_track_id"] is not None:
        carrier_tid = target_line["carrier_track_id"]
        matches = np.where(player_ids == carrier_tid)[0]
        if len(matches) > 0:
            carrier_idx = int(matches[0])

    return FrozenFrame(
        positions=positions,
        velocities=velocities,
        team_ids=team_ids,
        ball_pos=ball_pos,
        ball_carrier=carrier_idx,
        player_ids=player_ids,
    )
