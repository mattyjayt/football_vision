"""Tests for ByteTrack wrapper + Phase 4 tracking metrics.

sv.ByteTrack runs for real on synthetic detection sequences (it's fast and
dependency-present); identity H is used so pixel feet == pitch meters.
Metric logic is tested on handcrafted JSONL dicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from detection_pipeline.infer import Detection
from detection_pipeline.track import ByteTrackTracker, _iou_batch
from exploration.temporal_tracking import (
    MAX_HUMAN_SPEED_MS, analyze_tracks, tracking_table_markdown,
)

H_IDENTITY = np.eye(3)


def _det(x, y, w=20, h=40, conf=0.9, cls="player", cls_id=2):
    return Detection(xyxy=np.array([x, y, x + w, y + h], dtype=float),
                     confidence=conf, class_id=cls_id, class_name=cls)


class TestIouBatch:
    def test_identical_box_iou_one(self):
        b = np.array([0, 0, 10, 10], dtype=float)
        assert _iou_batch(b, b[None, :])[0] == pytest.approx(1.0)

    def test_disjoint_zero(self):
        b = np.array([0, 0, 10, 10], dtype=float)
        other = np.array([[100, 100, 110, 110]], dtype=float)
        assert _iou_batch(b, other)[0] == pytest.approx(0.0)

    def test_half_overlap(self):
        b = np.array([0, 0, 10, 10], dtype=float)
        other = np.array([[5, 0, 15, 10]], dtype=float)
        iou = _iou_batch(b, other)[0]
        assert 0.3 < iou < 0.4  # 50 inter / 150 union


class TestByteTrackWrapper:
    def test_persistent_ids_for_static_players(self):
        trk = ByteTrackTracker(fps=25, minimum_consecutive_frames=1)
        ids_seen = set()
        for _ in range(5):
            dets = [_det(100, 100), _det(500, 300)]
            tracked = trk.update(dets, H_IDENTITY)
            for tp in tracked:
                ids_seen.add(tp.track_id)
        # Two static players → exactly two track IDs over 5 frames.
        assert len(ids_seen) == 2

    def test_velocity_zero_for_static_player(self):
        trk = ByteTrackTracker(fps=25, minimum_consecutive_frames=1)
        for _ in range(4):
            tracked = trk.update([_det(100, 100)], H_IDENTITY)
        assert np.linalg.norm(tracked[0].pitch_vxy) == pytest.approx(0.0, abs=1e-9)

    def test_velocity_reflects_motion(self):
        """Player moving 10px/frame right at 25fps → 250 units/s on identity H."""
        trk = ByteTrackTracker(fps=25, minimum_consecutive_frames=1)
        v = None
        for f in range(6):
            tracked = trk.update([_det(100 + 10 * f, 100)], H_IDENTITY)
            if tracked:
                v = tracked[0].pitch_vxy
        # Smoothed finite diff should approach (not equal) 250 m/s raw.
        assert v[0] > 50   # clearly positive, smoothing-damped
        assert abs(v[1]) < 1e-6

    def test_class_and_conf_from_detection(self):
        trk = ByteTrackTracker(fps=25, minimum_consecutive_frames=1)
        tracked = trk.update([_det(50, 50, conf=0.77, cls="goalkeeper", cls_id=1)],
                             H_IDENTITY)
        assert tracked[0].class_name == "goalkeeper"
        assert tracked[0].confidence == pytest.approx(0.77)

    def test_empty_frame_returns_empty(self):
        trk = ByteTrackTracker(fps=25)
        assert trk.update([], H_IDENTITY) == []


class TestAnalyzeTracks:
    def _frames(self):
        # 3 frames; track 1 lives all 3, track 2 appears once (noise),
        # track 3 moves implausibly fast.
        def fr(i, players):
            return {"frame_id": i, "players": players}
        return [
            fr(0, [{"track_id": 1, "pitch_vxy_ms": [2.0, 0.0]},
                   {"track_id": 3, "pitch_vxy_ms": [0.0, 0.0]}]),
            fr(1, [{"track_id": 1, "pitch_vxy_ms": [3.0, 0.0]},
                   {"track_id": 3, "pitch_vxy_ms": [15.0, 0.0]}]),
            fr(2, [{"track_id": 1, "pitch_vxy_ms": [2.5, 0.0]},
                   {"track_id": 2, "pitch_vxy_ms": [1.0, 0.0]},
                   {"track_id": 3, "pitch_vxy_ms": [20.0, 0.0]}]),
        ]

    def test_lifetimes_and_noise(self):
        report, lifetimes, speeds = analyze_tracks(self._frames())
        assert lifetimes == {1: 3, 3: 3, 2: 1}
        assert report.single_frame_tracks == 1
        assert report.n_tracks == 3
        assert report.lifetime_median == 3.0

    def test_impossible_speeds_flagged(self):
        report, _, speeds = analyze_tracks(self._frames())
        # 2 of 7 player-frames exceed 10.5 m/s
        assert report.pct_impossible_speeds == pytest.approx(2 / 7 * 100, rel=1e-3)
        assert report.p95_speed_ms > MAX_HUMAN_SPEED_MS

    def test_new_ids_per_frame(self):
        report, _, _ = analyze_tracks(self._frames())
        # frame1: 0 new, frame2: 1 new → mean 0.5
        assert report.new_ids_per_frame == pytest.approx(0.5)

    def test_table(self):
        report, _, _ = analyze_tracks(self._frames())
        md = tracking_table_markdown([("vid.mp4", report)])
        assert "| vid.mp4 |" in md
        assert "impossible" in md.splitlines()[0]
