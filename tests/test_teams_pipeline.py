"""Tests for detection_pipeline.teams_pipeline — team fitting & assignment.

Heavy models (SigLIP, detector) are mocked; the workflow logic — crop
alignment, label routing, referee exclusion, attack-side resolution — is
tested for real, because that's where OUR bugs would live.
"""

from __future__ import annotations

import numpy as np
import pytest

from detection_pipeline.infer import Detection
from detection_pipeline.teams_pipeline import (
    assign_teams,
    crop_detections,
    resolve_attack_side,
)


def _det(x1, y1, x2, y2, cls="player", conf=0.9):
    return Detection(
        xyxy=np.array([x1, y1, x2, y2], dtype=float),
        confidence=conf, class_id=0, class_name=cls,
    )


class _MockClassifier:
    """TeamClassifier stand-in: predictable label sequence, records calls."""

    def __init__(self, labels):
        self._labels = list(labels)
        self.predict_calls = 0

    def predict(self, crops):
        self.predict_calls += 1
        return np.array(self._labels[: len(crops)])


class TestCropDetections:
    def test_crops_match_boxes(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[10:30, 20:60] = 255
        dets = [_det(20, 10, 60, 30)]
        crops = crop_detections(frame, dets)
        assert len(crops) == 1
        assert crops[0].shape == (20, 40, 3)
        assert crops[0].max() == 255

    def test_out_of_bounds_box_clamped(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        dets = [_det(-10, -10, 30, 30)]
        crops = crop_detections(frame, dets)
        assert crops[0].shape == (30, 30, 3)

    def test_empty_crop_replaced_by_placeholder(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        dets = [_det(100, 100, 200, 200)]  # fully outside
        crops = crop_detections(frame, dets)
        assert crops[0].size > 0  # placeholder, not empty


class TestAssignTeams:
    def test_players_get_classifier_labels_refs_excluded(self):
        dets = [_det(0, 0, 10, 10, "player"), _det(10, 0, 20, 10, "player"),
                _det(20, 0, 30, 10, "player"), _det(30, 0, 40, 10, "referee")]
        pitch_xy = np.array([[-30, 0], [-10, 5], [30, 0], [0, 10]], dtype=float)
        crops = [np.zeros((4, 4, 3), dtype=np.uint8)] * 4
        clf = _MockClassifier(labels=[0, 1, 0])

        out = assign_teams(crops, dets, pitch_xy, clf)
        assert clf.predict_calls == 1
        assert out.labels.tolist() == [0, 1, 0, -1]  # referee = -1

    def test_goalkeeper_resolved_by_centroid(self):
        # Team 0 left, team 1 right; GK on the left → team 0.
        dets = [_det(0, 0, 10, 10, "player"), _det(10, 0, 20, 10, "player"),
                _det(20, 0, 30, 10, "player"), _det(30, 0, 40, 10, "player"),
                _det(40, 0, 50, 10, "goalkeeper")]
        pitch_xy = np.array([[-30, 0], [-20, 5], [30, 0], [20, -5], [-50, 0]],
                            dtype=float)
        crops = [np.zeros((4, 4, 3), dtype=np.uint8)] * 5
        clf = _MockClassifier(labels=[0, 0, 1, 1])

        out = assign_teams(crops, dets, pitch_xy, clf)
        assert out.labels[4] == 0  # GK joins the left team

    def test_empty_detections(self):
        clf = _MockClassifier(labels=[])
        out = assign_teams([], [], np.empty((0, 2)), clf)
        assert len(out.labels) == 0
        assert out.attacking_team is None


class TestResolveAttackSide:
    def test_gk_on_left_attacks_right(self):
        dets = [_det(0, 0, 10, 10, "goalkeeper")]
        labels = np.array([0])
        pitch_xy = np.array([[-50, 0]], dtype=float)
        # Team 0's GK defends left goal → team 0 attacks +x
        assert resolve_attack_side(labels, pitch_xy, dets) == 0

    def test_gk_on_right_attacks_left(self):
        dets = [_det(0, 0, 10, 10, "goalkeeper")]
        labels = np.array([1])
        pitch_xy = np.array([[50, 0]], dtype=float)
        assert resolve_attack_side(labels, pitch_xy, dets) == 0

    def test_fallback_defensive_depth(self):
        # No GK visible: team with deeper defensive line (left) attacks +x.
        dets = [_det(0, 0, 10, 10, "player") for _ in range(6)]
        labels = np.array([0, 0, 0, 1, 1, 1])
        pitch_xy = np.array([[-40, 0], [-35, 5], [-30, -5],
                             [30, 0], [35, 5], [40, -5]], dtype=float)
        assert resolve_attack_side(labels, pitch_xy, dets) == 0

    def test_no_players_returns_none(self):
        dets = [_det(0, 0, 10, 10, "referee")]
        labels = np.array([-1])
        pitch_xy = np.array([[0, 0]], dtype=float)
        assert resolve_attack_side(labels, pitch_xy, dets) is None
