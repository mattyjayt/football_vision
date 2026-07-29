"""Tests for detection_pipeline.teams — team classification.

Strategy: SigLIP/UMAP/KMeans are heavy (model download, ~1 s/crop on CPU),
so the classifier tests MOCK extract_features with deterministic synthetic
embeddings. The clustering logic, batching, and goalkeeper assignment are
tested for real — that's where OUR bugs would live, not in Google's model.
"""

from __future__ import annotations

import numpy as np
import pytest

from detection_pipeline.teams import (
    TeamClassifier,
    create_batches,
    resolve_goalkeepers_team_id,
)


# ---------------------------------------------------------------------------
# create_batches
# ---------------------------------------------------------------------------

class TestCreateBatches:
    def test_even_split(self):
        batches = list(create_batches(range(6), 3))
        assert batches == [[0, 1, 2], [3, 4, 5]]

    def test_remainder_batch(self):
        batches = list(create_batches(range(7), 3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6]]

    def test_batch_larger_than_sequence(self):
        batches = list(create_batches([1, 2], 10))
        assert batches == [[1, 2]]

    def test_empty(self):
        assert list(create_batches([], 4)) == []

    def test_zero_batch_size_clamped_to_one(self):
        batches = list(create_batches([1, 2, 3], 0))
        assert batches == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# TeamClassifier — clustering logic with mocked feature extraction
# ---------------------------------------------------------------------------

def _make_classifier_no_heavy_init() -> TeamClassifier:
    """Build a TeamClassifier without downloading SigLIP.

    We bypass __init__ and attach lightweight stand-ins for the reducer and
    cluster model, plus a stubbed extract_features. Tests then exercise the
    REAL fit/predict flow (reduce → cluster), not the network.
    """
    clf = TeamClassifier.__new__(TeamClassifier)
    from sklearn.cluster import KMeans

    class IdentityReducer:
        """UMAP stand-in: pass features through unchanged (they're already
        low-dim in tests), with fit_transform/transform mirroring the UMAP API."""
        def fit_transform(self, data):
            return data

        def transform(self, data):
            return data

    clf.reducer = IdentityReducer()
    clf.cluster_model = KMeans(n_clusters=2, n_init=10, random_state=42)
    return clf


class TestTeamClassifierClustering:
    def _two_blobs(self, n_per_team: int = 20, seed: int = 0):
        """Two clearly separated 3D point clouds = two 'jersey colours'."""
        rng = np.random.default_rng(seed)
        team_a = rng.normal(loc=[0, 0, 0], scale=0.1, size=(n_per_team, 3))
        team_b = rng.normal(loc=[10, 10, 10], scale=0.1, size=(n_per_team, 3))
        return team_a, team_b

    def test_fit_predict_separates_two_teams(self):
        clf = _make_classifier_no_heavy_init()
        team_a, team_b = self._two_blobs()
        all_feats = np.vstack([team_a, team_b])

        # Mock: extract_features returns the synthetic embeddings as-is,
        # regardless of the (fake) crops passed in.
        clf.extract_features = lambda crops: all_feats[: len(crops)]

        clf.fit(["crop"] * len(all_feats))
        preds = clf.predict(["crop"] * len(all_feats))

        # Labels are arbitrary (0/1 can flip between runs) — test the SPLIT:
        # everything from blob A must share a label, and blob B the other.
        preds_a = preds[: len(team_a)]
        preds_b = preds[len(team_a):]
        assert len(set(preds_a.tolist())) == 1
        assert len(set(preds_b.tolist())) == 1
        assert preds_a[0] != preds_b[0]

    def test_predict_empty(self):
        clf = _make_classifier_no_heavy_init()
        clf.extract_features = lambda crops: np.zeros((0, 3))
        result = clf.predict([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_predict_single_crop(self):
        clf = _make_classifier_no_heavy_init()
        team_a, team_b = self._two_blobs()
        all_feats = np.vstack([team_a, team_b])
        clf.extract_features = lambda crops: all_feats[: len(crops)]
        clf.fit(["crop"] * len(all_feats))

        # A new crop from team A's region should land on team A's label.
        clf.extract_features = lambda crops: np.array([[0.05, -0.02, 0.01]])
        pred = clf.predict(["new_crop"])
        assert len(pred) == 1
        # Recompute team A's label from the training predictions.
        clf.extract_features = lambda crops: all_feats[: len(crops)]
        train_preds = clf.predict(["crop"] * len(all_feats))
        assert pred[0] == train_preds[0]


# ---------------------------------------------------------------------------
# resolve_goalkeepers_team_id
# ---------------------------------------------------------------------------

class TestResolveGoalkeepers:
    def test_nearest_centroid_wins(self):
        # Team 0 defends the left goal (x<0), team 1 the right (x>0).
        players_xy = np.array(
            [[-40, 0], [-30, 10], [-35, -10], [-20, 5],   # team 0 (left)
             [40, 0], [30, -10], [35, 10], [20, -5]]       # team 1 (right)
        )
        players_team_id = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        goalkeepers_xy = np.array([[-50, 0], [50, 0]])     # left GK, right GK

        result = resolve_goalkeepers_team_id(players_xy, players_team_id, goalkeepers_xy)
        assert result.tolist() == [0, 1]

    def test_empty_goalkeepers(self):
        players_xy = np.array([[0, 0], [10, 0]])
        players_team_id = np.array([0, 1])
        result = resolve_goalkeepers_team_id(
            players_xy, players_team_id, np.empty((0, 2))
        )
        assert len(result) == 0

    def test_single_goalkeeper(self):
        players_xy = np.array([[-10, 0], [-12, 3], [30, 0], [32, -3]])
        players_team_id = np.array([0, 0, 1, 1])
        goalkeepers_xy = np.array([[28, 1]])
        result = resolve_goalkeepers_team_id(players_xy, players_team_id, goalkeepers_xy)
        assert result.tolist() == [1]
