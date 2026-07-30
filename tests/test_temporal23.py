"""Tests for exploration.temporal_embeddings and temporal_spatial.

Heavy models (detector, embedder) are mocked; the geometry and statistics —
cosine geometry, per-frame silhouette plumbing, heatmap binning, coverage
frequencies, jitter math — are tested for real.
"""

from __future__ import annotations

import numpy as np
import pytest

from exploration.temporal_embeddings import (
    EmbeddingFrameStats, _cosine_distance_matrix, analyze_embeddings,
    write_embedding_csv,
)
from exploration.temporal_spatial import (
    SpatialFrameData, homography_jitter, keypoint_coverage, position_heatmaps,
)
from detection_pipeline.config import PitchSpec


# ---------------------------------------------------------------------------
# Embedding geometry
# ---------------------------------------------------------------------------

class TestCosineDistance:
    def test_identical_vectors_zero(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = _cosine_distance_matrix(x)
        assert D[0, 0] == pytest.approx(0.0)
        assert D[1, 1] == pytest.approx(0.0)

    def test_orthogonal_one(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = _cosine_distance_matrix(x)
        assert D[0, 1] == pytest.approx(1.0)

    def test_opposite_two(self):
        x = np.array([[1.0, 0.0], [-1.0, 0.0]])
        D = _cosine_distance_matrix(x)
        assert D[0, 1] == pytest.approx(2.0)


class TestAnalyzeEmbeddings:
    def _blob_data(self, n_frames=4, per_frame=8, seed=0):
        """Two well-separated clouds, crops spread across frames."""
        rng = np.random.default_rng(seed)
        embs, fidx = [], []
        for f in range(n_frames):
            a = rng.normal([1, 0, 0], 0.05, (per_frame // 2, 3))
            b = rng.normal([0, 1, 0], 0.05, (per_frame // 2, 3))
            e = np.vstack([a, b])
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            embs.append(e)
            fidx.append(np.full(len(e), f))
        embeddings = np.vstack(embs)
        frame_idx = np.concatenate(fidx)
        frame_ids = np.arange(n_frames) * 15
        times = frame_ids / 25.0
        return embeddings, frame_idx, frame_ids, times

    def test_per_frame_stats_shape_and_quality(self):
        embs, fidx, fids, times = self._blob_data()
        stats, labels, p_umap, p_tsne = analyze_embeddings(
            embs, fidx, fids, times, n_frames_total=len(fids),
        )
        assert len(stats) == 4
        assert len(labels) == len(embs)
        assert p_umap.shape == (len(embs), 2)
        assert p_tsne.shape == (len(embs), 2)
        # Separated blobs → good silhouette, healthy geometry
        for s in stats:
            assert s.silhouette > 0.3
            assert s.centroid_distance > s.intra_dist_team0
            assert s.centroid_distance > s.intra_dist_team1

    def test_csv_roundtrip_fields(self, tmp_path):
        s = EmbeddingFrameStats(frame_id=15, time_s=0.6, n_crops=12,
                                silhouette=0.5, centroid_distance=0.4,
                                intra_dist_team0=0.1, intra_dist_team1=0.1,
                                team_sizes="6/6")
        path = write_embedding_csv([s], tmp_path / "e.csv")
        text = path.read_text()
        assert "silhouette" in text and "6/6" in text


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

def _spatial_frame(fid=0, rms=0.5, pos=None, teams=None, kps=None, corners=None):
    return SpatialFrameData(
        frame_id=fid, time_s=fid / 25.0,
        positions=pos if pos is not None else np.array([[-20, 5], [30, -5]]),
        teams=teams if teams is not None else np.array([0, 1]),
        homography_rms=rms,
        keypoints_seen=kps if kps is not None else [0, 5, 13],
        corner_pixels=corners if corners is not None else np.array(
            [[100, 500], [100, 100], [1800, 500], [1800, 100]], dtype=float),
    )


class TestPositionHeatmaps:
    def test_team_split(self):
        frames = [_spatial_frame(fid=i) for i in range(4)]
        hm = position_heatmaps(frames, PitchSpec())
        assert hm["team0"].sum() == 4   # one team-0 point per frame
        assert hm["team1"].sum() == 4
        assert hm["good_h"].sum() + hm["bad_h"].sum() == 8

    def test_empty_frames(self):
        frames = [_spatial_frame(pos=np.empty((0, 2)), teams=np.zeros(0, int))]
        hm = position_heatmaps(frames, PitchSpec())
        assert hm["team0"].sum() == 0


class TestKeypointCoverage:
    def test_frequencies(self):
        frames = [
            _spatial_frame(kps=[0, 5]),
            _spatial_frame(kps=[0]),
            _spatial_frame(kps=[0, 5, 13]),
        ]
        cov = keypoint_coverage(frames)
        assert cov[0] == pytest.approx(1.0)
        assert cov[5] == pytest.approx(2 / 3)
        assert cov[13] == pytest.approx(1 / 3)
        assert cov[29] == pytest.approx(0.0)


class TestHomographyJitter:
    def test_zero_when_static(self):
        frames = [_spatial_frame(fid=i) for i in range(3)]
        j = homography_jitter(frames)
        assert np.isnan(j[0])
        assert j[1] == pytest.approx(0.0)
        assert j[2] == pytest.approx(0.0)

    def test_detects_shift(self):
        frames = [_spatial_frame(fid=0),
                  _spatial_frame(fid=1, corners=np.array(
                      [[110, 500], [100, 100], [1800, 500], [1800, 100]], dtype=float))]
        j = homography_jitter(frames)
        assert j[1] == pytest.approx(2.5)  # one corner moved 10px, mean = 2.5

    def test_nan_when_h_missing(self):
        frames = [_spatial_frame(fid=0),
                  _spatial_frame(fid=1, corners=np.full((4, 2), np.nan))]
        j = homography_jitter(frames)
        assert np.isnan(j[1])
