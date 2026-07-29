"""Tests for detection_pipeline.viz_embeddings — t-SNE sidecar.

t-SNE is stochastic and slow on large inputs, so tests use tiny synthetic
blobs with fixed seeds. We test invariants (shape, determinism, cluster
separation, perplexity clamping), never exact coordinates.
"""

from __future__ import annotations

import numpy as np

from detection_pipeline.viz_embeddings import tsne_project, render_tsne_scatter


def _two_blobs(n: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0, 0, 0], scale=0.05, size=(n, 3))
    b = rng.normal(loc=[10, 10, 10], scale=0.05, size=(n, 3))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


class TestTsneProject:
    def test_output_shape(self):
        feats, _ = _two_blobs()
        coords = tsne_project(feats, perplexity=8)
        assert coords.shape == (len(feats), 2)

    def test_deterministic_with_seed(self):
        feats, _ = _two_blobs()
        c1 = tsne_project(feats, perplexity=8, random_state=7)
        c2 = tsne_project(feats, perplexity=8, random_state=7)
        np.testing.assert_allclose(c1, c2)

    def test_separated_blobs_stay_separated(self):
        """The whole point of the viz: distinct clusters must not overlap."""
        feats, labels = _two_blobs()
        coords = tsne_project(feats, perplexity=8)
        # Distance between cluster centroids must exceed the spread within.
        ca = coords[labels == 0].mean(axis=0)
        cb = coords[labels == 1].mean(axis=0)
        between = np.linalg.norm(ca - cb)
        within = max(
            np.linalg.norm(coords[labels == 0] - ca, axis=1).max(),
            np.linalg.norm(coords[labels == 1] - cb, axis=1).max(),
        )
        assert between > 2 * within

    def test_perplexity_clamped_for_tiny_inputs(self):
        feats = np.random.default_rng(0).normal(size=(4, 8))
        coords = tsne_project(feats, perplexity=50)  # absurd for N=4
        assert coords.shape == (4, 2)


class TestRenderTsneScatter:
    def test_saves_file(self, tmp_path):
        feats, labels = _two_blobs()
        coords = tsne_project(feats, perplexity=8)
        out = tmp_path / "tsne.png"
        result = render_tsne_scatter(coords, labels, output_png=str(out))
        assert out.exists() and out.stat().st_size > 0
        assert result == str(out)

    def test_no_labels(self, tmp_path):
        feats, _ = _two_blobs()
        coords = tsne_project(feats, perplexity=8)
        out = tmp_path / "tsne_nolab.png"
        render_tsne_scatter(coords, None, output_png=str(out))
        assert out.exists()
