"""Tests for the exploration framework (embedders/metrics/plots).

No model downloads here: embedders are tested at the registry/interface
level; metrics and projections run on small synthetic blobs. The live model
comparison is exercised by exploration/compare_embedders.py, not pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from exploration.embedders import EMBEDDER_REGISTRY, load_embedder
from exploration.metrics import evaluate_clustering, metrics_table_markdown
from exploration.compare_embedders import project_2d


class TestEmbedderRegistry:
    def test_expected_models_registered(self):
        assert {"siglip_v1", "siglip_v2", "dinov2"} <= set(EMBEDDER_REGISTRY)

    def test_specs_have_hf_paths(self):
        for spec in EMBEDDER_REGISTRY.values():
            assert "/" in spec.hf_path  # org/model form
            assert spec.description

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown embedder"):
            load_embedder("not_a_model")


class TestMetrics:
    def _partition(self, n=20, seed=0, separation=10.0):
        rng = np.random.default_rng(seed)
        a = rng.normal(0, 0.5, (n, 3))
        b = rng.normal(separation, 0.5, (n, 3))
        proj = np.vstack([a, b])
        labels = np.array([0] * n + [1] * n)
        return proj, labels

    def test_well_separated_scores_good(self):
        proj, labels = self._partition(separation=10.0)
        m = evaluate_clustering("test", proj, proj, labels)
        assert m.silhouette > 0.8
        assert m.davies_bouldin < 0.5
        assert m.cluster_sizes == (20, 20)

    def test_overlapping_scores_worse(self):
        proj, labels = self._partition(separation=0.5)
        m = evaluate_clustering("test", proj, proj, labels)
        assert m.silhouette < 0.3

    def test_table_markdown_shape(self):
        proj, labels = self._partition()
        m = evaluate_clustering("model_x", proj, proj, labels)
        md = metrics_table_markdown([m])
        assert "| model_x |" in md
        assert "silhouette" in md.splitlines()[0]


class TestProjections:
    def test_umap_2d_shape(self):
        rng = np.random.default_rng(0)
        emb = np.vstack([rng.normal(0, 1, (12, 16)), rng.normal(5, 1, (12, 16))])
        coords = project_2d(emb, "umap")
        assert coords.shape == (24, 2)

    def test_tsne_2d_shape_and_perplexity_clamp(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(5, 8))
        coords = project_2d(emb, "tsne")  # perplexity must clamp for tiny N
        assert coords.shape == (5, 2)

    def test_unknown_reducer(self):
        with pytest.raises(ValueError):
            project_2d(np.zeros((10, 4)), "pca")  # not wired into project_2d
