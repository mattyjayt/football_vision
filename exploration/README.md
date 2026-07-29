# Engineering Exploration

A workbench for *experiments*, kept deliberately separate from the production
pipeline (`detection_pipeline/`) and the analytics brain (`src/footlab/`).

**Rule of the house:** nothing here is on the critical path. Code here answers
questions — "which embedder clusters jerseys best?", "does UMAP see the same
structure t-SNE sees?" — and its outputs are tables, figures, and HTML plots,
not JSONL. When an exploration produces a *winner*, the winner gets promoted
into the pipeline with proper tests.

## Layout

| Module | Question it answers |
|---|---|
| `embedders.py` | Pluggable registry of embedding models (SigLIP v1/v2, DINOv2, …) |
| `metrics.py` | Cluster-quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) |
| `plots.py` | 2D scatter grids (matplotlib) + interactive 3D (plotly) |
| `compare_embedders.py` | The runner: same crops → all models → metrics + figures |

## Running the embedder comparison

```bash
uv run python exploration/compare_embedders.py --source data/08fd33_0.mp4 --frame 80
```

Outputs (all under `data/exploration/`):
- `embedder_metrics.md` — the metrics table, ready to paste into notes
- `embedder_grid_2d.png` — one row per model, UMAP vs t-SNE side by side
- `embedder_3d_<model>_<reducer>.html` — interactive 3D scatter per combo

First run downloads ~1 GB of model weights (SigLIP v1/v2, DINOv2-small).
Subsequent runs use the HuggingFace cache.

## Reading the results (a short guide)

- **Silhouette (higher = better, −1…1):** are points closer to their own
  cluster than the other? >0.5 is strong; <0.2 means the "teams" blur.
- **Davies-Bouldin (lower = better):** within-cluster spread vs. between-
  cluster distance. <1.0 is good separation.
- **Calinski-Harabasz (higher = better):** variance ratio. Only meaningful
  *relative* to other rows in the same table, never as an absolute.
- **The plots are the real answer.** Metrics compress structure into one
  number; the scatters show *which* crops confuse the model (goalkeepers?
  referees? occluded players?). Look at both.

## Future studies (the map grows here)

- Reducer bake-off: PCA vs UMAP vs t-SNE on the same embeddings
- Cluster algorithm study: KMeans vs GMM vs HDBSCAN (referee-outlier handling)
- Data-condition audit: embeddings colored by lighting/camera/occlusion metadata
- Tracker comparison: ByteTrack vs OC-SORT vs DeepSORT on the same clip
