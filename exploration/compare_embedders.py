"""The runner: compare embedders on real player crops from one frame.

For each registered embedder:
    1. embed the SAME crops (identical input → fair comparison)
    2. project with UMAP (3D — the production reducer) and t-SNE (2D — the
       visualization specialist)
    3. cluster the UMAP-3D projection with KMeans(k=2) — same as production
    4. score the partition (silhouette / Davies-Bouldin / Calinski-Harabasz)
    5. render 2D comparison grid + interactive 3D HTML per model/reducer

Usage:
    uv run python exploration/compare_embedders.py \
        --source data/08fd33_0.mp4 --frame 80 \
        --models siglip_v1 dinov2          # subset, or omit for all
"""

from __future__ import annotations

import argparse
import os

# See scripts/test_teams_live.py — numba/UMAP parallelism can segfault on
# macOS after torch loads its OpenMP runtime. Must precede numeric imports.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_pipeline.config import PIPELINE
from detection_pipeline.infer import load_player_model, detect_players_and_ball
from exploration.embedders import EMBEDDER_REGISTRY, load_embedder
from exploration.metrics import ClusterMetrics, evaluate_clustering, metrics_table_markdown
from exploration.plots import render_comparison_grid_2d, render_interactive_3d

OUT_DIR = Path("data/exploration")


def detect_and_crop(source: str, frame_id: int):
    """Run the production detector on one frame; return (crops, detections)."""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id} from {source}")

    det_cfg = PIPELINE["detector"]
    model = load_player_model(det_cfg["player_model_path"], model_type="local")
    players, _ = detect_players_and_ball(
        frame, model,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"],
        ball_conf=det_cfg["ball_conf"],
    )
    crops, kept = [], []
    for det in players:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size:
            crops.append(crop)
            kept.append(det)
    return crops, kept


def project_2d(embeddings: np.ndarray, reducer: str) -> np.ndarray:
    if reducer == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    if reducer == "tsne":
        from sklearn.manifold import TSNE
        perp = min(15.0, max(1.0, (len(embeddings) - 1) / 3))
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=42).fit_transform(embeddings)
    raise ValueError(reducer)


def project_3d_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Embedder comparison study")
    parser.add_argument("--source", type=str, default="data/08fd33_0.mp4")
    parser.add_argument("--frame", type=int, default=80)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of registry keys; default = all")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_names = args.models or list(EMBEDDER_REGISTRY)

    crops, kept = detect_and_crop(args.source, args.frame)
    print(f"Frame {args.frame}: {len(crops)} crops from {len(kept)} detections")
    player_mask = np.array([d.class_name == "player" for d in kept])

    from sklearn.cluster import KMeans
    all_metrics: list[ClusterMetrics] = []
    projections_2d: dict[str, dict[str, np.ndarray]] = {}
    labels_per_model: dict[str, np.ndarray] = {}

    for name in model_names:
        spec, embedder = load_embedder(name, device="cpu")
        print(f"\n=== {spec.description} ===")
        embeddings = embedder.embed(crops)
        print(f"  embeddings: {embeddings.shape}")

        # Cluster on UMAP-3D (production-equivalent), fit on outfield players.
        proj3 = project_3d_umap(embeddings)
        fit_idx = np.where(player_mask)[0] if player_mask.sum() >= 4 else np.arange(len(crops))
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(proj3[fit_idx])
        labels = km.predict(proj3)

        all_metrics.append(evaluate_clustering(name, embeddings, proj3, labels))
        labels_per_model[name] = labels

        # 2D projections for the grid figure.
        projections_2d[name] = {
            "umap": project_2d(embeddings, "umap"),
            "tsne": project_2d(embeddings, "tsne"),
        }

        # Interactive 3D (UMAP only — t-SNE has no transform, and its 3D adds
        # little over its 2D; UMAP-3D is the production-consistent view).
        render_interactive_3d(
            proj3, labels,
            output_html=str(OUT_DIR / f"embedder_3d_{name}_umap.html"),
            title=f"{name} — UMAP 3D, colored by KMeans cluster",
        )
        print(f"  3D HTML: data/exploration/embedder_3d_{name}_umap.html")

    grid_png = render_comparison_grid_2d(
        projections_2d, labels_per_model,
        output_png=str(OUT_DIR / "embedder_grid_2d.png"),
        title=f"Embedder comparison — frame {args.frame} ({len(crops)} crops)",
    )
    table = metrics_table_markdown(all_metrics)
    (OUT_DIR / "embedder_metrics.md").write_text(
        f"# Embedder comparison — {args.source} frame {args.frame}\n\n{table}\n"
    )
    print(f"\n{table}\n")
    print(f"2D grid: {grid_png}")
    print(f"Metrics: {OUT_DIR / 'embedder_metrics.md'}")


if __name__ == "__main__":
    main()
