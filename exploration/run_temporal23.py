"""CLI runner: temporal Phase 2 (embeddings over time) + Phase 3 (spatial).

    uv run python exploration/run_temporal23.py \
        --source data/08fd33_0.mp4 --stride 15 --embedder dinov2

Outputs under data/exploration/temporal/<video_stem>/:
    embedding_stats.csv        — per-frame silhouette / centroid geometry
    embedding_map_umap.png     — shared UMAP projection, time-colored + windows
    embedding_map_tsne.png     — same for t-SNE
    embedding_traces.png       — silhouette & centroid-distance traces
    spatial_panels.png         — position heatmaps, keypoint coverage, H jitter
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_pipeline.config import KEYPOINT_VERTICES_M, PIPELINE, PitchSpec
from detection_pipeline.infer import load_keypoint_model, load_player_model
from detection_pipeline.teams_pipeline import fit_teams_from_video
from exploration.embedders import load_embedder
from exploration.temporal_embeddings import (
    analyze_embeddings, collect_embeddings_over_time, write_embedding_csv,
)
from exploration.temporal_plots23 import (
    render_embedding_traces, render_shared_projection_map, render_spatial_panels,
)
from exploration.temporal_spatial import (
    collect_spatial_over_time, homography_jitter, keypoint_coverage,
    position_heatmaps,
)

from footlab.pitch import draw_pitch


def main():
    parser = argparse.ArgumentParser(description="Temporal Phase 2+3 analytics")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--stride", type=int, default=15,
                        help="sample every Nth frame (15 ≈ 2 samples/sec at 30fps)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--embedder", type=str, default="dinov2",
                        help="registry key from exploration.embedders")
    parser.add_argument("--fit-stride", type=int, default=60,
                        help="team-classifier fit sampling stride")
    parser.add_argument("--skip-spatial-teams", action="store_true",
                        help="spatial heatmaps without team split (faster)")
    args = parser.parse_args()

    out_dir = Path("data/exploration/temporal") / Path(args.source).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    det_cfg = PIPELINE["detector"]
    hom_cfg = PIPELINE["homography"]

    player_model = load_player_model(det_cfg["player_model_path"], model_type="local")
    kp_type = det_cfg.get("keypoint_model_type", "roboflow")
    keypoint_model = (
        load_keypoint_model(det_cfg["keypoint_model_path"], model_type="local")
        if kp_type == "local"
        else load_keypoint_model(det_cfg["keypoint_model_id"])
    )

    # ---------------- Phase 2 ----------------
    print("\n=== Phase 2: embeddings over time ===")
    spec, embedder = load_embedder(args.embedder, device="cpu")
    print(f"embedder: {spec.description}")
    embeddings, frame_idx, frame_ids, times = collect_embeddings_over_time(
        args.source, player_model, embedder,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"], ball_conf=det_cfg["ball_conf"],
        stride=args.stride, max_frames=args.max_frames,
    )
    print(f"pooled embeddings: {embeddings.shape} from {len(frame_ids)} frames")

    stats, labels, proj2_umap, proj2_tsne = analyze_embeddings(
        embeddings, frame_idx, frame_ids, times, n_frames_total=len(frame_ids),
    )
    csv_path = write_embedding_csv(stats, out_dir / "embedding_stats.csv")

    sil = np.array([s.silhouette for s in stats])
    cdist = np.array([s.centroid_distance for s in stats])
    i0 = np.array([s.intra_dist_team0 for s in stats])
    i1 = np.array([s.intra_dist_team1 for s in stats])
    tvec = np.array([s.time_s for s in stats])

    render_embedding_traces(
        tvec, sil, cdist, i0, i1,
        output_png=str(out_dir / "embedding_traces.png"),
        title=f"Embedding structure over time — {Path(args.source).name} ({args.embedder})",
    )
    render_shared_projection_map(
        proj2_umap, frame_idx, times, labels,
        output_png=str(out_dir / "embedding_map_umap.png"),
        reducer_name="UMAP",
        title=f"Shared UMAP map — {Path(args.source).name} ({args.embedder})",
    )
    render_shared_projection_map(
        proj2_tsne, frame_idx, times, labels,
        output_png=str(out_dir / "embedding_map_tsne.png"),
        reducer_name="t-SNE",
        title=f"Shared t-SNE map — {Path(args.source).name} ({args.embedder})",
    )
    print(f"  traces + maps saved (silhouette mean {sil.mean():.3f})")

    # ---------------- Phase 3 ----------------
    print("\n=== Phase 3: spatial over time ===")
    clf = None
    if not args.skip_spatial_teams:
        print("fitting team classifier for heatmap split...")
        clf = fit_teams_from_video(
            args.source, player_model,
            class_map=det_cfg["class_map"],
            player_conf=det_cfg["player_conf"], ball_conf=det_cfg["ball_conf"],
            stride=args.fit_stride, max_frames=15,
        )
    frames, fps = collect_spatial_over_time(
        args.source, player_model, keypoint_model,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"], ball_conf=det_cfg["ball_conf"],
        keypoint_conf=det_cfg["keypoint_conf"],
        homography_cfg=hom_cfg,
        team_classifier=clf,
        stride=args.stride, max_frames=args.max_frames,
    )
    spec_pitch = PitchSpec()
    heatmaps = position_heatmaps(frames, spec_pitch)
    coverage = keypoint_coverage(frames)
    jitter = homography_jitter(frames)
    jtimes = np.array([f.time_s for f in frames])

    render_spatial_panels(
        heatmaps, coverage, KEYPOINT_VERTICES_M, jtimes, jitter,
        output_png=str(out_dir / "spatial_panels.png"),
        title=f"Spatial panels — {Path(args.source).name}",
        draw_pitch_fn=draw_pitch,
    )
    finite_j = jitter[np.isfinite(jitter)]
    print(f"  spatial panels saved (median jitter "
          f"{np.median(finite_j) if len(finite_j) else float('nan'):.1f}px)")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
