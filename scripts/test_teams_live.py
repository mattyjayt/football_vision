"""Live test: TeamClassifier + t-SNE viz on a single real frame.

Pipeline:
    1. Run the local player detector on one frame.
    2. Crop each detected player (goalkeepers + referees included so we can
       SEE where they land — the t-SNE plot is exactly for spotting this).
    3. SigLIP-embed the crops.
    4. Fit TeamClassifier (UMAP + KMeans) on the crops, predict team labels.
    5. t-SNE scatter of the embeddings, colored by predicted team.
    6. Annotated frame: boxes colored by predicted team.

Usage:
    uv run python scripts/test_teams_live.py --source data/08fd33_0.mp4 --frame 80
"""

from __future__ import annotations

import argparse
import os

# UMAP's numba/pynndescent parallelism can segfault on macOS (exit 139)
# after torch has loaded its OpenMP runtime. Force single-threaded numba
# BEFORE any numeric library gets imported.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_pipeline.config import PIPELINE
from detection_pipeline.infer import load_player_model, detect_players_and_ball
from detection_pipeline.teams import TeamClassifier
from detection_pipeline.viz_embeddings import tsne_project, render_tsne_scatter


def main():
    parser = argparse.ArgumentParser(description="Live TeamClassifier test on one frame")
    parser.add_argument("--source", type=str, default="data/08fd33_0.mp4")
    parser.add_argument("--frame", type=int, default=80)
    args = parser.parse_args()

    # --- 1. Read the frame ---
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {args.frame} from {args.source}")

    # --- 2. Detect players ---
    det_cfg = PIPELINE["detector"]
    model = load_player_model(det_cfg["player_model_path"], model_type="local")
    players, _ball = detect_players_and_ball(
        frame, model,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"],
        ball_conf=det_cfg["ball_conf"],
    )
    print(f"Detected {len(players)} player-ish boxes "
          f"(players + goalkeepers + referees)")

    # --- 3. Crop each detection ---
    crops = []
    kept = []
    for det in players:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        kept.append(det)
    print(f"Cropped {len(crops)} usable player images")

    # --- 4. Embed (SigLIP) ---
    print("Loading SigLIP (first run downloads ~400MB)...")
    clf = TeamClassifier(device="cpu")
    print("Extracting embeddings...")
    features = clf.extract_features(crops)
    print(f"  embeddings: {features.shape}")

    # --- 5. Cluster (UMAP + KMeans on player-class crops only) ---
    player_idx = [i for i, d in enumerate(kept) if d.class_name == "player"]
    if len(player_idx) >= 4:
        player_features = features[player_idx]
        clf.reducer.fit(player_features)
        clf.cluster_model.fit(clf.reducer.transform(player_features))
        labels = clf.cluster_model.predict(clf.reducer.transform(features))
    else:
        print("WARNING: too few outfield players to cluster — labelling all 0")
        labels = np.zeros(len(crops), dtype=int)

    sizes = {lab: int((labels == lab).sum()) for lab in set(labels.tolist())}
    print(f"Cluster sizes: {sizes}")

    # --- 6. t-SNE visualization ---
    coords = tsne_project(features)
    tsne_png = render_tsne_scatter(
        coords, labels,
        output_png=f"data/tsne_frame_{args.frame}.png",
        title=f"t-SNE — frame {args.frame} player embeddings (colored by predicted team)",
    )
    print(f"t-SNE scatter saved: {tsne_png}")

    # --- 7. Annotated frame with team-colored boxes ---
    team_colors = {0: (0, 0, 255), 1: (255, 0, 0)}  # BGR: red, blue
    annotated = frame.copy()
    for det, lab in zip(kept, labels):
        x1, y1, x2, y2 = det.xyxy.astype(int)
        color = team_colors.get(int(lab), (0, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"T{lab}:{det.class_name[:2]}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out_frame = f"data/teams_frame_{args.frame}.png"
    cv2.imwrite(out_frame, annotated)
    print(f"Annotated frame saved: {out_frame}")


if __name__ == "__main__":
    main()
