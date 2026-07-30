"""Demo 11 — END-TO-END: our own video → footlab analytics on a real frame.

THE milestone of Path B: everything we've built, chained.

    video frame
      → player detection (local fine-tuned model)
      → pitch keypoints (local 32-kpt model) → homography → pitch meters
      → TeamClassifier (fitted on crops sampled across the video)
      → radar JSONL (with teams + attacking side)
      → footlab.io_radar → FrozenFrame (attack/defend masks populated)
      → pitch control surface + value surface on a REAL frame

Usage:
    uv run python scripts/11_demo_e2e.py --source data/08fd33_0.mp4 --frame 80
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_pipeline.config import PIPELINE
from detection_pipeline.frames import write_jsonl
from detection_pipeline.__main__ import process_single_frame
from detection_pipeline.infer import load_player_model
from detection_pipeline.teams_pipeline import fit_teams_from_video

from footlab import pitch, pitch_control, value_surface, viz
from footlab.io_radar import load_radar_jsonl


def main():
    parser = argparse.ArgumentParser(description="E2E: video → FrozenFrame → analytics")
    parser.add_argument("--source", type=str, default="data/08fd33_0.mp4")
    parser.add_argument("--frame", type=int, default=80)
    parser.add_argument("--stride", type=int, default=60,
                        help="Team-fit sampling stride (frames)")
    parser.add_argument("--max-fit-frames", type=int, default=25)
    args = parser.parse_args()

    det_cfg = PIPELINE["detector"]
    out_jsonl = f"data/e2e_frame_{args.frame}.jsonl"

    print("=" * 64)
    print("E2E — video → detection → teams → FrozenFrame → pitch control")
    print("=" * 64)

    # --- 1. Fit the team classifier across the video ---
    print(f"\n[1/4] Fitting TeamClassifier (stride={args.stride}, "
          f"max {args.max_fit_frames} frames)...")
    t0 = time.perf_counter()
    player_model = load_player_model(det_cfg["player_model_path"], model_type="local")
    clf = fit_teams_from_video(
        args.source, player_model,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"], ball_conf=det_cfg["ball_conf"],
        stride=args.stride, max_frames=args.max_fit_frames,
    )
    print(f"  fit in {time.perf_counter() - t0:.1f}s")

    # --- 2. Run the pipeline on the chosen frame, with teams ---
    print(f"\n[2/4] Processing frame {args.frame} with team assignment...")
    radar_frame = process_single_frame(args.source, args.frame, PIPELINE,
                                       team_classifier=clf)
    write_jsonl([radar_frame], out_jsonl)
    n_t0 = sum(1 for p in radar_frame.players if p.team == "0")
    n_t1 = sum(1 for p in radar_frame.players if p.team == "1")
    print(f"  Written: {out_jsonl}")
    print(f"  Teams: {n_t0} vs {n_t1}, attacking_team={radar_frame.attacking_team}")

    # --- 3. Into FrozenFrame ---
    print(f"\n[3/4] Loading FrozenFrame via io_radar...")
    frozen = load_radar_jsonl(out_jsonl)
    print(f"  {frozen.n_players} players "
          f"({frozen.attack_mask.sum()} attack / {frozen.defend_mask.sum()} defend), "
          f"carrier={frozen.ball_carrier}")

    # --- 4. footlab analytics on the real frame ---
    print(f"\n[4/4] Computing pitch control + value surface...")
    _, _, centers = pitch.make_grid(cell_size=2.0)
    t0 = time.perf_counter()
    ctrl = pitch_control.pitch_control_surface(frozen, centers)
    val = value_surface.positional_value_surface(centers)
    cost = value_surface.defender_proximity_penalty(frozen, centers)
    combined = val * ctrl - cost
    print(f"  surfaces in {time.perf_counter() - t0:.1f}s "
          f"(mean att control {ctrl.mean():.1%})")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
    viz.plot_surface(frozen, ctrl, ax=axes[0], title="Pitch control (attack)")
    viz.plot_surface(frozen, val, ax=axes[1], title="Positional value (xT)")
    viz.plot_surface(frozen, combined, ax=axes[2],
                     title="Combined: value × control − pressure")
    fig.suptitle(
        f"E2E on real frame {args.frame} — {args.source}\n"
        f"teams {n_t0}v{n_t1}, carrier={frozen.ball_carrier}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = f"data/e2e_frame_{args.frame}_analytics.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_png}")
    print("=" * 64)


if __name__ == "__main__":
    main()
