"""CLI runner: temporal analysis of a video (Phase 1 — observability core).

    uv run python exploration/run_temporal.py --source data/08fd33_0.mp4 --half
    uv run python exploration/run_temporal.py --source data/08fd33_0.mp4 --full --stride 5

Outputs (data/exploration/temporal/<video_stem>/):
    stats.csv          — one row per sampled frame (the raw material)
    dashboard.png      — 5-panel observability dashboard
    dashboard.html     — interactive timeline (scrub + cross-reference)
    report.md          — one-row summary table for this video
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_pipeline.config import PIPELINE
from detection_pipeline.infer import load_keypoint_model, load_player_model
from exploration.temporal import (
    collect_frame_stats, report_table_markdown, summarize, write_csv,
)
from exploration.temporal_plots import render_dashboard, render_interactive


def main():
    parser = argparse.ArgumentParser(description="Temporal CV observability (Phase 1)")
    parser.add_argument("--source", type=str, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--half", action="store_true",
                       help="process first half of sampled frames (fast pass)")
    scope.add_argument("--full", action="store_true",
                       help="process all sampled frames (default)")
    parser.add_argument("--stride", type=int, default=5,
                        help="sample every Nth frame (default 5)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    det_cfg = PIPELINE["detector"]
    hom_cfg = PIPELINE["homography"]

    player_model = load_player_model(det_cfg["player_model_path"], model_type="local")
    kp_type = det_cfg.get("keypoint_model_type", "roboflow")
    keypoint_model = (
        load_keypoint_model(det_cfg["keypoint_model_path"], model_type="local")
        if kp_type == "local"
        else load_keypoint_model(det_cfg["keypoint_model_id"])
    )

    stats = collect_frame_stats(
        args.source, player_model, keypoint_model,
        class_map=det_cfg["class_map"],
        player_conf=det_cfg["player_conf"], ball_conf=det_cfg["ball_conf"],
        keypoint_conf=det_cfg["keypoint_conf"],
        homography_cfg=hom_cfg,
        stride=args.stride, half=args.half, max_frames=args.max_frames,
    )

    out_dir = Path("data/exploration/temporal") / Path(args.source).stem
    csv_path = write_csv(stats, out_dir / "stats.csv")

    from exploration.temporal_plots import load_csv
    rows = load_csv(csv_path)
    title = f"Temporal observability — {Path(args.source).name} ({len(stats)} frames)"
    png = render_dashboard(rows, str(out_dir / "dashboard.png"), title)
    html = render_interactive(rows, str(out_dir / "dashboard.html"), title)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    report = summarize(args.source, stats, fps)
    table = report_table_markdown([report])
    (out_dir / "report.md").write_text(
        f"# Temporal report — {args.source}\n\n{table}\n"
    )

    print(f"\n{table}\n")
    print(f"CSV:       {csv_path}")
    print(f"Dashboard: {png}")
    print(f"HTML:      {html}")
    print(f"Report:    {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
