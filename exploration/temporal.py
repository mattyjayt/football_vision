"""Temporal analysis — per-frame stats collector for a whole video.

Phase 1 of the temporal analytics plan (the observability core):
    * detection counts per frame (players / goalkeepers / referees / ball)
    * confidence distributions per frame (all detections)
    * keypoint visibility + homography RMS per frame
    * bounding-box area distribution per frame
    * class balance per frame

Design: the collector walks the video ONCE, running the production models
on each sampled frame, and appends one flat dict per frame to a CSV.
Plots and tables are views over that CSV — you can ask new questions of an
old run without paying inference again.

The `--half` / `--full` switch: half processes the first half of the sampled
frames (fast sanity pass); full walks everything. Combined with --stride you
control the cost/detail trade-off explicitly.
"""

from __future__ import annotations

import csv
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from detection_pipeline.infer import (
    detect_keypoints,
    detect_players_and_ball,
)
from detection_pipeline.transform import compute_homography


@dataclass
class FrameStats:
    """One row of the temporal CSV — everything measured on one frame."""

    frame_id: int
    time_s: float
    n_players: int
    n_goalkeepers: int
    n_referees: int
    n_total_detections: int
    ball_detected: int                 # 0/1
    conf_mean: float
    conf_median: float
    conf_p10: float
    conf_p90: float
    conf_min: float
    conf_max: float
    bbox_area_mean: float              # px^2
    bbox_area_p90: float
    n_keypoints: int                   # detected above pipeline threshold
    homography_ok: int                 # 0/1
    homography_rms_m: float            # 0.0 when skipped

    @classmethod
    def fieldnames(cls) -> list[str]:
        return list(cls.__dataclass_fields__.keys())


@dataclass
class TemporalReport:
    """Summary statistics over a whole run — one row of the video table."""

    source: str
    frames_sampled: int
    fps: float
    avg_players: float
    avg_total_detections: float
    ball_visibility_pct: float
    mean_conf: float
    keypoint_visibility_avg: float     # avg # of 32 landmarks detected
    homography_success_pct: float
    median_homography_rms_m: float     # over successful frames only


def collect_frame_stats(
    source: str,
    player_model: object,
    keypoint_model: object,
    class_map: dict[str, int],
    player_conf: float,
    ball_conf: float,
    keypoint_conf: float,
    homography_cfg: dict,
    stride: int = 5,
    half: bool = False,
    max_frames: int | None = None,
    verbose: bool = True,
) -> list[FrameStats]:
    """Walk the video and measure every sampled frame.

    Args:
        half: process only the first half of the sampled frame list.
        stride: measure every Nth frame (5 ≈ 6 samples/sec at 30 fps).
        max_frames: hard cap on sampled frames (None = no cap).
    """
    cap = cv2.VideoCapture(source)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_ids = list(range(0, max(n_frames, 1), stride))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if half:
        frame_ids = frame_ids[: max(1, len(frame_ids) // 2)]

    if verbose:
        print(f"collect_frame_stats: {len(frame_ids)} frames to sample "
              f"(video has {n_frames}, stride={stride}, half={half})")

    stats: list[FrameStats] = []
    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        players, ball = detect_players_and_ball(
            frame, player_model,
            class_map=class_map, player_conf=player_conf, ball_conf=ball_conf,
        )
        kps = detect_keypoints(frame, keypoint_model, min_conf=keypoint_conf)
        H, rms = compute_homography(
            kps,
            min_keypoints=homography_cfg["min_keypoints"],
            min_confidence=homography_cfg.get("min_keypoint_conf", 0.0),
        )
        rms_reject = homography_cfg.get("rms_reproj_threshold", 2.0)
        hom_ok = int(H is not None and rms <= rms_reject)

        confs = np.array([d.confidence for d in players]) if players else np.array([0.0])
        areas = (
            np.array([(d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]) for d in players])
            if players else np.array([0.0])
        )
        n_by_class = {"player": 0, "goalkeeper": 0, "referee": 0}
        for d in players:
            n_by_class[d.class_name] = n_by_class.get(d.class_name, 0) + 1

        stats.append(FrameStats(
            frame_id=fid,
            time_s=fid / fps,
            n_players=n_by_class["player"],
            n_goalkeepers=n_by_class["goalkeeper"],
            n_referees=n_by_class["referee"],
            n_total_detections=len(players),
            ball_detected=int(ball is not None),
            conf_mean=float(confs.mean()),
            conf_median=float(np.median(confs)),
            conf_p10=float(np.percentile(confs, 10)),
            conf_p90=float(np.percentile(confs, 90)),
            conf_min=float(confs.min()),
            conf_max=float(confs.max()),
            bbox_area_mean=float(areas.mean()),
            bbox_area_p90=float(np.percentile(areas, 90)),
            n_keypoints=kps.n_keypoints,
            homography_ok=hom_ok,
            homography_rms_m=float(rms) if hom_ok else 0.0,
        ))
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(frame_ids)} frames sampled...")

    cap.release()
    return stats


def summarize(source: str, stats: list[FrameStats], fps: float) -> TemporalReport:
    """Aggregate per-frame stats into a one-row video summary."""
    if not stats:
        raise ValueError("No frame stats to summarize")
    players = np.array([s.n_players for s in stats])
    totals = np.array([s.n_total_detections for s in stats])
    confs = np.array([s.conf_mean for s in stats])
    balls = np.array([s.ball_detected for s in stats])
    kps = np.array([s.n_keypoints for s in stats])
    homs = np.array([s.homography_ok for s in stats])
    rms_ok = np.array([s.homography_rms_m for s in stats if s.homography_ok])
    return TemporalReport(
        source=source,
        frames_sampled=len(stats),
        fps=fps,
        avg_players=float(players.mean()),
        avg_total_detections=float(totals.mean()),
        ball_visibility_pct=float(balls.mean() * 100),
        mean_conf=float(confs.mean()),
        keypoint_visibility_avg=float(kps.mean()),
        homography_success_pct=float(homs.mean() * 100),
        median_homography_rms_m=float(np.median(rms_ok)) if len(rms_ok) else 0.0,
    )


def write_csv(stats: list[FrameStats], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FrameStats.fieldnames())
        writer.writeheader()
        for s in stats:
            writer.writerow({k: getattr(s, k) for k in FrameStats.fieldnames()})
    return path


def report_table_markdown(reports: list[TemporalReport]) -> str:
    """One row per video — the cross-video quality comparison table."""
    header = (
        "| video | frames | avg players | avg dets | ball % | mean conf | "
        "avg kpts | H success % | med H RMS (m) |\n|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in reports:
        lines.append(
            f"| {Path(r.source).name} | {r.frames_sampled} | {r.avg_players:.1f} | "
            f"{r.avg_total_detections:.1f} | {r.ball_visibility_pct:.0f}% | "
            f"{r.mean_conf:.2f} | {r.keypoint_visibility_avg:.1f} | "
            f"{r.homography_success_pct:.0f}% | {r.median_homography_rms_m:.2f} |"
        )
    return "\n".join(lines)
