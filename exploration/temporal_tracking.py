"""Phase 4 — tracking quality metrics over a tracked JSONL.

Reads the multi-frame JSONL from RADAR_VIDEO and measures the tracker's
health — the metrics single-frame analysis cannot see:

1. TRACK LIFETIMES
   Per track ID: how many frames did it persist? A healthy broadcast clip
   has a long tail (players visible for most of the video) and a short head
   (brief spurious tracks). Rendered as a histogram.

2. ID SWITCHES
   A track that *should* be one physical player but changes ID. Exact ID-
   switch counting needs ground truth; our proxy: same-class detections
   whose pitch position jumps implausibly between consecutive frames while
   a new ID appears nearby. We report both raw track counts and the
   "reactivation rate" (new IDs per frame beyond the first).

3. VELOCITY SANITY
   Physics as a BS-detector: footballers don't exceed ~10.5 m/s. The speed
   histogram should die out by ~9-10 m/s; a fat tail beyond means tracking
   or homography errors (positions teleporting), not sprinters.

All metrics come from the JSONL alone — no video, no models. Old runs are
re-analyzable forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Football physics: sustained sprint ~9.5 m/s; absolute peak ~10.5 m/s.
MAX_HUMAN_SPEED_MS = 10.5


@dataclass
class TrackingReport:
    """Summary of tracker health over one JSONL."""

    n_frames: int
    n_tracks: int
    lifetime_mean: float
    lifetime_median: float
    lifetime_max: float
    single_frame_tracks: int        # tracks seen in exactly 1 frame (noise)
    mean_speed_ms: float
    p95_speed_ms: float
    pct_impossible_speeds: float    # % of player-frames above MAX_HUMAN_SPEED_MS
    new_ids_per_frame: float        # mean new track IDs appearing per frame


def load_tracked_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze_tracks(frames: list[dict]) -> tuple[TrackingReport, dict[int, int], np.ndarray]:
    """Compute tracker-health metrics.

    Returns:
        report:    the summary dataclass
        lifetimes: {track_id: n_frames_seen}
        speeds:    all player-frame speeds (m/s) as a flat array
    """
    lifetimes: dict[int, int] = {}
    speeds: list[float] = []
    first_seen_frame: dict[int, int] = {}

    for fi, frame in enumerate(frames):
        for p in frame["players"]:
            tid = p["track_id"]
            lifetimes[tid] = lifetimes.get(tid, 0) + 1
            first_seen_frame.setdefault(tid, fi)
            vx, vy = p["pitch_vxy_ms"]
            speeds.append(float(np.hypot(vx, vy)))

    speeds_arr = np.array(speeds) if speeds else np.zeros(1)
    life_arr = np.array(list(lifetimes.values())) if lifetimes else np.zeros(1)

    n_frames = len(frames)
    # New IDs per frame beyond frame 0 (frame 0 seeds all initial tracks).
    new_ids = [sum(1 for f0 in first_seen_frame.values() if f0 == fi)
               for fi in range(1, n_frames)] or [0]

    report = TrackingReport(
        n_frames=n_frames,
        n_tracks=len(lifetimes),
        lifetime_mean=float(life_arr.mean()),
        lifetime_median=float(np.median(life_arr)),
        lifetime_max=float(life_arr.max()),
        single_frame_tracks=int((life_arr == 1).sum()),
        mean_speed_ms=float(speeds_arr.mean()),
        p95_speed_ms=float(np.percentile(speeds_arr, 95)),
        pct_impossible_speeds=float((speeds_arr > MAX_HUMAN_SPEED_MS).mean() * 100),
        new_ids_per_frame=float(np.mean(new_ids)),
    )
    return report, lifetimes, speeds_arr


def render_tracking_dashboard(
    lifetimes: dict[int, int],
    speeds: np.ndarray,
    report: TrackingReport,
    output_png: str,
    title: str,
) -> str:
    """Three-panel tracker health figure: lifetimes, speeds, summary text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    life_arr = np.array(list(lifetimes.values()))
    axes[0].hist(life_arr, bins=30, color="tab:blue", alpha=0.85)
    axes[0].axvline(np.median(life_arr), color="red", ls="--",
                    label=f"median {np.median(life_arr):.0f}")
    axes[0].set_xlabel("track lifetime (frames)")
    axes[0].set_ylabel("# tracks")
    axes[0].legend()
    axes[0].set_title("Track lifetimes — long tail = persistent IDs")

    axes[1].hist(speeds, bins=50, color="tab:green", alpha=0.85)
    axes[1].axvline(MAX_HUMAN_SPEED_MS, color="red", ls="--",
                    label=f"human max ~{MAX_HUMAN_SPEED_MS} m/s")
    axes[1].set_xlabel("speed (m/s)")
    axes[1].set_ylabel("# player-frames")
    axes[1].legend()
    axes[1].set_title("Velocity sanity — tail past the red line = errors")

    axes[2].axis("off")
    summary = (
        f"frames: {report.n_frames}\n"
        f"tracks: {report.n_tracks}\n"
        f"lifetime mean/median/max: {report.lifetime_mean:.1f} / "
        f"{report.lifetime_median:.0f} / {report.lifetime_max:.0f}\n"
        f"single-frame tracks: {report.single_frame_tracks}\n"
        f"new IDs per frame: {report.new_ids_per_frame:.2f}\n"
        f"speed mean/p95: {report.mean_speed_ms:.2f} / {report.p95_speed_ms:.2f} m/s\n"
        f"impossible speeds (> {MAX_HUMAN_SPEED_MS} m/s): {report.pct_impossible_speeds:.2f}%"
    )
    axes[2].text(0.05, 0.95, summary, va="top", fontsize=11, family="monospace")
    axes[2].set_title("Summary")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def tracking_table_markdown(reports: list[tuple[str, TrackingReport]]) -> str:
    """One row per tracked video — cross-run tracker comparison."""
    header = (
        "| video | frames | tracks | life med | 1-frame tracks | new IDs/frame | "
        "p95 speed | impossible % |\n|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for name, r in reports:
        lines.append(
            f"| {name} | {r.n_frames} | {r.n_tracks} | {r.lifetime_median:.0f} | "
            f"{r.single_frame_tracks} | {r.new_ids_per_frame:.2f} | "
            f"{r.p95_speed_ms:.2f} | {r.pct_impossible_speeds:.2f}% |"
        )
    return "\n".join(lines)
