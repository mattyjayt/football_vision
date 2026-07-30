"""Annotated video renderer — broadcast frame + projected radar, side by side.

The evidence artifact: proof that detection, tracking, and homography work
TOGETHER over time. Reads the tracked JSONL (RADAR_VIDEO output — data, not
re-inference) and renders every frame:

    LEFT:  the broadcast frame with per-player ID + team-colored ellipse at
           the feet (roboflow-style aesthetics: ellipse > box — the feet are
           what the homography projects)
    RIGHT: the top-down pitch with the same players as dots (same colors,
           same IDs), the ball, and trails of recent positions

Same IDs on both sides = you can watch track 14 make a run on the left and
see the dot glide on the right. If an ID jumps or a dot teleports, you SEE
it — that's the point of the artifact.

Usage:
    uv run python scripts/12_annotated_video.py \
        --source data/08fd33_0.mp4 --jsonl data/radar_video_tracked.jsonl \
        --out data/annotated_side_by_side.mp4 [--stride 2] [--max-frames 200]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footlab.pitch import PITCH_LENGTH, PITCH_WIDTH

# Team colors in BGR (frame) — matched on the radar.
TEAM_COLORS = {
    "0": (60, 60, 220),      # red-ish
    "1": (220, 120, 40),     # blue-ish
    "referee": (0, 215, 255),  # gold
    None: (160, 160, 160),   # grey = unassigned
}

# Radar canvas: 100 px per meter → 1050x680 plus margin.
SCALE = 9
MARGIN = 40
RADAR_W = int(PITCH_LENGTH * SCALE) + 2 * MARGIN
RADAR_H = int(PITCH_WIDTH * SCALE) + 2 * MARGIN

TRAIL_SECONDS = 1.0  # length of the position trail behind each dot

# Display orientation: set from PIPELINE["output"]["radar_y_up_positive"] in
# render_video; module-level so pitch_to_px stays a pure function.
_Y_UP_POSITIVE = True


def pitch_to_px(xy: np.ndarray) -> tuple[int, int]:
    """Pitch meters (center origin) → radar canvas pixels."""
    x_px = int(MARGIN + (xy[0] + PITCH_LENGTH / 2) * SCALE)
    y_world = xy[1] if _Y_UP_POSITIVE else -xy[1]
    y_px = int(MARGIN + (PITCH_WIDTH / 2 - y_world) * SCALE)  # +y draws up
    return x_px, y_px


def draw_radar_base() -> np.ndarray:
    """Static pitch lines on the radar canvas (drawn once, reused)."""
    img = np.full((RADAR_H, RADAR_W, 3), (30, 90, 30), dtype=np.uint8)

    def line(x1, y1, x2, y2, color=(255, 255, 255), th=2):
        cv2.line(img, pitch_to_px(np.array([x1, y1])), pitch_to_px(np.array([x2, y2])),
                 color, th)

    hl, hw = PITCH_LENGTH / 2, PITCH_WIDTH / 2
    # boundary + halfway
    for x1, y1, x2, y2 in [
        (-hl, -hw, hl, -hw), (hl, -hw, hl, hw), (hl, hw, -hl, hw), (-hl, hw, -hl, -hw),
        (0, -hw, 0, hw),
    ]:
        line(x1, y1, x2, y2)
    # center circle
    cv2.circle(img, pitch_to_px(np.array([0.0, 0.0])), int(9.15 * SCALE),
               (255, 255, 255), 2)
    cv2.circle(img, pitch_to_px(np.array([0.0, 0.0])), 3, (255, 255, 255), -1)
    # boxes (both ends)
    for sign in (-1, 1):
        gx = sign * hl
        # penalty box 40.32 wide, 16.5 deep; goal box 18.32 wide, 5.5 deep
        line(gx, -20.16, gx - sign * 16.5, -20.16)
        line(gx - sign * 16.5, -20.16, gx - sign * 16.5, 20.16)
        line(gx - sign * 16.5, 20.16, gx, 20.16)
        line(gx, -9.16, gx - sign * 5.5, -9.16)
        line(gx - sign * 5.5, -9.16, gx - sign * 5.5, 9.16)
        line(gx - sign * 5.5, 9.16, gx, 9.16)
        cv2.circle(img, pitch_to_px(np.array([gx - sign * 11.0, 0.0])), 3,
                   (255, 255, 255), -1)
    return img


def render_video(
    source: str,
    jsonl_path: str,
    out_path: str,
    stride: int = 2,
    max_frames: int | None = None,
    trail_seconds: float = TRAIL_SECONDS,
    verbose: bool = True,
) -> str:
    """Render the side-by-side annotated video."""
    from detection_pipeline.config import PIPELINE, KEYPOINT_VERTICES_M
    from detection_pipeline.frames import read_jsonl

    global _Y_UP_POSITIVE
    _Y_UP_POSITIVE = PIPELINE["output"].get("radar_y_up_positive", True)

    radar_frames = {rf.frame_id: rf for rf in read_jsonl(jsonl_path)}
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = fps / stride
    out_w = frame_w + RADAR_W
    out_h = max(frame_h, RADAR_H)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h),
    )

    trails: dict[int, deque] = defaultdict(
        lambda: deque(maxlen=max(2, int(trail_seconds * out_fps)))
    )

    processed = 0
    for fid in sorted(radar_frames):
        if max_frames is not None and processed >= max_frames:
            break
        rf = radar_frames[fid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        H_inv = np.linalg.inv(np.array(rf.homography)) if rf.homography else None
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:frame_h, :frame_w] = frame

        radar = draw_radar_base()

        for p in rf.players:
            color = TEAM_COLORS.get(p.team, TEAM_COLORS[None])
            xy = np.array(p.pitch_xy_m)
            tid = p.track_id

            # LEFT: ellipse at feet via H⁻¹ (pitch → pixel)
            if H_inv is not None:
                pt = xy.reshape(1, 1, 2).astype(np.float64)
                px = cv2.perspectiveTransform(pt, H_inv).reshape(2)
                ex, ey = int(px[0]), int(px[1])
                cv2.ellipse(canvas, (ex, ey), (28, 10), 0, 0, 360, color, 3)
                cv2.putText(canvas, str(tid), (ex - 10, ey - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # RIGHT: dot + trail + ID on radar
            trails[tid].append(xy)
            for k, past in enumerate(trails[tid]):
                alpha = (k + 1) / len(trails[tid])
                r_px = pitch_to_px(past)
                cv2.circle(radar, r_px, max(1, int(2 + 3 * alpha)),
                           tuple(int(c * alpha) for c in color), -1)
            r_px = pitch_to_px(xy)
            cv2.circle(radar, r_px, 6, color, -1)
            cv2.circle(radar, r_px, 6, (255, 255, 255), 1)
            cv2.putText(radar, str(tid), (r_px[0] - 8, r_px[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ball
        if rf.ball is not None:
            b_px = pitch_to_px(np.array(rf.ball["pitch_xy_m"]))
            cv2.circle(radar, b_px, 5, (255, 255, 255), -1)
            cv2.circle(radar, b_px, 5, (0, 0, 0), 1)
            if H_inv is not None:
                pt = np.array(rf.ball["pitch_xy_m"]).reshape(1, 1, 2)
                bp = cv2.perspectiveTransform(pt, H_inv).reshape(2)
                cv2.circle(canvas, (int(bp[0]), int(bp[1])), 8, (255, 255, 255), 2)

        # Keypoint orientation overlay: detected landmarks drawn at their TRUE
        # pitch positions with IDs. If the radar is oriented correctly, each
        # number sits where the broadcast shows that landmark (far touchline
        # landmarks appear at the TOP of the radar).
        if processed == 0 or fid % 50 == 0:  # light overlay, not every frame
            for kp_id in rf.keypoints_used:
                if kp_id in KEYPOINT_VERTICES_M:
                    kx, ky = KEYPOINT_VERTICES_M[kp_id]
                    kpx = pitch_to_px(np.array([kx, ky]))
                    cv2.circle(radar, kpx, 7, (255, 255, 0), 1)
                    cv2.putText(radar, str(kp_id), (kpx[0] - 5, kpx[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # carrier highlight on radar
        if rf.carrier_track_id is not None:
            for p in rf.players:
                if p.track_id == rf.carrier_track_id:
                    c_px = pitch_to_px(np.array(p.pitch_xy_m))
                    cv2.circle(radar, c_px, 12, (0, 255, 0), 2)

        cv2.putText(radar, f"frame {fid}", (MARGIN, RADAR_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        canvas[:RADAR_H, frame_w:frame_w + RADAR_W] = radar
        writer.write(canvas)
        processed += 1
        if verbose and processed % 50 == 0:
            print(f"  {processed} frames rendered...")

    cap.release()
    writer.release()
    print(f"Annotated video: {out_path} ({processed} frames @ {out_fps:.1f} fps)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Side-by-side annotated video")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/annotated_side_by_side.mp4")
    parser.add_argument("--stride", type=int, default=2,
                        help="must match the stride used to produce the JSONL")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    render_video(args.source, args.jsonl, args.out,
                 stride=args.stride, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
