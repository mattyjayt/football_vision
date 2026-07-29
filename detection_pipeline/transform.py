"""Homography computation and pixel→pitch coordinate projection.

Our own implementation (~25 lines of actual logic). No dependency on the
Roboflow ``sports`` package — we only need cv2's findHomography and
perspectiveTransform.

The keypoint→pitch-coordinate mapping comes from ``config.KEYPOINT_VERTICES_M``,
which encodes the canonical 105×68 m pitch in meter units (origin at center).

Reference:
    Hartley, R. & Zisserman, A. (2004). "Multiple View Geometry in Computer
    Vision." Cambridge University Press. Chapter 4 — estimation of 2D
    homographies via DLT (Direct Linear Transform).
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import KEYPOINT_VERTICES_M
from .infer import KeypointSet


def compute_homography(
    keypoints: KeypointSet,
    min_keypoints: int = 4,
    min_confidence: float = 0.0,
) -> tuple[np.ndarray | None, float]:
    """Compute the 3×3 homography H mapping pixel coords → pitch meters.

    Args:
        keypoints: detected keypoints in pixel space.
        min_keypoints: minimum number of corresponding points required.
        min_confidence: minimum confidence threshold for keypoints (0.0 = use all).

    Returns:
        H: 3×3 homography matrix, or None if insufficient keypoints.
        rms_error: root-mean-square reprojection error in meters (0 if H is None).
    """
    # Filter to keypoints we have pitch coordinates for AND meet confidence threshold
    src_points = []  # pixel space
    dst_points = []  # pitch meters
    # Roboflow approach: ignore confidence, mask only undetected points at (0,0) —
    # a low-conf point that is roughly right still constrains least-squares more
    # than an absent point. min_confidence kept as an escape hatch (default 0.0).
    for kp_id, pix_xy in keypoints.points.items():
        if kp_id in KEYPOINT_VERTICES_M:
            conf = keypoints.confidences.get(kp_id, 0.0)
            if conf >= min_confidence and pix_xy[0] > 1 and pix_xy[1] > 1:
                src_points.append(pix_xy)
                dst_points.append(np.array(KEYPOINT_VERTICES_M[kp_id]))

    if len(src_points) < min_keypoints:
        return None, 0.0

    src = np.array(src_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(dst_points, dtype=np.float64).reshape(-1, 1, 2)

    # Use all keypoints (no RANSAC) — we have high-confidence detections from
    # a good model, so outliers are rare. RANSAC can fit collinear subsets.
    H, _ = cv2.findHomography(src, dst, 0)  # 0 = use all points

    if H is None:
        return None, 0.0

    # Compute RMS reprojection error on all points
    projected = cv2.perspectiveTransform(src, H)
    errors = np.linalg.norm(dst - projected, axis=2).flatten()
    rms = float(np.sqrt(np.mean(errors ** 2)))

    # Consistency check: if RMS error is very high (>5m), the keypoints are
    # likely misidentified by the model (e.g., left/right confusion)
    if rms > 5.0:
        print(f"  WARNING: homography RMS error {rms:.1f}m is very high — "
              f"keypoints may be misidentified")
        return None, rms

    return H, rms


def project_points(
    points_pix: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Project pixel coordinates to pitch coordinates via homography H.

    Args:
        points_pix: (N, 2) or (2,) pixel coordinates.
        H: 3×3 homography from compute_homography.

    Returns:
        (N, 2) or (2,) pitch coordinates in meters.
    """
    pts = np.asarray(points_pix, dtype=np.float64)
    single = pts.ndim == 1
    pts_2d = np.atleast_2d(pts).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts_2d, H)
    result = projected.reshape(-1, 2)
    return result[0] if single else result


def inverse_homography(H: np.ndarray) -> np.ndarray:
    """Return H⁻¹ for pitch→pixel projection (useful for visualization)."""
    return np.linalg.inv(H)
