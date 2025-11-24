# src/core/features.py

import numpy as np


def hand_size_metric(pts: np.ndarray) -> float:
    """
    Compute a scale reference for the hand.
    Uses distance from wrist (0) to middle_mcp (9).
    pts: (21, 2) ndarray of pixel coords.
    """
    wrist = pts[0]
    mid_mcp = pts[9]
    d = np.linalg.norm(mid_mcp - wrist)
    return max(float(d), 1e-6)  # avoid division by zero


def to_feature_vec(landmarks, w: int, h: int) -> np.ndarray:
    """
    Convert MediaPipe normalized landmarks to a 42-D feature vector:
      1) (0..1) coords -> pixel coords using frame width/height.
      2) Center at wrist (subtract wrist coord).
      3) Scale by hand size so near/far don’t matter.
      4) Flatten to [x0, y0, x1, y1, ..., x20, y20].
    """
    pts = []
    for lm in landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    # Center at wrist (landmark 0)
    wrist = pts[0].copy()
    pts -= wrist

    # Scale by hand size
    scale = hand_size_metric(pts + wrist)
    pts /= scale

    return pts.reshape(-1)  # (42,)


def landmarks_to_pts(landmarks, w: int, h: int) -> np.ndarray:
    """MediaPipe landmarks -> (21, 2) numpy array of pixel coords."""
    pts = []
    for lm in landmarks.landmark:
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float32)


def pinch_ratio_from_pts(pts: np.ndarray) -> float:
    """
    Scale-invariant pinch ratio = distance(thumb_tip=4, index_tip=8)
    / distance(wrist=0, middle_mcp=9). Lower = tighter pinch.
    """
    pinch_d = np.linalg.norm(pts[4] - pts[8])
    hand_sz = np.linalg.norm(pts[0] - pts[9])
    return float(pinch_d / max(hand_sz, 1e-6))
