# tests/test_pinch_ratio.py
import importlib
import numpy as np

def _lm_to_pts(hand_lms, w, h):
    pts = []
    for lm in hand_lms.landmark:
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float32)

def test_pinch_ratio_orders_correctly(tight_pinch_hand, wide_open_hand):
    gr = importlib.import_module("single_gesture_run")

    w, h, hand_lms_tight = tight_pinch_hand
    w2, h2, hand_lms_open = wide_open_hand

    pts_t = _lm_to_pts(hand_lms_tight, w, h)
    pts_o = _lm_to_pts(hand_lms_open, w2, h2)

    r_t = gr.pinch_ratio_from_pts(pts_t)
    r_o = gr.pinch_ratio_from_pts(pts_o)

    assert r_t < r_o, "Tight pinch should have smaller ratio than open hand"
    assert r_t < 0.25 and r_o > 0.25  # rough bounds; adjust if your thresholds differ
