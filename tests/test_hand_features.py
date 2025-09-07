# tests/test_hand_features.py
import importlib
import numpy as np

def test_to_feature_vec_center_and_scale(unit_square_hand):
    w, h, hand_lms = unit_square_hand
    tr = importlib.import_module("single_gesture_train")  # or gesture_run if the helper lives there

    vec = tr.to_feature_vec(hand_lms, w, h)
    assert vec.shape == (42,)

    # Un-flatten and check wrist approx (0,0) after centering
    pts = vec.reshape(21, 2)
    wrist = pts[0]
    assert np.allclose(wrist, np.array([0.0, 0.0]), atol=1e-5)

    # Scale sanity: distance wrist(0) -> middle_mcp(9) ~ 1.0 after normalization
    d = np.linalg.norm(pts[9] - pts[0])
    assert 0.9 <= d <= 1.1  # allow small numeric drift
