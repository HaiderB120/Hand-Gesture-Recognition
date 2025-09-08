"""
gesture_run.py

Loads gestures.json, builds a template for each gesture by averaging its samples,
then runs live recognition from the webcam. For each frame, it computes a 42-D
feature vector (same normalization as training) and compares it to each template
using cosine similarity. If the best match exceeds SIM_THRESHOLD and cooldown
has passed, the gesture's action is triggered (e.g., screenshot).
"""

import time
import json
import os
import cv2
import numpy as np
import mediapipe as mp

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # prevent FailSafeException if mouse hits corners
    HAVE_PYAUTOGUI = True
except Exception:
    HAVE_PYAUTOGUI = False

# ===== Pinch+Drag config =====
PINCH_ON = 0.18          # ratio threshold to turn ON pinch
PINCH_OFF = 0.22         # ratio threshold to turn OFF pinch (hysteresis)
CONFIRM_ON_FRAMES = 3     # require N consecutive frames to confirm ON
CONFIRM_OFF_FRAMES = 3    # require N consecutive frames to confirm OFF
SMOOTH_ALPHA = 0.35       # EMA smoothing for cursor (0..1) higher = snappier
DEADZONE_PX = 4           # ignore tiny movements
LOST_RELEASE_FRAMES = 2   # if hand lost, release drag after N frames

# Screen size
if HAVE_PYAUTOGUI:
    SCREEN_W, SCREEN_H = pyautogui.size()
else:
    SCREEN_W, SCREEN_H = (1920, 1080)  # fallback for simulation

DATA_PATH = "gestures.json"

# 92% match with stored gesture 
SIM_THRESHOLD = 0.92

# 5 second cooldown between repeating action for the same gesture
COOLDOWN_SEC = 5.0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load the gesture database if it exists
def load_db():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def hand_size_metric(pts):
    """
    Same scale metric used in training: wrist(0) to middle_mcp(9) distance.
    Keep train/test processing identical for best performance.
    """
    wrist = pts[0]
    mid_mcp = pts[9]
    d = np.linalg.norm(mid_mcp - wrist)
    return max(d, 1e-6)

def to_feature_vec(landmarks, w, h):
    """
    Identical to training: convert to pixel coords, center at wrist, scale by hand size,
    flatten to 42 numbers.
    """
    pts = []
    for lm in landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    wrist = pts[0].copy()
    pts -= wrist

    scale = hand_size_metric(pts + wrist)
    pts /= scale

    return pts.reshape(-1)

def cosine_sim(a, b):
    """
    Cosine similarity between two vectors:
      sim = (a·b) / (|a|*|b|)
    Returns 1.0 for identical directions, ~0 for orthogonal.
    """
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def landmarks_to_pts(landmarks, w, h):
    """MediaPipe landmarks -> (21,2) numpy array of pixel coords."""
    pts = []
    for lm in landmarks.landmark:
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float32)

def pinch_ratio_from_pts(pts):
    """
    Scale-invariant pinch ratio = distance(thumb_tip=4, index_tip=8) / distance(wrist=0, middle_mcp=9).
    Lower = tighter pinch.
    """
    pinch_d = np.linalg.norm(pts[4] - pts[8])
    hand_sz = np.linalg.norm(pts[0] - pts[9])
    return float(pinch_d / max(hand_sz, 1e-6))

def build_templates(db):
    """
    Average all samples for each gesture to make a single template vector.
    Also L2-normalize the mean vector for stable cosine similarity.
    Returns: { name: {'action': str, 'mean': np.ndarray}, ... }
    """
    templates = {}
    for name, item in db.items():
        samples = np.array(item.get("samples", []), dtype=np.float32)
        if len(samples) == 0:
            # No samples -> skip this gesture
            continue

        # Mean over sample dimension -> a single 42-D template.
        mean_vec = samples.mean(axis=0)

        # Normalize to unit length to make cosine similarity purely angular.
        n = np.linalg.norm(mean_vec)
        if n > 1e-6:
            mean_vec = mean_vec / n

        templates[name] = {
            "action": item.get("action", "screenshot"),
            "mean": mean_vec
        }
    return templates

# Execute the action for recognized gesture. *ALL NEW GESTURES ADDED HERE
def do_action(action_name):

    if action_name == "screenshot":
        ts = time.strftime("%Y%m%d_%H%M%S") # Timestamp variable
        filename = f"gesture_screenshot_{ts}.png"
        if HAVE_PYAUTOGUI:
            pyautogui.screenshot(filename)
            print(f"Saved screenshot: {filename}")
        else:
            print(f"Saved screenshot as {filename}")
    else:
        print(f"Unknown action '{action_name}' (no-op)")

    if action_name == "copy":
        pyautogui.hotkey('ctrl','t')
        pyautogui.hotkey('y','enter')
        print("Text Copied Successfully")

    


def main():
    # Load trained database and build templates.
    db = load_db()
    templates = build_templates(db)
    if not templates:
        print("No templates found. Please run gesture_train.py first and collect samples.")
        return

    names = list(templates.keys())
    print("Loaded gestures:", names)

    # Initialize MediaPipe Hands for live tracking.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Open camera.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # One cooldown timer per gesture so different gestures can still fire independently.
    last_trigger = {name: 0.0 for name in names}

        # ===== Pinch runtime state =====
    pinch_active = False
    on_streak = 0
    off_streak = 0
    lost_frames = 0

    # EMA smoothing state for cursor
    ema_x = None
    ema_y = None

    # Live loop.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        best_name = None
        best_sim = 0.0

        if results.multi_hand_landmarks:
            # Use the first detected hand.
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Compute feature vector for this frame.
            feat = to_feature_vec(hand_lms, w, h)

            # Compare with each gesture template.
            for name, t in templates.items():
                sim = cosine_sim(feat, t["mean"])
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            
                                # --- PINCH + DRAG LOGIC ---
            # Convert landmarks to pixel points and compute pinch ratio
            pts = landmarks_to_pts(hand_lms, w, h)
            ratio = pinch_ratio_from_pts(pts)

            # Decide ON/OFF streaks using hysteresis band
            if ratio <= PINCH_ON:
                on_streak += 1
                off_streak = 0
            elif ratio >= PINCH_OFF:
                off_streak += 1
                on_streak = 0
            else:
                # In the gray zone; don't accumulate either streak
                on_streak = 0
                off_streak = 0

            # Map index fingertip (id=8) to screen coords
            idx = pts[8]
            nx = np.clip(idx[0] / w, 0.0, 1.0)
            ny = np.clip(idx[1] / h, 0.0, 1.0)
            sx = int(nx * SCREEN_W)
            sy = int(ny * SCREEN_H)

            # EMA smoothing
            if ema_x is None:
                ema_x, ema_y = sx, sy
            else:
                ema_x = int((1 - SMOOTH_ALPHA) * ema_x + SMOOTH_ALPHA * sx)
                ema_y = int((1 - SMOOTH_ALPHA) * ema_y + SMOOTH_ALPHA * sy)

            # Activation: OFF -> ON after CONFIRM_ON_FRAMES
            if not pinch_active and on_streak >= CONFIRM_ON_FRAMES:
                # Seed cursor to current smoothed target to avoid jump
                if HAVE_PYAUTOGUI:
                    pyautogui.moveTo(ema_x, ema_y, duration=0)
                    pyautogui.mouseDown()  # start dragging
                else:
                    print("[SIM] mouseDown()")
                pinch_active = True
                lost_frames = 0  # reset lost counter

            # While active: move cursor each frame (with deadzone)
            if pinch_active:
                if abs(ema_x - sx) > DEADZONE_PX or abs(ema_y - sy) > DEADZONE_PX:
                    if HAVE_PYAUTOGUI:
                        pyautogui.moveTo(ema_x, ema_y, duration=0)
                    else:
                        print(f"[SIM] moveTo({ema_x},{ema_y})")

            # Deactivation: ON -> OFF after CONFIRM_OFF_FRAMES
            if pinch_active and off_streak >= CONFIRM_OFF_FRAMES:
                if HAVE_PYAUTOGUI:
                    pyautogui.mouseUp()
                else:
                    print("[SIM] mouseUp()")
                pinch_active = False
                on_streak = off_streak = 0

            # Overlay some debug text so you can see what's happening
            cv2.putText(frame, f"Pinch ratio: {ratio:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Drag: {'ON' if pinch_active else 'OFF'}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if pinch_active else (0, 0, 255), 2)


            # Display best match and score for transparency/debugging.
            cv2.putText(
                frame, f"Best: {best_name} ({best_sim:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2
            )

            # Fire if above threshold and cooldown has passed.
            if best_name and best_sim >= SIM_THRESHOLD:
                now = time.time()
                if now - last_trigger[best_name] >= COOLDOWN_SEC:
                    do_action(templates[best_name]["action"])
                    last_trigger[best_name] = now
        else:
            cv2.putText(
                frame, "No hand detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            # No hand detected this frame
            lost_frames += 1
            on_streak = 0
            off_streak = 0
            if pinch_active and lost_frames >= LOST_RELEASE_FRAMES:
                # Safety: release the mouse if we lose the hand briefly
                if HAVE_PYAUTOGUI:
                    pyautogui.mouseUp()
                else:
                    print("[SIM] mouseUp() due to lost hand")
                pinch_active = False



        cv2.imshow("Gesture Run", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
