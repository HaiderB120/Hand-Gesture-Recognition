"""
gesture_train.py

Collects labeled gesture samples from your webcam and stores them in gestures.json.
Each sample is a 42-D feature vector (x,y for 21 landmarks), normalized for position and scale.
You choose the gesture name (e.g., "peace") and the action (e.g., "screenshot").
Press SPACE to capture a sample, ENTER to save and exit, q to quit without saving.
"""

import json
import os
import cv2
import numpy as np
import mediapipe as mp

# Location of the on-disk database of gestures and samples.
DATA_PATH = "gestures.json"

# MediaPipe modules for detecting and drawing hand landmarks.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def load_db():
    """
    Load gestures.json if it exists; otherwise return an empty dict.
    Format:
    {
      "gesture_name": {
        "action": "screenshot",
        "samples": [[42 floats], [42 floats], ...]
      },
      ...
    }
    """
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    """Write the in-memory gesture database back to gestures.json (human-readable)."""
    with open(DATA_PATH, "w") as f:
        json.dump(db, f, indent=2)

def hand_size_metric(pts):
    """
    Compute a scale reference for the hand.
    We use the distance from wrist(0) to middle_mcp(9). This is fairly stable & avoids using a bbox.
    pts: (21, 2) ndarray of pixel coords.
    """
    wrist = pts[0]
    mid_mcp = pts[9]
    d = np.linalg.norm(mid_mcp - wrist)
    return max(d, 1e-6)  # avoid division by zero

def to_feature_vec(landmarks, w, h):
    """
    Convert MediaPipe normalized landmarks to a 42-D feature vector:
    1) Convert normalized (0..1) coords -> pixel coords using frame width/height.
    2) Center at wrist (subtract wrist coord).
    3) Scale by "hand size" so near/far don’t matter.
    4) Flatten to [x0, y0, x1, y1, ..., x20, y20].
    """
    pts = []
    for lm in landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    # Center at the wrist (landmark 0). After this, wrist is ~[0,0].
    wrist = pts[0].copy()
    pts -= wrist

    # Scale by a stable measure of hand size.
    # Note: hand_size_metric expects original coordinates for indexing, so we add wrist back temporarily.
    scale = hand_size_metric(pts + wrist)
    pts /= scale

    return pts.reshape(-1)  # shape (42,)

def main():
    # --- Simple CLI to decide where to save samples and what action they map to.
    gesture_name = input("Enter a gesture name: ").strip()
    action = input("Enter the action the gesture will take: ").strip()

    # Load / init the database record for this gesture.
    db = load_db()
    if gesture_name not in db:
        db[gesture_name] = {"action": action, "samples": []}
    else:
        print(f"Appending samples to existing gesture '{gesture_name}' (action: {db[gesture_name]['action']})")

    # Initialize MediaPipe Hands detector/tracker.
    hands = mp_hands.Hands(
        static_image_mode=False,  # stream mode
        max_num_hands=1,          # one-hand training to keep things simple
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Open the default camera (index 0). If your setup needs it, try cv2.CAP_DSHOW on Windows.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("\nControls:")
    print("  SPACE  -> Capture your current gesture")
    print("  ENTER  -> Save & Exit")
    print("  q      -> Quit without saving\n")

    capture_count = 0

    # --- Main capture loop.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Not getting frame input from camera")
            break

        # Mirror the image
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # RGB input for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # UI overlays
        if results.multi_hand_landmarks:
            # Use the first detected hand and draw its skeleton.
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(
                frame, f"Ready to capture (PRESS SPACE). Collected: {capture_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2
            )
        else:
            cv2.putText(
                frame, "No Hand Detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        # Show which gesture/action you're recording for.
        cv2.putText(
            frame, f"Gesture: {gesture_name}  Action: {action}",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )

        cv2.imshow("Train Gesture", frame)

        # Key handling
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Quit without saving.")
            break
        elif k == 13:  # Key value for enter key
            print("Saving and exiting...")
            save_db(db)
            break
        elif k == 32:  # Key value for space key
            if results.multi_hand_landmarks:
                # Convert current hand pose to a feature vector and store it.
                hand_lms = results.multi_hand_landmarks[0]
                feat = to_feature_vec(hand_lms, w, h)
                db[gesture_name]["samples"].append(feat.tolist())  
                capture_count += 1
                print(f"Captured sample #{capture_count}")
            else:
                print("Cannot Capture. No Hand Detected!")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
