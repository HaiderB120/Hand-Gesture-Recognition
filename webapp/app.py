from flask import Flask, render_template, request, redirect, url_for
import os
import sys
import json

# Add repo root to path so we can import src_version_2
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from src_version_2 import gesture_db, actions  # noqa: E402

app = Flask(__name__)


CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
GESTURES_PATH = os.path.join(BASE_DIR, "gestures.json")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"pinch_drag_enabled": True}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def load_gestures_safe():
    try:
        db = gesture_db.load_db()
    except FileNotFoundError:
        db = {}
    return db


def record_gesture_samples(gesture_name, action_name):
    """
    Opens a webcam window and records samples for a gesture.
    Blocks until user finishes. Uses same feature extraction as v2.
    """
    import cv2
    import mediapipe as mp
    import numpy as np
    from src_version_2.features import to_feature_vec

    db = load_gestures_safe()
    if gesture_name not in db:
        db[gesture_name] = {"action": action_name, "samples": []}
    else:
        db[gesture_name]["action"] = action_name

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Training gesture:", gesture_name)
    print("Action:", action_name)
    print("SPACE: capture sample, ENTER: save and exit, q: quit without saving.\n")

    capture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(
                frame,
                f"Samples: {capture_count}  (SPACE to capture)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 220, 50),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("AeroDesk - Train Gesture", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            print("Quit without saving.")
            break
        elif k == 13:
            print("Saving and exiting...")
            break
        elif k == 32 and results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            feat = to_feature_vec(hand_lms, w, h)
            db[gesture_name]["samples"].append(feat.tolist())
            capture_count += 1
            print(f"Captured sample #{capture_count}")

    cap.release()
    cv2.destroyAllWindows()

    if capture_count > 0:
        gesture_db.save_db(db)
        print(f"Saved {capture_count} samples for gesture '{gesture_name}'.")


@app.route("/")
def index():
    cfg = load_config()
    db = load_gestures_safe()
    gestures_count = len(db)
    pinch_enabled = bool(cfg.get("pinch_drag_enabled", True))
    return render_template(
        "index.html",
        app_name="AeroDesk",
        gestures_count=gestures_count,
        pinch_enabled=pinch_enabled,
    )


@app.route("/map-gesture", methods=["GET", "POST"])
def map_gesture():
    db = load_gestures_safe()
    available_actions = actions.ACTIONS
    message = None

    if request.method == "POST":
        gesture_name = request.form.get("gesture_name", "").strip()
        action_name = request.form.get("action_name", "").strip()

        if not gesture_name or action_name not in available_actions:
            message = "Please enter a gesture name and select a valid action."
        else:
            record_gesture_samples(gesture_name, action_name)
            message = f"Trained gesture '{gesture_name}' for action '{action_name}'."

    return render_template(
        "map_gesture.html",
        app_name="AeroDesk",
        actions=available_actions,
        message=message,
    )


@app.route("/gestures")
def gestures():
    db = load_gestures_safe()
    rows = []
    for name, item in db.items():
        samples = item.get("samples", [])
        rows.append(
            {
                "name": name,
                "action": item.get("action", ""),
                "count": len(samples),
            }
        )
    return render_template(
        "gestures.html",
        app_name="AeroDesk",
        gestures=rows,
    )


@app.route("/delete-gesture/<gesture_name>", methods=["POST"])
def delete_gesture(gesture_name):
    db = load_gestures_safe()
    if gesture_name in db:
        db.pop(gesture_name)
        gesture_db.save_db(db)
    return redirect(url_for("gestures"))


@app.route("/toggle-pinch", methods=["POST"])
def toggle_pinch():
    cfg = load_config()
    current = bool(cfg.get("pinch_drag_enabled", True))
    cfg["pinch_drag_enabled"] = not current
    save_config(cfg)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
