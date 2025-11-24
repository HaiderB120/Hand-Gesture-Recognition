# src/core/recognizer.py

import time

import cv2
import mediapipe as mp
import numpy as np

from .features import (
    to_feature_vec,
    landmarks_to_pts,
    pinch_ratio_from_pts,
)
from .gesture_db import load_db, build_templates
from .actions import do_action

# ----- Pinch + drag configuration -----
PINCH_ON = 0.18
PINCH_OFF = 0.22
CONFIRM_ON_FRAMES = 3
CONFIRM_OFF_FRAMES = 3
SMOOTH_ALPHA = 0.35
DEADZONE_PX = 4
LOST_RELEASE_FRAMES = 2

# Recognition + cooldown
SIM_THRESHOLD = 0.92
COOLDOWN_SEC = 5.0

# Set up screen size (used for pinch drag)
try:
    import pyautogui

    pyautogui.FAILSAFE = False
    SCREEN_W, SCREEN_H = pyautogui.size()
    HAVE_PYAUTOGUI = True
except Exception:
    SCREEN_W, SCREEN_H = (1920, 1080)
    HAVE_PYAUTOGUI = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors:
      sim = (a·b) / (|a|*|b|)
    Returns 1.0 for identical directions, ~0 for orthogonal.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class GestureRuntime:
    """
    High-level runtime for live gesture recognition + actions + pinch-drag.
    This is basically your v1 gesture_run logic, but refactored into a class.
    """

    def __init__(self,
                 sim_threshold: float = SIM_THRESHOLD,
                 cooldown_sec: float = COOLDOWN_SEC):
        db = load_db()
        self.templates = build_templates(db)
        if not self.templates:
            raise RuntimeError(
                "No gesture templates found. Run v1_gesture_train.py to collect samples."
            )

        self.names = list(self.templates.keys())
        self.sim_threshold = sim_threshold
        self.cooldown_sec = cooldown_sec
        self.last_trigger = {name: 0.0 for name in self.names}

        # Pinch-drag state
        self.pinch_active = False
        self.on_streak = 0
        self.off_streak = 0
        self.lost_frames = 0
        self.ema_x = None
        self.ema_y = None

        print("[Runtime] Loaded gestures:", self.names)

    def _handle_pinch_drag(self, pts, frame):
        """Update pinch drag state and control the mouse if available."""
        ratio = pinch_ratio_from_pts(pts)

        # Decide ON/OFF streaks using hysteresis
        if ratio <= PINCH_ON:
            self.on_streak += 1
            self.off_streak = 0
        elif ratio >= PINCH_OFF:
            self.off_streak += 1
            self.on_streak = 0
        else:
            self.on_streak = 0
            self.off_streak = 0

        # Map index fingertip (id=8) to screen coords
        idx = pts[8]
        nx = np.clip(idx[0] / frame.shape[1], 0.0, 1.0)
        ny = np.clip(idx[1] / frame.shape[0], 0.0, 1.0)
        sx = int(nx * SCREEN_W)
        sy = int(ny * SCREEN_H)

        # EMA smoothing
        if self.ema_x is None:
            self.ema_x, self.ema_y = sx, sy
        else:
            self.ema_x = int((1 - SMOOTH_ALPHA) * self.ema_x + SMOOTH_ALPHA * sx)
            self.ema_y = int((1 - SMOOTH_ALPHA) * self.ema_y + SMOOTH_ALPHA * sy)

        # Activation: OFF -> ON
        if not self.pinch_active and self.on_streak >= CONFIRM_ON_FRAMES:
            if HAVE_PYAUTOGUI:
                pyautogui.moveTo(self.ema_x, self.ema_y, duration=0)
                pyautogui.mouseDown()
            else:
                print("[SIM] mouseDown()")
            self.pinch_active = True
            self.lost_frames = 0

        # While active: move cursor each frame
        if self.pinch_active:
            if (abs(self.ema_x - sx) > DEADZONE_PX or
                    abs(self.ema_y - sy) > DEADZONE_PX):
                if HAVE_PYAUTOGUI:
                    pyautogui.moveTo(self.ema_x, self.ema_y, duration=0)
                else:
                    print(f"[SIM] moveTo({self.ema_x},{self.ema_y})")

        # Deactivation: ON -> OFF
        if self.pinch_active and self.off_streak >= CONFIRM_OFF_FRAMES:
            if HAVE_PYAUTOGUI:
                pyautogui.mouseUp()
            else:
                print("[SIM] mouseUp()")
            self.pinch_active = False
            self.on_streak = self.off_streak = 0

        # Debug overlays
        cv2.putText(frame, f"Pinch ratio: {ratio:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Drag: {'ON' if self.pinch_active else 'OFF'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.pinch_active else (0, 0, 255), 2)

    def run(self):
        """Main webcam loop."""
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
                self.lost_frames = 0  # we see a hand this frame
                hand_lms = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                feat = to_feature_vec(hand_lms, w, h)

                # Compare with each template
                for name, t in self.templates.items():
                    sim = cosine_sim(feat, t["mean"])
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                # Pinch + drag
                pts = landmarks_to_pts(hand_lms, w, h)
                self._handle_pinch_drag(pts, frame)

                # Display best match
                cv2.putText(
                    frame, f"Best: {best_name} ({best_sim:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2
                )

                # Fire action if threshold + cooldown satisfied
                if best_name and best_sim >= self.sim_threshold:
                    now = time.time()
                    if now - self.last_trigger[best_name] >= self.cooldown_sec:
                        action_name = self.templates[best_name]["action"]
                        do_action(action_name)
                        self.last_trigger[best_name] = now
            else:
                cv2.putText(
                    frame, "No hand detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                self.lost_frames += 1
                self.on_streak = 0
                self.off_streak = 0
                if self.pinch_active and self.lost_frames >= LOST_RELEASE_FRAMES:
                    if HAVE_PYAUTOGUI:
                        pyautogui.mouseUp()
                    else:
                        print("[SIM] mouseUp() due to lost hand")
                    self.pinch_active = False

            cv2.imshow("Gesture Runtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
