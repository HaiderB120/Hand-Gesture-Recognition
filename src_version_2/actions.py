# src_version_2/actions.py

import time

try:
    import pyautogui

    pyautogui.FAILSAFE = False
    HAVE_PYAUTOGUI = True
except Exception:
    HAVE_PYAUTOGUI = False

ACTIONS = [
    "screenshot",
    "copy",
]


def do_action(action_name: str) -> None:
    if action_name == "screenshot":
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_screenshot_{ts}.png"
        if HAVE_PYAUTOGUI:
            pyautogui.screenshot(filename)
            print(f"[ACTION] Saved screenshot: {filename}")
        else:
            print(f"[ACTION] Would save screenshot as: {filename}")
        return

    if action_name == "copy":
        if not HAVE_PYAUTOGUI:
            print("[ACTION] copy (pyautogui not available)")
            return
        pyautogui.hotkey("ctrl", "c")
        print("[ACTION] Copied selection to clipboard")
        return

    print(f"[ACTION] Unknown action '{action_name}' (no-op)")
