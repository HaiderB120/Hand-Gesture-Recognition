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
    "new_tab",
    "close_tab",
    "reload",
    "new_window",
    "scroll_up",
    "scroll_down",
]


def do_action(action_name: str) -> None:
    if not HAVE_PYAUTOGUI:
        print(f"[ACTION] {action_name} (pyautogui not available)")
        return

    if action_name == "screenshot":
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_screenshot_{ts}.png"
        pyautogui.screenshot(filename)
        print(f"[ACTION] Saved screenshot: {filename}")
        return

    if action_name == "new_tab":
        pyautogui.hotkey("ctrl", "t")
        print("[ACTION] Opened new tab")
        return

    if action_name == "close_tab":
        pyautogui.hotkey("ctrl", "w")
        print("[ACTION] Closed tab")
        return

    if action_name == "reload":
        pyautogui.hotkey("ctrl", "r")
        print("[ACTION] Reloaded page")
        return

    if action_name == "new_window":
        pyautogui.hotkey("ctrl", "n")
        print("[ACTION] Opened new window")
        return

    if action_name == "scroll_up":
        pyautogui.scroll(500)
        print("[ACTION] Scrolled up")
        return

    if action_name == "scroll_down":
        pyautogui.scroll(-500)
        print("[ACTION] Scrolled down")
        return

    print(f"[ACTION] Unknown action '{action_name}' (no-op)")
