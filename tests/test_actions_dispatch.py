# tests/test_actions_dispatch.py
import importlib

def test_screenshot_action_calls_pyautogui(fake_pyautogui):
    gr = importlib.import_module("single_gesture_run")
    gr.do_action("screenshot")
    # Ensure a screenshot event was recorded
    assert any(e[0] == "screenshot" for e in fake_pyautogui.events)

def test_copy_action_calls_hotkey(fake_pyautogui, monkeypatch):
    gr = importlib.import_module("single_gesture_run")
    # Provide a minimal branch in do_action for "copy" if not present yet
    if "copy" not in [ "screenshot", "copy" ]:        pass
    try:
        gr.do_action("copy")
        assert any(e[0] == "hotkey" and ("ctrl","c") == e[1] for e in fake_pyautogui.events)
    except AttributeError:
        assert False, "Implement do_action('copy') to send Ctrl+C via pyautogui.hotkey"
