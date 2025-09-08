**WEEK 1**
Progress made in hand_tracking.py. Was able to connect camera and set up basic hand detection using MediaPipe.
Program detects hand from other body parts, and draws line between different parts of hand.
Text output in window whenever hand is detected.

**WEEK 2**
This week focused on making a training model.
gesture_training.py asks user to enter a name for a gesture and then peform that gesture once the camera opens.
The user then presses enter, then spacebar to save the gesture. The model then stores all hand coordinates in gestures.json.
In gesture_run.py, I was able to use the pyautogui library to perform different computer tasks whenever a saved gesture is performed.

1st Task: Screenshot
Update: Successful
Was able to successfully take a screenshot upon showing a peace sign in front of camera.

2nd Task: Copy Text, Paste it in google, and search
Update: Successful
Was able to use pyautogui's hotkey method to press 'ctrl', 'c'(copy), 't'(to open new tab), 'v'(paste), 'enter'(to search)
whenever a thumbs-up gesture is performed in front of camera


