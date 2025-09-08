**OVERVIEW**

Hand Gesture Recognition program that lets you control your computer with hand movements captured by your webcam. The model can currently
be trained to learn any gesture the user wants, and do a task whenever that gesture is repeated in the future. The program currently uses
the pyautogui library to perform computer tasks such as taking screenshots and copying and pasting text. 

**PURPOSE**
The purpose of this program is to make computer tasks faster for everyday people.
Another future goal of this program is to make doing computer tasks more accessible for people with disabilities, including incorporating
ASL (sign language) into the program.

**RUNNING THE PROGRAM**

**Clone The Repo**
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition

**Install Dependencies**
pip install -r requirements.txt

**Running Currently Trained Gestures**
python gesture_run.py
1. Do a peace sign in front of the camera to take a screenshot of the laptop screen
2. Do a thumbs up to copy selected text.
     *If already inside google, the program will automatically copy selected text, open a new tab, paste the text, and search.

**Train A New Gesture**
python gesture_train.py
1. Enter Gesture Name
2. Enter Action Name
3. Press 'Space' to capture samples
4. Press 'Enter' to save


