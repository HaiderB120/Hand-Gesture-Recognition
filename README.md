**Hand Gesture Recognition System**

_A real-time human–computer interaction project using Python, MediaPipe, and OpenCV._

Insert GIF demo here (5–10 seconds showing gesture + on-screen action).
<!-- TODO: Insert demo.gif demonstrating a gesture (pinch, screenshot, drag) with the system responding in real-time. -->

Table of Contents

1. Overview

2. Key Features

3. Motivation

4. System Architecture

5. How It Works

6. Gesture Training Pipeline

7. Tech Stack

8. Installation

9. Run the System

10. Repository Structure

11. Example Gestures

12. Future Improvements

13. Research Documentation

14. Citing This Project

**Overview**
This project implements a real-time hand gesture recognition system that allows users to control their computer using natural hand movements. The system captures hand landmarks using MediaPipe, processes them into normalized feature vectors, classifies gestures based on user-recorded samples, and triggers automations such as pinch-and-drag, screenshot actions, virtual clicking, and text manipulation.

The system runs continuously in the background with low latency, enabling intuitive, touch-less computer interaction.

**Key Features**
Real-time hand tracking using MediaPipe Hands

User-trainable gestures (record ~50 samples to teach the system a new gesture)

Robust gesture classification using normalized landmark features

Gesture-triggered automations:

Pinch → click & drag

Fist → screenshot

Open-hand → release drag

Custom gestures → user-defined actions

Low-latency background operation

Smoothing, stabilization, and debouncing for reliable detection

Scalable design for adding new gestures and behaviors

Insert system diagram here.
<!-- TODO: Insert a block diagram: Webcam → MediaPipe → Feature Extraction → Gesture Classifier → Action Module -->

Motivation

The project explores human–computer interaction (HCI) beyond keyboard and mouse input. Gesture-based interfaces offer:

hands-free control

increased accessibility

natural interaction

integration with productivity workflows

foundations for AR/VR and intelligent assistants

This work demonstrates how lightweight computer vision approaches can create responsive, intuitive interfaces.

System Architecture

Insert architecture figure here (recommended).
<!-- TODO: Add architecture.png showing modules: Capture → Tracking → Normalization → Classification → Automation -->

Core Components

Frame Capture – Reads and preprocesses webcam frames

Hand Landmark Detection – MediaPipe extracts 21 key points

Normalization – Scale & rotation-invariant transformation

Gesture Classification – Matches live landmarks against stored samples

Action Engine – Triggers system actions (drag, screenshot, etc.)

How It Works
1. Landmark Extraction

MediaPipe returns 21 3D coordinates for the hand. Example:

Insert snippet of raw MediaPipe landmarks.
<!-- TODO: Add Python snippet showing MediaPipe landmark output. -->

2. Normalization

To handle distance/size issues, landmarks are converted to:

relative coordinates

wrist-centered vectors

scale-independent values

This ensures gestures remain consistent regardless of camera distance.

3. Feature Vector Creation

The system flattens all relative landmarks into a single feature vector (63 values).

4. Classification Strategy

The system compares live vectors against user-recorded samples (≈50 per gesture).
Distance metrics and multi-frame agreement ensure robustness.

5. Action Execution

Recognized gestures are mapped to actions using PyAutoGUI and custom logic.

Gesture Training Pipeline

This system allows users to teach it new gestures.

Steps

Run training script

Perform gesture for multiple frames

System automatically records ~50 samples

Samples saved as JSON

Classifier loads them during runtime

Insert training screenshot here.
<!-- TODO: Add screenshot of terminal or UI during gesture training. -->

Example training command:

python gesture_train.py --gesture_name pinch


Stored samples example:

Insert a small JSON snippet.
<!-- TODO: Add JSON sample showing landmark vector list. -->

Challenges and Solutions

Developing the system brought practical challenges that required careful iteration. Gesture ambiguity was a persistent issue—similar hand shapes (like loose pinches or partially open hands) produced overlapping landmark patterns. I resolved this by validating gestures over multiple consecutive frames, tuning angle- and distance-based thresholds, and normalizing landmark ratios to remove scale variations caused by hand distance from the camera.

Pinch-and-drag behavior also required refinement. Small tremors and camera noise caused the cursor to jitter or break the drag unintentionally. I addressed this by applying moving-average smoothing and velocity limiting, resulting in smoother cursor movement and more stable dragging.

Real-time performance presented another challenge. Early implementations lagged because some operations were executed redundantly between frames. Moving feature extraction into optimized NumPy operations, reducing MediaPipe calls, and caching non-changing states significantly improved responsiveness and allowed the system to run continuously in the background without affecting overall system performance.

Tech Stack

Python 3.x

MediaPipe

OpenCV

NumPy

PyAutoGUI

Flask (for optional UI endpoints)

YAML / JSON for gesture storage

Installation
git clone https://github.com/HaiderB120/Hand-Gesture-Recognition
cd Hand-Gesture-Recognition
pip install -r requirements.txt

Run the System
Start gesture detection
python gesture_run.py

Train a new gesture
python gesture_train.py --gesture_name fist

Repository Structure
Hand-Gesture-Recognition/
│
├── src/
│   ├── gesture_run.py
│   ├── gesture_train.py
│   ├── utils/
│   │   ├── landmark_processing.py
│   │   ├── smoothing.py
│   │   ├── action_engine.py
│   │   └── classifier.py
│
├── gestures/
│   ├── pinch.json
│   ├── fist.json
│   └── open_hand.json
│
├── tests/
│   ├── test_imports.py
│   ├── test_classifier.py
│   └── ...
│
├── docs/
│   ├── research_report.pdf     <!-- TODO: place your report here -->
│   ├── architecture.png         <!-- TODO -->
│   └── demo.gif                 <!-- TODO -->
│
├── requirements.txt
├── README.md
└── LICENSE                      <!-- TODO: MIT recommended -->


Adjust file paths if yours differ — I can rewrite this to exactly match your repo.

Example Gestures

Insert screenshots or GIFs here for each gesture.
<!-- TODO: Add images showing pinch, open-hand, fist, etc. -->

Pinch → Drag object / rotate 3D models

Fist → Screenshot

Open-hand → Release drag

Custom gestures → User-defined actions

Future Improvements

Add an ML model (SVM or small neural network) for more complex gesture sets

Add dual-hand recognition

Voice activation / JARVIS integration

Chrome extension for gesture-based browser automation

Export gesture sets to share across machines

Research Documentation

Insert or link your research report.
<!-- TODO: Add research_report.pdf to docs/ and link below. -->

Full research write-up:
docs/research_report.pdf

Citing This Project

If you use or reference this project:

Baig, Haider. *Hand Gesture Recognition for Human–Computer Interaction.* 2025. GitHub Repository.
