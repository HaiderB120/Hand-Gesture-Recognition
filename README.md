**AeroDesk**

AeroDesk is a lightweight hand-gesture control system that lets you perform desktop actions using only your hand in front of a webcam. You can map your own gestures, choose what action they trigger, and use a pinch-and-drag gesture to move windows or interact with the screen.
I also trained a pinch and drag gesture, in which the user is th

This project includes:

- A web interface for training, organizing, and mapping gestures

- A real-time recognition engine

- Custom gesture training using your webcam

- Configurable desktop actions

- Pinch-and-drag cursor control

  
**Gesture Training**
I built a web application called AeroDesk that can allow users to be able to train the model to remember a custom gesture, and then they are able to map that gesture to different computer actions.

<img width="1919" height="965" alt="image" src="https://github.com/user-attachments/assets/e590fb41-bf82-41b6-9eba-533967c6c894" />
The landing page of my web application. Users are able to create a new gesture or view their gesture library.



<img width="1919" height="961" alt="image" src="https://github.com/user-attachments/assets/7e3ea1a0-ab50-4bf2-996b-bc2612bce25e" />
Here the user can give their custom gesture a name, and from the dropdown list select a computer action they want the gesture to be mapped to.


<img width="1917" height="964" alt="image" src="https://github.com/user-attachments/assets/bc372142-8692-40ad-bac0-47e57fa430f3" />
After clicking on "Train Gesture" the webcam opens up, and the user is able to capture samples of their gesture, moving their hands a little across the screen to train the gesture to be able to handle a little variation, approximately 40 samples is recommended. 


<img width="1919" height="967" alt="image" src="https://github.com/user-attachments/assets/b60cb961-2290-44aa-8b68-ceff46f61004" />
This is the gesture library where the user is able to view their stored gestures and are able to delete a stored gesture.


**How To Run The Website**

In GitBash:
- cd Hand-Gesture-Recognition
- python -m venv venv
- source venv/Scripts/activate
- pip install requirements.txt
- python webapp/app.py
- Go To http://127.0.0.1:5000/


**Using The Trained Model**
In GitBash:
- cd Hand-Gesture-Recognition
- python -m venv venv
- source venv/Scripts/activate
- pip install requirements.txt
- python app_run.py

A window will then open where the user is able to view the application running, although they can minimize this and resume their normal activties while it runs in the background.

<img width="630" height="498" alt="image" src="https://github.com/user-attachments/assets/d7094429-3d26-4c7e-b4cf-13e8e17be707" />
The window will say "No Hand Detected" when there are no hands visible


<img width="630" height="512" alt="image" src="https://github.com/user-attachments/assets/641adddc-cae2-44ca-a72d-2c326d1f5b01" />
The model is then able to detect a gesture from the trained gestures. In this case, it detected my open palm.


<img width="637" height="509" alt="image" src="https://github.com/user-attachments/assets/9321402c-b758-4b0d-bee6-323f4864604a" />
It now detects my "peace" sign.



I also included a pinch and drag event in my model, which is included by default, without user training. It works when you bring your thumb finger and index finger close to each other in a pinch method, which then triggers a hold down action, same way if you hold down on a mouse. Then you are able to move that pinch around and the cursor will follow the pinch, still holding down. This way, it is as if they are able to grab objects on the screen and move them around physically with thier hand. It works great when viewing something in 3d, as the users are able to pinch and move their pinch around the screen and move the object as well.
<img width="627" height="514" alt="image" src="https://github.com/user-attachments/assets/f5befbf4-4711-4402-91a9-6100be509ac9" />
This gesture detects the pinch, and lets the user know that drag is on. It will disengage as soon as the user breaks the pinch.



  


