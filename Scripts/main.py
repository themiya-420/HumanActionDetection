import cv2 # type: ignore
import numpy as np # type: ignore
import os
import time
import mediapipe as mp # type: ignore
from IPython.display import display, clear_output # type: ignore
import PIL.Image # type: ignore
from matplotlib import pyplot as plt # type: ignore

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
    
        #reading the frames by camera
        ret, frame = cap.read()
    
        #make Detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
    
        #showing the feed from the camera
        cv2.imshow('Live Feed', frame)
    
        #breaking the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)