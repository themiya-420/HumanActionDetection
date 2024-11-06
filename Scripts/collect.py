import cv2 # type: ignore
import numpy as np # type: ignore
import os
import time
import mediapipe as mp # type: ignore
from IPython.display import display, clear_output # type: ignore
import PIL.Image # type: ignore
from matplotlib import pyplot as plt # type: ignore

#Media Pipe Variables

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Drawing Landmarks on the image feed
def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_IRISES)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# Extracting KeyPoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])






# Folder Setup

DATA_SET = os.path.join('DATA')

actions = np.array(['assault', 'stabbing', 'gun violence'])

no_sequences = 30

sequence_length = 30

# Creation of folders
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_SET, action, str(sequence)))
        except:
            pass




cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
    
                #reading the frames by camera
                ret, frame = cap.read()
            
                #make Detections
                image, results = mediapipe_detection(frame, holistic)
                #print(results)

                # Draw 
                draw_landmarks(image, results)

                # Wait logic between videos

                if frame_num == 0 :
                    cv2.putText(image, 'Starting Collection', (100,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting Frames for {} video {}'.format(action, sequence), (15, 12),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.waitKey(2000)

                else :

                    cv2.putText(image, 'Collecting Frames for {} video {}'.format(action, sequence), (15, 12),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)


                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_SET, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)         
            

    
                #showing the feed from the camera
                cv2.imshow('Live Feed', image)
                #print(results.pose_landmark.landmark)

                #breaking the process
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()


