"""
Open the camera
Detect the skeleton
feed the skeleton coordinates to your saved .pkl model
display the exercise name and count reps on the screen .
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

#load the model
with open('body_language.pkl','rb') as f :
    model = pickle.load(f)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#counting
current_stage = None
counter = 0
prediction_probability = 0

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose :
    while cap.isOpened() :
        ret , frame = cap.read()
        if not ret : break

        frame = cv2.flip(frame,1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks :
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            try :
                pose_row = list(np.array([[landmark.x , landmark.y , landmark.z , landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())

                X = pd.DataFrame([pose_row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                prediction_probability = round(body_language_prob[np.argmax(body_language_prob)] , 2)

                if body_language_class == 'curl_down' and prediction_probability > 0.7 :
                    current_stage = "down"
                elif body_language_class == 'curl_up' and current_stage == 'down' and prediction_probability > 0.7 :
                    current_stage = "up"
                    counter += 1
                    print(f"Rep Count : {counter}")

            except Exception as e :
                pass

            cv2.rectangle(image, (0,0) , (250,60) , (245,117,16) , -1)
            # class name
            cv2.putText(image, 'CLASS' , (95,12) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0) , 1 , cv2.LINE_AA)
            cv2.putText(image, body_language_class.split('_')[0] , (99,40) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 , 255) , 2 ,cv2.LINE_AA)

            # probability
            cv2.putText(image, 'PROB' , (15,12) , cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0) , 1, cv2.LINE_AA)
            cv2.putText(image, str(prediction_probability), (10,40) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 2 , cv2.LINE_AA)

            #counter
            cv2.putText(image, 'COUNT', (180,12) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,0) , 1, cv2.LINE_AA)
            cv2.putText(image, str(counter) , (175,40) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255) , 2 , cv2.LINE_AA)

        cv2.imshow("AI Trainer ",image)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

cap.release()
cv2.destroyAllWindows()





