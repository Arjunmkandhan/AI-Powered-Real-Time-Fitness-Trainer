import cv2
import csv
import os
import numpy as np
import mediapipe as mp



DATA_FILE = "exercise_data.csv"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def setup_csv() :
    if not os.path.exists(DATA_FILE) :
        landmarks = ['label']
        for i in range(33) : #media pipe has 33 landmarks
            landmarks += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]

        with open(DATA_FILE, mode = 'w' , newline = '') as f :
            csv_writer = csv.writer(f,delimiter = ',',quotechar = '"' , quoting = csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

setup_csv()
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose :
    while cap.isOpened():
        ret , frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame,1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks :
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pose_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())

            cv2.putText(image,"Press 'u' for Curl UP, 'd' for Curl DOWN",(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('u') :
                row = ['curl_up'] + pose_row
                with open(DATA_FILE, mode = 'a', newline = '') as f :
                    csv_writer = csv.writer(f,delimiter = ',',quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
                print("Saved: Curl Up")
            elif key == ord('d') :
                row = ['curl_down'] + pose_row
                with open(DATA_FILE , mode = 'a',newline = '') as f :
                    csv_writer = csv.writer(f,delimiter = ',',quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
                print("Saved: Curl Down")

        cv2.imshow("Data collector",image)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
cap.release()
cv2.destroyAllWindows()

