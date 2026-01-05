\# AI Gym Tracker: Real-Time Exercise Rep Counter 🏋️‍♂️



!\[Python](https://img.shields.io/badge/Python-3.10-blue) !\[OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green) !\[MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange) !\[Scikit-Learn](https://img.shields.io/badge/Sklearn-Random%20Forest-yellow)



A computer vision application that acts as a virtual personal trainer. It uses \*\*MediaPipe\*\* for pose estimation and a custom \*\*Machine Learning model\*\* (Random Forest) to detect exercise states (e.g., "Up" vs. "Down") and accurately count repetitions in real-time.



\## 📹 Demo

\*(Place a screenshot or GIF of your project here to show it in action)\*



\## 🧠 How It Works



1\.  \*\*Pose Estimation:\*\* The webcam feed is processed by MediaPipe to detect 33 skeletal landmarks (joints) on the human body.

2\.  \*\*Data Extraction:\*\* The geometric coordinates (x, y, z) of the skeleton are extracted and normalized.

3\.  \*\*Classification:\*\* A trained Machine Learning model (`body\_language.pkl`) analyzes the coordinates to determine the user's current exercise stage (e.g., `curl\_up` or `curl\_down`) with a probability score.

4\.  \*\*Logic \& Counting:\*\* A state machine tracks the transition between stages. A repetition is counted only when a full cycle is completed with high confidence probability.



\## 📂 Project Structure



```text

├── app.py                # Main Application: Runs the webcam \& counter

├── train.py              # Trainer Script: Reads CSV -> Trains Model -> Saves .pkl

├── collect\_data.py       # Data Collector: Captures your poses for the dataset

├── exercise\_data.csv     # The dataset containing skeletal coordinates

├── body\_language.pkl     # The trained "Brain" (Model file)

├── requirements.txt      # List of dependencies

└── README.md             # Project Documentation

🛠️ Installation

Prerequisite: This project is optimized for Python 3.10.



Clone the repository:



```bash



git clone \[https://github.com/your-username/ai-gym-tracker.git](https://github.com/arjunmkandhan/ai-gym-tracker.git)

cd ai-gym-tracker

