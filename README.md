# AI Gym Tracker: Real-Time Exercise Rep Counter 🏋️‍♂️


A computer vision application that acts as a virtual personal trainer. It uses \*\*MediaPipe\*\* 
for pose estimation and a custom \*\*Machine Learning model\*\* (Random Forest) to detect exercise states 
(e.g., "Up" vs. "Down") and accurately count repetitions in real-time.



## 📹 Demo



## 🧠 How It Works



1\.  \*\*Pose Estimation:\*\* The webcam feed is processed by MediaPipe to detect 33 skeletal landmarks (joints) on the human body.

2\.  \*\*Data Extraction:\*\* The geometric coordinates (x, y, z) of the skeleton are extracted and normalized.

3\.  \*\*Classification:\*\* A trained Machine Learning model (`body\_language.pkl`) analyzes the coordinates to determine the user's current exercise stage (e.g., `curl\_up` or `curl\_down`) with a probability score.

4\.  \*\*Logic \& Counting:\*\* A state machine tracks the transition between stages. A repetition is counted only when a full cycle is completed with high confidence probability.



## 📂 Project Structure



```text

├── app.py                # Main Application: Runs the webcam \& counter

├── train.py              # Trainer Script: Reads CSV -> Trains Model -> Saves .pkl

├── collect\_data.py       # Data Collector: Captures your poses for the dataset

├── exercise\_data.csv     # The dataset containing skeletal coordinates

├── body\_language.pkl     # The trained "Brain" (Model file)

├── requirements.txt      # List of dependencies

└── README.md             # Project Documentation
```
## 🛠️ Installation

Prerequisite: This project is optimized for Python 3.10. 
Create a folder named ai-gym-tracker 
Go to that folder 
```bash 
cd ai-gym-tracker
```
Clone the repository:
```bash
https://github.com/Arjunmkandhan/AI-Powered-Real-Time-Fitness-Trainer.git
```
Install dependencies:

```Bash
pip install -r requirements.txt
```

## 🚀 Usage Guide
This project comes with a pre-trained model for Bicep Curls. You can run it immediately or train it on your own exercises.

Option 1: Run the App (Immediate)
Simply run the main script to start the fitness tracker:

```Bash
python app.py
```
Press 'q' to exit the application.


Option 2: Train Your Own Exercise (From Scratch)
Step 1: Collect Data Run the data collector script to capture your own movements.

```Bash
python collect_data.py
```
Hold the "Up" position and press u to save coordinates.

Hold the "Down" position and press d to save coordinates.

Aim for ~100 samples of each.

Step 2: Train the Model Run the training script to generate a new brain.

```Bash
python train.py
```
This reads exercise_data.csv.

It trains a Random Forest Classifier.

If accuracy > 90%, it saves a new body_language.pkl file.

Step 3: Run the App

```Bash
python app.py
```

## ⚙️ Tech Stack
Language: Python 3.10

Libraries:

mediapipe: For skeletal tracking.

opencv-python: For video processing and UI drawing.

scikit-learn: For the Random Forest classification algorithm.

pandas & numpy: For data structure handling.

## ⚠️ Troubleshooting
Error: AttributeError: module 'mediapipe' has no attribute 'solutions'
Cause: This happens if protobuf is version 4.x or higher.
Fix: Run this command to downgrade protobuf:
```Bash
pip install protobuf==3.20.3
```

### 📜 License
This project is open-source and available under the MIT License.