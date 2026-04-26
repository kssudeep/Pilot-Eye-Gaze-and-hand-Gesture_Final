Automated Pilot Eye Gaze and Hand Gesture Recognition for Cockpit Assistance
A real-time computer vision system that monitors pilot attention through eye gaze tracking and enables touchless avionics interaction through hand gesture recognition.
________________________________________
Project Overview
This system implements two parallel deep learning pipelines:
•	Gaze Estimation: Multi-input ResNet-18 CNN → 0.34° mean angular error, 97.2% directional accuracy
•	Gesture Recognition: Spatial Encoder + Bidirectional LSTM → 94.2% accuracy across 20 gesture classes
•	Real-Time Pipeline: 125.6ms mean latency on CPU, meeting the safety-critical <150ms requirement
________________________________________
Repository Structure
cockpit_vision/
│
├── main_pipeline.ipynb          # Main real-time inference pipeline
├── gesture_training.ipynb       # Gesture model data collection + training
├── gaze_training.ipynb          # Gaze model data collection + training
├── evaluation.ipynb             # Full evaluation script + visualizations
│
├── gesture_model.pth            # Trained gesture model weights (94.2%)
├── gaze_model.pth               # Trained gaze model weights (0.34°)
│
├── face_landmarker.task         # MediaPipe face landmark model
├── hand_landmarker.task         # MediaPipe hand landmark model
│
├── gesture_data.json            # Collected gesture training data
├── gaze_data.npz                # Collected gaze training data
│
├── gesture_confusion_matrix.png # Evaluation visualization
├── gesture_f1_scores.png        # Evaluation visualization
├── gaze_evaluation.png          # Evaluation visualization
├── gaze_direction_confusion.png # Evaluation visualization
├── latency_profile.png          # Evaluation visualization
│
└── README.md                    # This file
________________________________________
Large File Downloads (Google Drive)
Due to GitHub's 25MB file size limit, large files are hosted on Google Drive.
Download these files and place them in the same folder as the notebook before running.
File	Size	Link
gaze_data.npz	258.6 MB	- https://drive.google.com/file/d/1nXDKSnHjy2E4m_lPWz1_2mISEYdWvHOi/view?usp=drive_link

gaze_model.pth	132.9 MB	- https://drive.google.com/file/d/1hBT3cKWfjduld_kKe2KmqZTkoFj1GG1o/view?usp=drive_link

gesture_data.json	80.3 MB - https://drive.google.com/file/d/1C1rMzSjnode0a7y_LwHuO337dSZBHVuz/view?usp=drive_link

Note: gesture_model.pth, face_landmarker.task and hand_landmarker.task are included directly in the GitHub repository.
________________________________________
Hardware
•	Laptop or desktop with webcam
•	CPU sufficient (GPU optional but recommended for faster training)
Software
•	Python 3.10+
•	Anaconda or Miniconda (recommended)
Python Dependencies
pip install torch torchvision opencv-python mediapipe numpy \
            scikit-learn matplotlib seaborn tqdm ipywidgets albumentations
Or install all at once inside Jupyter:
import sys
!{sys.executable} -m pip install torch torchvision opencv-python mediapipe \
    numpy scikit-learn matplotlib seaborn tqdm ipywidgets albumentations
________________________________________
Setup Instructions
Step 1 — Clone or unzip the project
unzip cockpit_vision.zip
cd cockpit_vision
Step 2 — Create a Conda environment (recommended)
conda create -n cockpit_env python=3.10
conda activate cockpit_env
pip install torch torchvision opencv-python mediapipe numpy \
            scikit-learn matplotlib seaborn tqdm ipywidgets albumentations
Step 3 — Download MediaPipe model files
The .task files are included in the zip. If missing, run this in a notebook cell:
import urllib.request
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "face_landmarker.task"
)
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "hand_landmarker.task"
)
Step 4 — Launch Jupyter Notebook
jupyter notebook
________________________________________
How to Run
Section 1 — Run the pre-trained real-time system (recommended)
1.	Open main_pipeline.ipynb
2.	Run all cells in order (Cell 1 through Cell 9)
3.	The system will open your webcam and display live gaze + gesture detection
Note: Pre-trained weights (gesture_model.pth, gaze_model.pth) are included. No training required to run the demo.
Section 2— Train from scratch
Gesture model:
1.	Open gesture_training.ipynb
2.	Run Cell T1 (imports)
3.	Run Cell T2 (data collection) — follow on-screen prompts, press SPACE to record each gesture
4.	Run Cells T3–T5 (train) — saves gesture_model.pth
Section 3 - Gaze model:
1.	Open gaze_training.ipynb
2.	Run Cell G1–G2 (imports + helpers)
3.	Run Cell G4 (data collection) — look at each screen region and press SPACE
4.	Run Cells G5–G8 (train) — saves gaze_model.pth
Section 4— Run the evaluation
1.	Open evaluation.ipynb
2.	Run all cells (E1–E7)
3.	Evaluation plots are saved as .png files in the project directory
________________________________________
Notebook Cell Guide (main_pipeline.ipynb)
Cell	Contents
Cell 1	Install dependencies
Cell 2	Imports
Cell 3	Download MediaPipe models
Cell 4	Config (gesture classes, parameters)
Cell 5	GazeEstimator model definition
Cell 6	GestureRecognizer model definition
Cell 7	Preprocessing helpers + Kalman filter
Cell 8	CockpitVisionPipeline class
Cell 9	Run pipeline
________________________________________
Controls
Key	Action
Run Cell 9	Start the pipeline
Interrupt kernel	Stop the pipeline
max_frames=300	Processes ~10 seconds at 30FPS (increase as needed)
________________________________________
Model Architecture Summary
Gaze Estimator
•	Input: Left eye (64×64) + Right eye (64×64) + Face (224×224) + Head pose (3D)
•	Backbone: ResNet-18 (pretrained, partially frozen)
•	Fusion: 1152-dim concatenation → FC(512) → FC(256) → FC(2)
•	Output: Pitch, yaw in radians
•	Smoothing: Kalman filter (Q=0.01, R=0.1)
Gesture Recognizer
•	Input: 32-frame sequence of 21-point 3D MediaPipe hand landmarks
•	Spatial encoder: FC(63→128→256→512) with LayerNorm
•	Temporal encoder: Bidirectional LSTM (256 hidden, 2 layers)
•	Attention: Learned temporal attention over 32 frames
•	Output: Softmax over 20 gesture classes
•	Post-processing: Majority voting over 5-frame window
________________________________________
Performance Results
Metric	Result	Target
Gesture Accuracy	94.2%	>90% ✅
Gaze Angular Error	0.34°	<5° ✅
Gaze Direction Accuracy	97.2%	—
Mean Latency	125.6ms	<150ms ✅
95th Percentile Latency	149.3ms	<150ms ✅
________________________________________
Troubleshooting
ModuleNotFoundError: No module named 'cv2'
import sys
!{sys.executable} -m pip install opencv-python
AttributeError: module 'mediapipe' has no attribute 'solutions' This project uses MediaPipe 0.10+ Tasks API. Ensure mediapipe ≥ 0.10.9 is installed.
Camera not accessible Try cv2.VideoCapture(1) instead of cv2.VideoCapture(0) if you have multiple cameras.
Loading widget... with no video The project uses matplotlib for display. Ensure ipywidgets is installed and run:
jupyter nbextension enable --py widgetsnbextension --sys-prefix
