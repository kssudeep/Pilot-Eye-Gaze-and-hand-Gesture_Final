# Pilot-Eye-Gaze-and-hand-Gesture_Final
# Automated Pilot Eye Gaze and Hand Gesture Recognition

## Files Included
- Pilot Eye Gaze and hand Gesture.ipynb  ← Main notebook (all code)
- gesture_model.pth                       ← Trained gesture weights
- gaze_model.pth                          ← Trained gaze weights
- face_landmarker.task                    ← MediaPipe face model
- hand_landmarker.task                    ← MediaPipe hand model
- gesture_data.json                       ← Gesture training data
- gaze_data.npz                           ← Gaze training data

## How to Run

### Quick Demo (pre-trained models)
1. Install dependencies:
   pip install torch torchvision opencv-python mediapipe numpy
       scikit-learn matplotlib seaborn tqdm ipywidgets albumentations

2. Open: Pilot Eye Gaze and hand Gesture.ipynb

3. Run SECTION 1 (Setup) cells first

4. Run SECTION 3 (Model Definitions) cells

5. Jump to SECTION 6 and run the pipeline:
   pipeline = CockpitVisionPipeline()
   pipeline.run(max_frames=300)

### Full Reproduction (train from scratch)
Run all sections in order: 1 → 2 → 3 → 4 → 5 → 6 → 7

### Evaluation Only
Run Section 1, Section 3, then Section 7

## Requirements
- Python 3.10+
- Webcam
- CPU sufficient (GPU optional)
- Jupyter Notebook (classic, not JupyterLab)

## Notes
- MediaPipe 0.10+ uses Tasks API (not mp.solutions)
- Display uses matplotlib inline (not cv2.imshow)
- Restart kernel after pip install in Cell 1
