## Football Detection

# 📖 Project Overview
This project implements a computer vision system for analyzing football (soccer) videos using object detection and tracking. It detects key elements like the ball, players, goalkeepers, and referees in video frames. The model is trained using YOLO (You Only Look Once) from Ultralytics and applied to track objects across video sequences. The system processes input videos, annotates detections, and outputs an analyzed video with bounding boxes and tracks.
Ideal for sports analytics, such as player movement tracking, ball possession analysis, or referee decision support.

# ✨ Features

Object Detection: Identifies ball, goalkeeper, player, and referee in football videos.
Object Tracking: Tracks detected objects across frames for consistent identification.
Video Processing: Reads input videos, applies detections/tracking, and saves annotated output.
Modular Design: Separate notebooks for training and inference for easy extension.
Real-Time Potential: Optimized for GPU acceleration with YOLO.


# 📊 Dataset
The model is trained on the football-players-detection dataset from Roboflow (Workspace: roboflow-jvuqo, Project: football-players-detection-3zvbc, Version: 1).

Total Images: 663 (Train: 612, Validation: 38, Test: 13).
Classes: ball, goalkeeper, player, referee.
Labels Distribution: Not explicitly detailed; inferred from training logs (e.g., balanced across classes with focus on players).
Preprocessing: None applied.
Augmentation:

Outputs per training example: 3
Horizontal flip
Saturation adjustment: -25% to +25%
Brightness adjustment: -20% to +20%


Usage Notes: Dataset focuses on football match scenes; suitable for object detection in sports videos.
License: Not specified in dataset details (check Roboflow for terms; often CC BY 4.0 for public datasets).

Training results (after 80 epochs): mAP50-95 ~0.600, with high precision for players (~0.962) and referees (~0.896).

# 🏗️ Project Structure
textfootball-video-analysis/
├── Train_Data_Robflow.ipynb     # Dataset download, model training with YOLO
├── Football_Analysis.ipynb      # Video processing, detection, tracking, and annotation
├── models/
│   └── best.pt                  # Trained YOLO model (generated during training)
├── output_videos/
│   └── output.avi               # Annotated output video (generated during analysis)
├── README.md                    # Project documentation
└── LICENSE                      # MIT License

🔧 Design and Architecture
1. Data Handling

Video Reading: Uses OpenCV to read frames from input videos (e.g., 'CV_Task.mkv').
Dataset Integration: Downloads Roboflow dataset and prepares it for YOLO training.

2. Model Architecture

YOLO Model: Ultralytics YOLO11l (pre-trained on COCO, fine-tuned on football dataset).

Layers: 190 (fused), 25M parameters, 86.6 GFLOPs.
Classes: 4 (ball, goalkeeper, player, referee).


Training Configuration:

Epochs: 80
Batch Size: 16
Image Size: 640x640
Optimizer: Auto (likely AdamW)
Augmentations: As per dataset (flips, brightness/saturation adjustments).


Detection: Processes frames in batches for efficiency.

3. Tracking and Analysis

Tracker: Custom Tracker class using YOLO detections and persistence (stub files for tracks).
Annotations: Draws bounding boxes and tracks on frames using supervision library.
Output: Saves processed video with annotations (e.g., 'output.avi').

The system leverages GPU for inference, making it suitable for real-time or batch video analysis.

# 🚀 Installation

Install Dependencies:
bashpip install ultralytics opencv-python numpy supervision easyocr

Roboflow Setup (for training):

Sign up at Roboflow and get an API key.
Update the API key in Train_Data_Robflow.ipynb.


GPU Support (optional but recommended):

Ensure CUDA is installed for faster training/inference.




# 🛠️ Usage
1. Training the Model

Open Train_Data_Robflow.ipynb in Jupyter/Colab.
Execute cells to:

Download the dataset from Roboflow.
Train the YOLO model.
Save the best model to models/best.pt.



2. Analyzing Videos

Open Football_Analysis.ipynb.
Update video path (e.g., 'CV_Task.mkv').
Execute cells to:

Load the trained model.
Process video frames for detection and tracking.
Save annotated video to output_videos/output.avi.



Example:

Input: Football match video clip.
Output: Video with bounding boxes around ball, players, goalkeepers, and referees, plus tracks.


# 📋 Dependencies

Python 3.10+
Ultralytics 8.3+
OpenCV 4.x
NumPy
Supervision
EasyOCR (for potential text extraction, if extended)
Roboflow (for dataset access)


# ⚠️ Limitations

Optimized for football videos; may not generalize to other sports or low-quality footage.
Ball detection can be challenging due to small size and speed (mAP ~0.214 in tests).
Tracking assumes consistent frame rates; occlusions may cause ID switches.
No handling for multi-camera views or advanced analytics (e.g., speed calculation).
