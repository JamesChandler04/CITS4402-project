# CITS4402 Project – Face Detection and Matching

## Overview

This project is a Python based GUI application for face detection, facial landmark detection, face alignment, and identity clustering.

The program provides two main modes:

- Single Image  
  Loads one image, detects faces, detects facial landmarks, aligns faces, and displays the processed output in the GUI.

- Bulk Processing  
  Loads all valid images from a selected folder, processes them one by one, extracts aligned faces, computes face embeddings, clusters them into identities, and saves the final aligned face crops into a Processed_Images folder.

The GUI displays:
- the input image on the left
- the processed image on the right
- status messages
- processing time and result summary

## Features

- Face detection using OpenCV DNN
- Skin colour segmentation used to assist face detection
- Facial landmark detection using MediaPipe Face Landmarker
- Face alignment using similarity transformation
- Display of aligned 125 x 125 face crops at image corners
- Bulk processing of all images inside a selected folder
- Face embedding extraction using InsightFace buffalo_l
- Automatic identity clustering using Agglomerative Clustering
- Final clustered output saved as:

Identity_N_face_M.jpg

## Project Structure

Make sure your project folder is arranged like this:

CITS4402-project/
│
├── program.py
├── requirements.txt
├── README.md
├── .gitignore 
└── models/
    ├── deploy.prototxt
    ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
    ├── face_landmarker.task
    └── insightface-0.7.3-cp311-cp311-win_amd64.whl

If your script file has a different name, replace program.py in the run command below with your actual filename.

## Requirements

This project was developed with:
- Python 3.11
- NumPy 1.26.4
- OpenCV 4.11.0.86
- MediaPipe 0.10.35
- ONNX Runtime
- scikit-learn
- Pillow
- InsightFace wheel file stored locally in the models folder

A clean requirements.txt should look like this:

numpy==1.26.4
opencv-python==4.11.0.86
pillow
mypy
scikit-learn
mediapipe==0.10.35
onnxruntime
models\insightface-0.7.3-cp311-cp311-win_amd64.whl

## Installation

### 1. Create and activate a virtual environment

Windows Command Prompt

python -m venv .cvLabs
.cvLabs\Scripts\activate

Windows PowerShell

python -m venv .cvLabs
.\.cvLabs\Scripts\Activate.ps1

### 2. Install dependencies

pip install -r requirements.txt

If the local wheel path in requirements.txt causes any issue, install packages manually:

pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install pillow
pip install mypy
pip install scikit-learn
pip install mediapipe==0.10.35
pip install onnxruntime
pip install models\insightface-0.7.3-cp311-cp311-win_amd64.whl

## How to Run

Run the script using:

python program.py

If your file is named differently, for example:

python YourScriptName.py

## How to Use

### Single Image Mode

1. Launch the program.
2. Click Single Image.
3. Choose an image file.
4. The input image will appear on the left.
5. The processed image will appear on the right.
6. The GUI will display:
   - processing time
   - raw detections
   - skin validated faces

### Bulk Processing Mode

1. Launch the program.
2. Click Bulk Processing.
3. Select a folder containing input images.
4. The program will:
   - process all valid images in the folder
   - detect and align faces
   - extract face embeddings
   - cluster the faces into identities
   - save final outputs in a new folder named Processed_Images

The GUI will display:
- total images processed
- total processing time
- total raw detections
- total validated faces
- total face embeddings extracted
- total clustered faces saved
- number of unique identities found

## Output Folder

Bulk processing creates this folder automatically:

Processed_Images

It is created in the same location as the selected input folder.

If the folder already exists, the program will delete old files inside it before saving new results so outputs from different runs do not mix.

Saved files follow this naming format:

Identity_1_face_1.jpg
Identity_1_face_2.jpg
Identity_2_face_1.jpg

## Important Notes

### Initial run may take longer

On the first bulk processing run, the program may take extra time because the InsightFace buffalo_l model needs to be prepared and loaded.

This is normal.

During this step, the GUI status text may show a message similar to:
- Loading pretrained face feature model. First run may take time.
- Please wait while buffalo_l model is prepared.

Later runs are usually faster.

### Supported input image formats

The program accepts common image formats such as:
- .png
- .jpg
- .jpeg
- .bmp
- .gif
- .tif
- .tiff

## Troubleshooting

### 1. Could not find model file

Make sure all required model files are present in the models folder:
- deploy.prototxt
- res10_300x300_ssd_iter_140000_fp16.caffemodel
- face_landmarker.task
- insightface-0.7.3-cp311-cp311-win_amd64.whl

### 2. InsightFace import or binary compatibility error

If you get errors related to NumPy binary compatibility, make sure NumPy is pinned correctly:

pip uninstall -y numpy
pip install numpy==1.26.4

Then reinstall the InsightFace wheel if needed.

### 3. Bulk processing finds no images

Make sure the selected folder contains valid image files and not subfolders only.

### 4. First bulk run seems slow

This is expected due to model initialization.

Wait for the first run to complete before assuming the program is stuck.

## Method Summary

The project pipeline is:
1. Read image
2. Detect faces using OpenCV DNN
3. Use skin colour segmentation to validate detections
4. Detect landmarks inside each face region
5. Align faces using similarity transformation
6. Resize aligned face crops to 125 x 125
7. Show aligned faces on image corners
8. In bulk processing:
   - extract face embeddings using InsightFace buffalo_l
   - cluster embeddings automatically
   - save clustered faces as identity-based output files

## Notes for Demonstration / Submission

- The script is self contained and launches the GUI directly.
- The project uses pretrained models for:
  - face detection
  - landmark detection
  - face embedding extraction
- The code is commented and structured for clarity.
- Bulk outputs are saved without landmarks as required for identity clustering output.

## Authors

- Group Member 1 : James Chandler (23348041)
- Group Member 2 : Hamza Hanif Mir (24410764)
- Group Member 3 : George Guillain (25077521)

