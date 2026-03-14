# OpenCV Vision Tools 

A collection of OpenCV-based Python scripts for Computer Vision tasks, including ArUco marker detection, autonomous rover gate alignment, and specific object (e.g., dark stone) detection. 

Ideal for robotics, autonomous vehicles (rovers), and general computer vision projects.

## Included Scripts

### 1. `aruco_gate_alignment.py`
Detects exactly two ArUco markers, calculates the center point between them, and helps align a rover or camera system to the gate center. Perfect for autonomous navigation tasks where a vehicle needs to pass through a specific gate.

### 2. `aruco_auto_capture.py`
Detects ArUco markers in real-time using a webcam and automatically saves an image to your local directory when a new marker gets close enough to the camera.

### 3. `black_stone_detector.py`
Analyzes a static image to detect and isolate dark-colored stones. It utilizes Region of Interest (ROI) cropping, color thresholding, and morphological operations to eliminate background noise. It evaluates the "darkness" of each detected shape and highlights the darkest stone with a distinct bounding box.

## Requirements

Make sure you have Python 3 installed, along with the required libraries. You can install the dependencies using pip:

```bash
pip install opencv-contrib-python numpy
