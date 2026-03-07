# arucotagdetecter
Python scripts for ArUco marker detection, gate alignment, and automatic marker image capture using OpenCV.
# arucodetecter

A small OpenCV-based Python project for ArUco marker detection.

This repository includes two scripts:

- **aruco_gate_alignment.py**  
  Detects exactly two ArUco markers, calculates the center point between them, and helps align a rover or camera system to the gate center.

- **aruco_auto_capture.py**  
  Detects ArUco markers in real time and automatically saves an image when a new marker gets close enough to the camera.

## Requirements

- Python 3
- OpenCV with ArUco module
- NumPy

## Notes

- The scripts are configured for **DICT_5X5_100** ArUco markers.
- Press **q** to quit the camera window.
- Make sure your camera is connected and accessible before running the scripts.
