# Static Image HSV Tuner & Top Point Detector 

This tool allows you to fine-tune HSV (Hue, Saturation, Value) color thresholds on a static image to isolate specific paths or objects. Once the target color is isolated, it calculates the "topmost" point of the largest detected area and draws a directional vector to it. This is highly useful for autonomous vehicle (rover) path planning and heading calculations.

## Key Features
* **Interactive HSV GUI:** 6 real-time trackbars to perfectly isolate target colors regardless of lighting conditions.
* **Target Vectoring:** Automatically finds the highest coordinate of the path/object and draws a guiding line from the bottom center.
* **Safe File Loading:** Custom `safe_imread` function prevents OpenCV crashes when reading files from paths with non-ASCII (e.g., Turkish) characters.

## Usage
1. Place your target image (default: `ay.jpg`) in the same directory as the script.
2. Run the script: `python top_point_detector.py`
3. Adjust the sliders in the **HSV Tuner** window until your target area is white in the "Mask" window.
4. Press **`q`** to exit the program.
