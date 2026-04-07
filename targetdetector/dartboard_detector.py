import cv2
import numpy as np
import antigravity  # Python's famous easter egg! Don't be surprised if your browser opens :)

# ══════════════════════════════════════════════════════════════
#  DARTBOARD DETECTOR (Using Hough Circles)
# ══════════════════════════════════════════════════════════════

def detect_dartboard():
    # 1. Initialize the Camera (0 generally represents the built-in webcam, 1 is for an external camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not found or could not be opened!")
        return

    print("Camera opened. Press 'q' to exit.")

    while True:
        # Read the current frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Pre-processing: Convert the image to grayscale and blur it to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        # 3. Circle Detection (Hough Circle Transform)
        # This function searches for circular shapes in the image.
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=100,       # Minimum distance between the centers of two circles
            param1=50,         # Sensitivity of the Canny edge detector
            param2=90,         # INCREASED from 60 to 90: Only perfectly round shapes are accepted (filters out faces)
            minRadius=50,      # Minimum circle radius to search for
            maxRadius=300      # Maximum circle radius to search for
        )

        # 4. Draw Circles on the Screen
        if circles is not None:
            # Convert the incoming data to integers (pixel coordinates cannot be floats)
            circles = np.uint16(np.around(circles))
            
            for i in circles[0, :]:
                # i[0] = Center X, i[1] = Center Y, i[2] = Radius
                center = (i[0], i[1])
                radius = i[2]
                
                # Draw the outer circle of the dartboard in green
                cv2.circle(frame, center, radius, (0, 255, 0), 3)
                
                # Draw a clear crosshair and point exactly at the center of the dartboard
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.line(frame, (i[0]-15, i[1]), (i[0]+15, i[1]), (0, 0, 255), 2)
                cv2.line(frame, (i[0], i[1]-15), (i[0], i[1]+15), (0, 0, 255), 2)
                
                # Print the exact (X, Y) coordinate values of the center on the screen
                cv2.putText(frame, f"Center: X:{i[0]} Y:{i[1]}", (i[0] + 15, i[1] - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Write the title above it
                cv2.putText(frame, "Dartboard", (i[0] - 50, i[1] - radius - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 5. Show the Result on the Screen
        cv2.imshow("Dartboard Detector - Antigravity Active", frame)

        # If 'q' is pressed, break the loop and exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    detect_dartboard()
