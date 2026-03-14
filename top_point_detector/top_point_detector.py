import cv2
import numpy as np
import os

filename = "ay.jpg"
script_dir = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(script_dir, filename)

def safe_imread(file_path):
    # Handles paths with non-ASCII characters
    try:
        with open(file_path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_array = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

image = safe_imread(full_path)
if image is None:
    print("Error: Could not read the file!")
    exit()

image = cv2.resize(image, (800, 450))

cv2.namedWindow("HSV Tuner")
cv2.resizeWindow("HSV Tuner", 600, 350)

def empty(x): pass

cv2.createTrackbar("H Min", "HSV Tuner", 0, 179, empty)
cv2.createTrackbar("H Max", "HSV Tuner", 179, 179, empty)
cv2.createTrackbar("S Min", "HSV Tuner", 0, 255, empty)
cv2.createTrackbar("S Max", "HSV Tuner", 55, 255, empty)
cv2.createTrackbar("V Min", "HSV Tuner", 60, 255, empty)
cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, empty)

print("Started! Press 'q' to exit.")

while True:
    output = image.copy()
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
    h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
    s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
    s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
    v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
    v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")
    
    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            
            cv2.circle(output, topmost, 10, (0, 0, 255), -1)
            cv2.line(output, (400, 450), topmost, (255, 0, 0), 2)
            cv2.putText(output, f"TOP: {topmost}", (topmost[0]+10, topmost[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Result", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
