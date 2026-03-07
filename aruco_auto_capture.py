import cv2
import numpy as np
import time
import os

def find_available_camera_index(max_test=10):
    for camera_index in range(max_test):
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.release()
            print(f"Camera found: index = {camera_index}")
            return camera_index
        camera.release()
    return None

camera_index = find_available_camera_index()
if camera_index is None:
    raise RuntimeError("No camera index could be opened!")

camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
print(f"Camera (index={camera_index}) opened. Press 'q' to quit.")

aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
detector_parameters = cv2.aruco.DetectorParameters()

captured_marker_ids = set()
area_threshold = 0.20

output_directory = "captured_tags"
os.makedirs(output_directory, exist_ok=True)

window_name = "Free ArUco Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\nSystem ready! You can show any ArUco marker to the camera...")

while True:
    success, frame = camera.read()
    if not success:
        print("Frame could not be captured, retrying...")
        time.sleep(0.2)
        continue

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        grayscale,
        aruco_dictionary,
        parameters=detector_parameters
    )

    if marker_ids is not None:
        flat_marker_ids = marker_ids.flatten()

        for index, marker_id in enumerate(flat_marker_ids):
            if marker_id not in captured_marker_ids:
                frame_height, frame_width = frame.shape[:2]
                points = marker_corners[index][0].astype(int)
                area_ratio = cv2.contourArea(points) / (frame_height * frame_width)

                if area_ratio >= area_threshold:
                    print(f"\nNEW MARKER CAPTURED! ID {marker_id} (Area: %{area_ratio * 100:.1f})")

                    file_name = f"tag_{marker_id}.jpg"
                    file_path = os.path.join(output_directory, file_name)
                    cv2.imwrite(file_path, frame)
                    print(f"Photo saved: {file_path}")

                    captured_marker_ids.add(marker_id)

        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    status_text = f"Total Captured: {len(captured_marker_ids)}"
    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nExit complete. Photos of {len(captured_marker_ids)} different markers were captured.")
        break

camera.release()
cv2.destroyAllWindows()
