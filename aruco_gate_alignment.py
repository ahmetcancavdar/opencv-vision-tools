import cv2
import cv2.aruco as aruco

camera = cv2.VideoCapture(0)
aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
detector_parameters = aruco.DetectorParameters()

CENTER_THRESHOLD = 40

window_name = "Rover Gate Centering System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("System started! Show the tags to the camera. (Press 'q' to quit)")

while True:
    success, frame = camera.read()
    if not success:
        break

    frame_height, frame_width = frame.shape[:2]
    screen_center_x = frame_width // 2

    cv2.line(frame, (screen_center_x, 0), (screen_center_x, frame_height), (0, 255, 0), 1)

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = aruco.detectMarkers(
        grayscale,
        aruco_dictionary,
        parameters=detector_parameters
    )

    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        number_of_tags = len(marker_ids)

        if number_of_tags == 1 or number_of_tags == 3:
            error_message = f"ERROR: {number_of_tags} tag(s) detected! (2 required)"
            cv2.putText(frame, error_message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print(error_message)

        elif number_of_tags == 2:
            first_marker = marker_corners[0][0]
            first_x = int((first_marker[0][0] + first_marker[2][0]) / 2)
            first_y = int((first_marker[0][1] + first_marker[2][1]) / 2)

            second_marker = marker_corners[1][0]
            second_x = int((second_marker[0][0] + second_marker[2][0]) / 2)
            second_y = int((second_marker[0][1] + second_marker[2][1]) / 2)

            gate_center_x = int((first_x + second_x) / 2)
            gate_center_y = int((first_y + second_y) / 2)

            cv2.line(frame, (first_x, first_y), (second_x, second_y), (255, 0, 0), 2)
            cv2.circle(frame, (gate_center_x, gate_center_y), 8, (0, 255, 255), -1)

            alignment_error = gate_center_x - screen_center_x

            if abs(alignment_error) <= CENTER_THRESHOLD:
                status_message = "ROVER CENTERED! MOVE FORWARD"
                status_color = (0, 255, 0)
            elif alignment_error > 0:
                status_message = f"TURN RIGHT (Error: +{alignment_error} px)"
                status_color = (0, 165, 255)
            else:
                status_message = f"TURN LEFT (Error: {alignment_error} px)"
                status_color = (0, 165, 255)

            cv2.putText(frame, status_message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            print(status_message)

        else:
            cv2.putText(
                frame,
                f"UNKNOWN STATE: {number_of_tags} tags",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

    else:
        cv2.putText(frame, "Waiting for tags...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
