import cv2
import numpy as np

# =========================
# CONFIGURATION
# =========================
CAMERA_INDEX = 0

MIN_AREA = 500
MIN_CIRCULARITY = 0.75
CENTER_TOLERANCE = 15
MIN_CONCENTRIC_CONTOURS = 3

SMOOTHING_ALPHA = 0.35
LOST_TIMEOUT_FRAMES = 60


# =========================
# HELPER FUNCTIONS
# =========================
def compute_circularity(contour):
    """
    Compute the circularity of a contour.
    A perfect circle is close to 1.0.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    return 4 * np.pi * area / (perimeter * perimeter)


def preprocess_frame(frame):
    """
    Convert the frame to grayscale, blur it,
    and create a binary mask for dark circular regions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return gray, binary


def find_concentric_target(frame):
    """
    Detect a concentric ring target in the frame and return:
    - center coordinates
    - outer radius
    - number of detected ring-like contours
    """
    gray, binary = preprocess_frame(frame)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        circularity = compute_circularity(contour)
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if not (0.75 <= aspect_ratio <= 1.25):
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 15:
            continue

        candidates.append({
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "area": area,
            "contour": contour
        })

    if not candidates:
        return None, binary

    # Group contours that share nearly the same center
    groups = []
    for candidate in candidates:
        placed = False

        for group in groups:
            group_cx = np.mean([item["cx"] for item in group])
            group_cy = np.mean([item["cy"] for item in group])

            if np.hypot(candidate["cx"] - group_cx, candidate["cy"] - group_cy) <= CENTER_TOLERANCE:
                group.append(candidate)
                placed = True
                break

        if not placed:
            groups.append([candidate])

    valid_groups = [group for group in groups if len(group) >= MIN_CONCENTRIC_CONTOURS]

    if not valid_groups:
        return None, binary

    best_group = max(valid_groups, key=lambda g: (len(g), max(item["radius"] for item in g)))

    center_x = int(round(np.median([item["cx"] for item in best_group])))
    center_y = int(round(np.median([item["cy"] for item in best_group])))
    outer_radius = int(round(max(item["radius"] for item in best_group)))

    result = {
        "center": (center_x, center_y),
        "radius": outer_radius,
        "rings_found": len(best_group)
    }

    return result, binary


def smooth_point(previous_point, new_point, alpha=0.35):
    """
    Smooth the detected center to reduce jitter.
    """
    if previous_point is None:
        return new_point

    x = int(round((1 - alpha) * previous_point[0] + alpha * new_point[0]))
    y = int(round((1 - alpha) * previous_point[1] + alpha * new_point[1]))
    return (x, y)


# =========================
# DRAWING FUNCTIONS
# =========================
def draw_target_marker(image, center):
    """
    Draw a highly visible marker at the detected target center.
    """
    cx, cy = center

    # Outer white ring
    cv2.circle(image, (cx, cy), 18, (255, 255, 255), 2)

    # Black outline for the crosshair
    cv2.line(image, (cx - 28, cy), (cx + 28, cy), (0, 0, 0), 4)
    cv2.line(image, (cx, cy - 28), (cx, cy + 28), (0, 0, 0), 4)

    # Red crosshair
    cv2.line(image, (cx - 28, cy), (cx + 28, cy), (0, 0, 255), 2)
    cv2.line(image, (cx, cy - 28), (cx, cy + 28), (0, 0, 255), 2)

    # Center point
    cv2.circle(image, (cx, cy), 8, (255, 255, 255), -1)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    # Label
    cv2.putText(
        image, "CENTER", (cx + 15, cy - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4
    )
    cv2.putText(
        image, "CENTER", (cx + 15, cy - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )


def draw_detection(frame, center, radius, rings_found):
    """
    Draw the active target detection result.
    """
    output = frame.copy()
    cx, cy = center

    # Draw outer detected circle
    cv2.circle(output, (cx, cy), radius, (0, 255, 0), 3)

    # Draw center marker
    draw_target_marker(output, (cx, cy))

    # Status text
    cv2.putText(output, "TARGET DETECTED", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(output, f"Center: X={cx} Y={cy}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(output, f"Ring-like contours: {rings_found}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return output


def draw_lost_target(frame, last_center, lost_frames):
    """
    Draw the last known target position when the target is lost.
    """
    output = frame.copy()

    if last_center is not None:
        cx, cy = last_center

        # Mark the last known position
        cv2.circle(output, (cx, cy), 22, (0, 255, 255), 2)
        cv2.line(output, (cx - 24, cy), (cx + 24, cy), (0, 255, 255), 2)
        cv2.line(output, (cx, cy - 24), (cx, cy + 24), (0, 255, 255), 2)

        cv2.putText(output, "TARGET LOST", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(output, f"Last seen position: X={cx} Y={cy}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(output, "LAST SEEN HERE", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(output, f"Lost for {lost_frames} frame(s)", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    else:
        cv2.putText(output, "TARGET HAS NOT BEEN SEEN YET", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return output


# =========================
# MAIN TRACKING FUNCTION
# =========================
def detect_and_track_target():
    """
    Run real-time concentric target detection and tracking from the camera.
    """
    capture = cv2.VideoCapture(CAMERA_INDEX)

    if not capture.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    tracked_center = None
    last_seen_center = None
    lost_frames = 0
    has_ever_been_seen = False

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read a frame from the camera.")
            break

        result, binary = find_concentric_target(frame)

        if result is not None:
            raw_center = result["center"]
            raw_radius = result["radius"]

            tracked_center = smooth_point(tracked_center, raw_center, SMOOTHING_ALPHA)
            last_seen_center = tracked_center
            lost_frames = 0
            has_ever_been_seen = True

            visualization = draw_detection(
                frame,
                tracked_center,
                raw_radius,
                result["rings_found"]
            )

            print(
                f"\rTARGET DETECTED -> X={tracked_center[0]} Y={tracked_center[1]}     ",
                end=""
            )

        else:
            lost_frames += 1

            if lost_frames > LOST_TIMEOUT_FRAMES:
                tracked_center = None

            if has_ever_been_seen:
                visualization = draw_lost_target(frame, last_seen_center, lost_frames)

                if last_seen_center is not None:
                    print(
                        f"\rTARGET LOST -> LAST SEEN X={last_seen_center[0]} "
                        f"Y={last_seen_center[1]} | lost_frames={lost_frames}     ",
                        end=""
                    )
                else:
                    print("\rTARGET LOST     ", end="")
            else:
                visualization = draw_lost_target(frame, None, lost_frames)
                print("\rTARGET NOT FOUND YET     ", end="")

        cv2.imshow("Target Tracking", visualization)
        cv2.imshow("Binary Mask", binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_track_target()
