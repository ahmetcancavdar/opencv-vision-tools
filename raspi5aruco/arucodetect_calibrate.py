import cv2
import time
import json
from pathlib import Path
import numpy as np

# ==============================
# AYARLAR
# ==============================
MARKER_SIZE_M = 0.20               # 20 cm = 0.20 m
CALIBRATION_DISTANCE_M = 0.50      # Kalibrasyon sırasında marker kameradan kaç metre uzakta olacak?
FRAME_DIR = Path("/tmp/aruco_frames")
CALIB_FILE = Path("aruco_distance_calibration.json")
WINDOW_NAME = "ArUco Distance Estimation"

# ArUco sözlüğü
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Parametreler
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Detector
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def get_latest_frame():
    files = list(FRAME_DIR.glob("frame*.jpg"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def edge_lengths(corners):
    pts = corners.reshape((4, 2)).astype(np.float32)
    lengths = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        lengths.append(np.linalg.norm(p2 - p1))
    return lengths


def marker_pixel_size(corners):
    # Marker'ın görüntüdeki kenar uzunluklarının ortalaması (piksel)
    lengths = edge_lengths(corners)
    return float(np.mean(lengths))


def marker_center(corners):
    pts = corners.reshape((4, 2)).astype(np.int32)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy


def load_calibration():
    if CALIB_FILE.exists():
        try:
            data = json.loads(CALIB_FILE.read_text())
            return float(data["focal_px"])
        except Exception:
            return None
    return None


def save_calibration(focal_px):
    data = {
        "focal_px": focal_px,
        "marker_size_m": MARKER_SIZE_M,
        "calibration_distance_m": CALIBRATION_DISTANCE_M
    }
    CALIB_FILE.write_text(json.dumps(data, indent=2))


def estimate_distance_m(pixel_size, focal_px):
    if pixel_size <= 0 or focal_px is None:
        return None
    return (MARKER_SIZE_M * focal_px) / pixel_size


if not FRAME_DIR.exists():
    print(f"Hata: {FRAME_DIR} klasörü yok.")
    print("Önce rpicam-vid komutunu çalıştır.")
    raise SystemExit

focal_px = load_calibration()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

last_file = None
fps_counter = 0
fps_start = time.time()
fps_value = 0.0

while True:
    frame_path = get_latest_frame()

    if frame_path is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Kameradan kare bekleniyor...", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, blank)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
        continue

    if frame_path == last_file:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
        continue

    last_file = frame_path

    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue

    corners, ids, rejected = detector.detectMarkers(frame)

    largest_marker_idx = None
    largest_marker_px = -1.0

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        ids_flat = ids.flatten()

        for i, (marker_corners, marker_id) in enumerate(zip(corners, ids_flat)):
            px_size = marker_pixel_size(marker_corners)
            cx, cy = marker_center(marker_corners)

            if px_size > largest_marker_px:
                largest_marker_px = px_size
                largest_marker_idx = i

            # ID yaz
            pts = marker_corners.reshape((4, 2)).astype(int)
            x, y = pts[0]
            cv2.putText(
                frame,
                f"ID: {int(marker_id)}",
                (x, max(25, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Merkez işaretle
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Mesafe hesapla
            distance_m = estimate_distance_m(px_size, focal_px)
            if distance_m is not None:
                cv2.putText(
                    frame,
                    f"Mesafe: {distance_m:.2f} m",
                    (x, max(50, y + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    frame,
                    "Kalibrasyon yok (c)",
                    (x, max(50, y + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )
    else:
        cv2.putText(
            frame,
            "ArUco bulunamadi",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # FPS
    fps_counter += 1
    now = time.time()
    if now - fps_start >= 1.0:
        fps_value = fps_counter / (now - fps_start)
        fps_counter = 0
        fps_start = now

    # Bilgi yazıları
    cv2.putText(
        frame,
        f"FPS: {fps_value:.1f}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    if focal_px is not None:
        cv2.putText(
            frame,
            f"focal_px: {focal_px:.1f}",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame,
            f"Kalibrasyon: markeri {CALIBRATION_DISTANCE_M:.2f} m'ye koy, c'ye bas",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    # q = çıkış
    if key == ord('q'):
        break

    # c = kalibrasyon
    if key == ord('c'):
        if ids is not None and len(ids) > 0 and largest_marker_idx is not None:
            calib_px = marker_pixel_size(corners[largest_marker_idx])
            focal_px = (calib_px * CALIBRATION_DISTANCE_M) / MARKER_SIZE_M
            save_calibration(focal_px)
            print(f"[OK] Kalibrasyon yapildi. focal_px = {focal_px:.3f}")
        else:
            print("[UYARI] Kalibrasyon için önce marker görünmeli.")

cv2.destroyAllWindows()
