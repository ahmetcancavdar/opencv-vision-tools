import cv2
import time
from pathlib import Path

# --- 1. ADIM: GEREKLİ KURULUMLAR VE HAZIRLIK ---

# ArUco sözlüğü
aruco_sozluk = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Tespit parametreleri
aruco_parametreler = cv2.aruco.DetectorParameters()

# Köşe iyileştirme açalım; tespiti biraz daha kararlı yapar
aruco_parametreler.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Detector nesnesi
detector = cv2.aruco.ArucoDetector(aruco_sozluk, aruco_parametreler)

# --- 2. ADIM: KAMERA KAYNAĞI YERİNE KAYDEDİLEN KARELERİ OKUMA ---

FRAME_DIR = Path("/tmp/aruco_frames")
PENCERE_ADI = "ArUco Tespit Edici"

def en_son_kareyi_bul():
    dosyalar = list(FRAME_DIR.glob("frame*.jpg"))
    if not dosyalar:
        return None
    return max(dosyalar, key=lambda p: p.stat().st_mtime)

# --- 3. ADIM: GERÇEK ZAMANLI GÖRÜNTÜ İŞLEME DÖNGÜSÜ ---

if not FRAME_DIR.exists():
    print(f"Hata: {FRAME_DIR} klasörü bulunamadı.")
    print("Önce rpicam-vid komutunu çalıştır.")
    exit()

cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)

son_okunan_dosya = None
fps_sayac = 0
fps_baslangic = time.time()
fps_degeri = 0.0

while True:
    kare_yolu = en_son_kareyi_bul()

    if kare_yolu is None:
        bos = 255 * (cv2.UMat(480, 640, cv2.CV_8UC3).get() * 0)
        cv2.putText(
            bos,
            "Kameradan kare bekleniyor...",
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow(PENCERE_ADI, bos)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
        continue

    # Aynı kareyi tekrar tekrar işlememek için
    if son_okunan_dosya == kare_yolu:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
        continue

    son_okunan_dosya = kare_yolu

    kare = cv2.imread(str(kare_yolu))
    if kare is None:
        time.sleep(0.01)
        continue

    # ArUco tespiti
    kose_noktalari, etiket_idleri, reddedilenler = detector.detectMarkers(kare)

    # Tespit varsa kutu çiz + ID yaz
    if etiket_idleri is not None and len(etiket_idleri) > 0:
        cv2.aruco.drawDetectedMarkers(kare, kose_noktalari, etiket_idleri)

        # Her marker için ekrana daha belirgin ID yazalım
        for corners, marker_id in zip(kose_noktalari, etiket_idleri.flatten()):
            pts = corners.reshape((4, 2)).astype(int)
            x, y = pts[0]

            cv2.putText(
                kare,
                f"ID: {int(marker_id)}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Merkez noktayı da işaretleyelim
            merkez_x = int(pts[:, 0].mean())
            merkez_y = int(pts[:, 1].mean())
            cv2.circle(kare, (merkez_x, merkez_y), 4, (0, 0, 255), -1)
    else:
        cv2.putText(
            kare,
            "ArUco bulunamadi",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # FPS bilgisi
    fps_sayac += 1
    simdi = time.time()
    if simdi - fps_baslangic >= 1.0:
        fps_degeri = fps_sayac / (simdi - fps_baslangic)
        fps_sayac = 0
        fps_baslangic = simdi

    cv2.putText(
        kare,
        f"FPS: {fps_degeri:.1f}",
        (20, kare.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(PENCERE_ADI, kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. ADIM: TEMİZLİK ---
cv2.destroyAllWindows()
