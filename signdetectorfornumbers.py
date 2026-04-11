## Gereksinimler
Bu kodun çalışabilmesi için bilgisayarınızda Python yüklü olmalı ve aşağıdaki iki aşamalı kurulum adımları tamamlanmalıdır.

### 1. Python Kütüphanelerinin Kurulumu
Görüntü işleme ve karakter okuma işlemleri için gerekli Python kütüphanelerini PowerShell'i açıp aşağıdaki komutu yazarak yükleyin:

```powershell
pip install opencv-python pytesseract numpy
```

### 2. Tesseract-OCR Kurulumu (Önemli!)
`pytesseract` kütüphanesi yalnızca bir köprü görevi görür; asıl yazıyı okuyan yapay zeka motoru Tesseract'tır. Windows'ta bu motoru bilgisayarınıza manuel olarak kurmanız gerekir:

1. [Bu bağlantıya tıklayarak](https://github.com/UB-Mannheim/tesseract/wiki) indirme sayfasına gidin.
2. Sayfada yer alan güncel **tesseract-ocr-w64-setup.exe** (64-bit) dosyasını indirin.
3. İndirdiğiniz dosyayı çalıştırın ve varsayılan yola (`C:\Program Files\Tesseract-OCR`) hiçbir ayarı değiştirmeden kurulumu yapın.

*(Not: Kodunuzun 6. satırına bu programın yolunu zaten ekledik, bu yüzden sizin kurulum dışında ekstra bir kod değiştirmenize gerek kalmamıştır.)*

## Sistemin Çalıştırılması

Tüm kurulumları tamamladıktan sonra, kodun bulunduğu konuma giderek (veya doğrudan tam yolunu yazarak) çalıştırabilirsiniz. 

PowerShell uygulamasını açın ve şu komutu girin:

```powershell
python "C:\Users\bilgisayaradi\Downloads\dosyaadı.py"
```

## Kullanım ve Kontroller

- Programı çalıştırdığınızda bilgisayarınızın varsayılan kamerası (webcam) aktifleşecek ve "Sign Detection System" adında bir pencere açılacaktır.
- Kameraya kırmızı bir levha ya da çerçeve tuttuğunuzda etrafında yeşil bir kutu belirecek ve levha içindeki sayı okunup ekrana "Hiz: X" şeklinde yansıtılacaktır.
- Sol üst köşede o anki FPS (saniyedeki kare sayısı) performansını görebilirsiniz.
- Uygulamayı kapatmak ve kamerayı serbest bırakmak için pencere aktifken klavyenizden **`q`** tuşuna basmanız yeterlidir.


import cv2
import numpy as np
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -- OPTİMİZASYON İÇİN DEĞİŞKENLER --
last_detected_text = ""
last_ocr_time = 0

def process_frame(frame):
    global last_detected_text, last_ocr_time
    # 1. HSV Dönüşümü
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Renk Filtreleme: Kırmızı için alt ve üst HSV aralıkları
    lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Ufak gürültüleri temizlemek için basit bir morfolojik işlem
    # Optimizasyon: Büyük kernel'lardan ve iterasyonlardan kaçınıldı
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 3. Şekil Tespiti (Sadece dış konturları alarak işlem yükünü azaltıyoruz)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # Çok küçük kırmızı lekeleri yoksay
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            # Dairesellik (Circularity) kontrolü: 4*pi*Alan / Çevre^2 (Tam daire 1'dir)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if 0.7 < circularity < 1.2:
                # Tabela bulundu, bounding box al
                x, y, w, h = cv2.boundingRect(cnt)

                # 4. ROI Kırpma (Kırmızı çerçeveyi dışarıda bırak, içteki beyaz alana odaklan)
                margin = int(w * 0.15)
                roi_y1, roi_y2 = y + margin, y + h - margin
                roi_x1, roi_x2 = x + margin, x + w - margin

                # ROI mantıklı sınırların içindeyse işleme devam et
                if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # Grayscale ve Otsu Threshold ile sayıyı siyah, arka planı beyaz yap
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    # -- OPTİMİZASYON: Tesseract'ı her karede değil, saniyede sadece 2 kez çalıştır --
                    current_t = time.time()
                    if current_t - last_ocr_time > 0.5:
                        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
                        text = pytesseract.image_to_string(thresh, config=config).strip()
                        if text:
                            last_detected_text = text
                        last_ocr_time = current_t

                    if last_detected_text:
                        # 5. Görselleştirme: Sadece sayının etrafına kare çizmek için threshold'dan kontur bul
                        num_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if num_contours:
                            # En büyük siyah alanı sayı olarak kabul et
                            num_cnt = max(num_contours, key=cv2.contourArea)
                            nx, ny, nw, nh = cv2.boundingRect(num_cnt)

                            # Koordinatları orijinal frame'e göre ayarla
                            box_x = roi_x1 + nx
                            box_y = roi_y1 + ny

                            # Sayının etrafına yeşil kare çiz
                            cv2.rectangle(frame, (box_x, box_y), (box_x + nw, box_y + nh), (0, 255, 0), 2)
                            
                            # Sayıyı karenin üstüne yazdır
                            cv2.putText(frame, f"Hiz: {last_detected_text}", (box_x, box_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def main():
    # Test için kamerayı başlat (0 varsayılan web kamerası)
    # Raspberry Pi'de PiCamera kullanıyorsan cv2.VideoCapture(0) çalışacaktır.
    cap = cv2.VideoCapture(0)
    
    # Optimizasyon: Çözünürlüğü düşürerek FPS'i artır
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0

    print("[BİLGİ] Kamera akışı başlatıldı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # -- OPTİMİZASYON: Görüntüyü manuel küçültmek hızı çok artırır --
        frame = cv2.resize(frame, (640, 480))

        # FPS Hesaplama
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        processed_frame = process_frame(frame)

        # FPS'i ekrana yazdır
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Sign Detection System", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
