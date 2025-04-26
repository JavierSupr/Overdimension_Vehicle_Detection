import pytesseract
import cv2
from ultralytics import YOLO

model = YOLO("license_number.pt")

import cv2

def detect_license_plate(frame, truck_bbox, save_path="detected_plate.jpg"):
    """
    Mendeteksi plat nomor kendaraan dalam gambar menggunakan YOLOv8,
    hanya di dalam bounding box truk, dan menyimpan gambar dengan bounding box.

    Parameters:
        frame (numpy array): Frame gambar dari video.
        truck_bbox (tuple): Bounding box truk (x1, y1, x2, y2).
        save_path (str): Path untuk menyimpan gambar hasil deteksi.

    Returns:
        list: Daftar bounding box plat nomor [(x1, y1, x2, y2)] dalam koordinat frame asli.
    """
    x1_t, y1_t, x2_t, y2_t = truck_bbox[0]

    # Crop frame hanya di area truk
    cropped_frame = frame[y1_t:y2_t, x1_t:x2_t]

    # Deteksi dengan model
    results = model(cropped_frame,verbose=False)
    plates = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])

            # Adjust koordinat ke frame asli
            abs_x1, abs_y1 = x1 + x1_t, y1 + y1_t
            abs_x2, abs_y2 = x2 + x1_t, y2 + y1_t

            # Simpan koordinat
            plates.append((abs_x1, abs_y1, abs_x2, abs_y2))

            # Gambar bounding box ke frame asli

    # Simpan frame dengan bounding box

    return plates


def extract_license_plate_text(frame, plates):
    """
    Mengekstrak teks dari plat nomor menggunakan Tesseract OCR.
    Parameters:
        frame (numpy array): Frame gambar dari video.
        plates (list): Daftar bounding box plat nomor [(x1, y1, x2, y2)].
    Returns:
        tuple: (str, numpy array) -> Teks plat nomor dan citra plat yang diperbesar
    """
    if not plates:
        return "Unknown", None  # Tidak ada plat nomor terdeteksi oleh YOLO

    extracted_texts = []
    enlarged_images = []

    for (x1, y1, x2, y2) in plates:
        plate_img = frame[y1:y2, x1:x2]  # Crop area plat nomor

        # Perbesar ukuran image 2x
        enlarged_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        # Preprocessing
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Konfigurasi OCR
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()

        if text == "":
            text = "Unrecognized"

        extracted_texts.append(text)
        enlarged_images.append(enlarged_img)

    return extracted_texts[0], enlarged_images[0]