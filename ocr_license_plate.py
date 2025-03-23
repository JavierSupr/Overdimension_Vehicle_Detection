import pytesseract
import cv2
from ultralytics import YOLO


model = YOLO("license_number.pt")

def detect_license_plate(frame):
    """
    Mendeteksi plat nomor kendaraan dalam gambar menggunakan YOLOv8.
    Parameters:
        frame (numpy array): Frame gambar dari video.
    Returns:
        list: Daftar bounding box plat nomor [(x1, y1, x2, y2)].
    """
    results = model(frame)  # Prediksi YOLOv8
    plates = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Ambil koordinat bounding box
            plates.append((x1, y1, x2, y2))

    return plates

def extract_license_plate_text(frame, plates):
    """
    Mengekstrak teks dari plat nomor menggunakan Tesseract OCR.
    Parameters:
        frame (numpy array): Frame gambar dari video.
        plates (list): Daftar bounding box plat nomor [(x1, y1, x2, y2)].
    Returns:
        str: Teks plat nomor yang diekstrak atau "Unrecognized" jika OCR gagal.
    """
    if not plates:
        return "Unknown"  # Tidak ada plat nomor terdeteksi oleh YOLO

    extracted_texts = []

    for (x1, y1, x2, y2) in plates:
        plate_img = frame[y1:y2, x1:x2]  # Crop area plat nomor
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        # Preprocessing untuk meningkatkan akurasi OCR
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Konfigurasi Tesseract OCR untuk membaca huruf kapital dan angka saja
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()

        # Jika OCR gagal mendeteksi karakter, set "Unrecognized"
        if text == "":
            text = "Unrecognized"

        extracted_texts.append(text)

    return extracted_texts[0] if extracted_texts else "Unrecognized"
