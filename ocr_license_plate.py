from ultralytics import YOLO
import easyocr
import re

model = YOLO("license_number.pt")

reader = easyocr.Reader(['en'])  
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
    # Pastikan bbox dalam bentuk tuple/list 4 elemen
    if isinstance(truck_bbox, (list, tuple)) and len(truck_bbox) == 1:
        truck_bbox = truck_bbox[0]
    if len(truck_bbox) != 4:
        print(f"Invalid bounding box: {truck_bbox}")
        return []

    x1_t, y1_t, x2_t, y2_t = map(int, truck_bbox)

    # Validasi agar tidak keluar batas dan crop tidak kosong
    x1_t, y1_t = max(0, x1_t), max(0, y1_t)
    x2_t, y2_t = min(frame.shape[1], x2_t), min(frame.shape[0], y2_t)

    if x2_t <= x1_t or y2_t <= y1_t:
        print(f"Invalid crop area: {(x1_t, y1_t, x2_t, y2_t)}")
        return []

    # Crop frame hanya di area truk
    cropped_frame = frame[y1_t:y2_t, x1_t:x2_t]
    if cropped_frame.size == 0:
        print("Cropped frame is empty.")
        return []

    # Deteksi dengan model
    results = model(cropped_frame, verbose=False)
    plates = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])

            # Adjust koordinat ke frame asli
            abs_x1, abs_y1 = x1 + x1_t, y1 + y1_t
            abs_x2, abs_y2 = x2 + x1_t, y2 + y1_t

            plates.append((abs_x1, abs_y1, abs_x2, abs_y2))

    return plates

import re

def correct_plate_format(text):
    """
    Koreksi karakter OCR hanya pada bagian huruf.
    Misalnya: 8 → B, 0 → O, 1 → I hanya untuk bagian huruf.
    """
    text = text.strip().upper().replace("  ", " ")
    pattern = re.match(r"^([A-Z]{1,2}) ?(\d{1,4}) ?([A-Z]{1,3})$", text)

    if not pattern:
        return text  # Tidak cocok dengan pola, jangan ubah

    first, number, last = pattern.groups()

    # Koreksi hanya bagian huruf
    def fix_letters(s):
        return s.replace('0', 'O').replace('8', 'B').replace('1', 'I')

    corrected_first = fix_letters(first)
    corrected_last = fix_letters(last)

    return f"{corrected_first} {number} {corrected_last}"

def extract_license_plate_text(frame, plates):
    """
    Mengekstrak teks dari plat nomor menggunakan EasyOCR dengan koreksi OCR
    hanya pada bagian huruf, dan validasi regex.

    Parameters:
        frame (numpy array): Frame gambar dari video.
        plates (list): Daftar bounding box plat nomor [(x1, y1, x2, y2)].

    Returns:
        tuple: (str, numpy array) -> Teks plat nomor (asli atau hasil koreksi) dan citra plat
    """
    if not plates:
        return "Unknown", None

    extracted_texts = []
    enlarged_images = []

    for (x1, y1, x2, y2) in plates:
        x1 -= 10
        x2 += 10
        y1 -= 10
        y2 += 10
        plate_img = frame[y1:y2, x1:x2]

        results = reader.readtext(plate_img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
        matched_text = "Unrecognized"

        for text in results:
            cleaned = text.strip().upper().replace("  ", " ")

            # Hasil koreksi hanya pada huruf
            corrected = correct_plate_format(cleaned)

            if re.match(r"^[A-Z]{1,2} ?\d{1,4} ?[A-Z]{1,3}$", corrected):
                matched_text = corrected
                break
            else:
                matched_text = cleaned if cleaned else "Unrecognized"

        extracted_texts.append(matched_text)
        enlarged_images.append(plate_img)

    return extracted_texts[0], enlarged_images[0]
