import cv2
from ultralytics import YOLO
import easyocr

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
    Mengekstrak teks dari plat nomor menggunakan EasyOCR.

    Parameters:
        frame (numpy array): Frame gambar dari video.
        plates (list): Daftar bounding box plat nomor [(x1, y1, x2, y2)].

    Returns:
        tuple: (str, numpy array) -> Teks plat nomor dan citra plat yang diperbesar
    """
    if not plates:
        return "Unknown", None

    extracted_texts = []
    enlarged_images = []

    for (x1, y1, x2, y2) in plates:
        x1 -= 20
        x2 += 20
        y1 -= 20
        y2 += 20
        plate_img = frame[y1:y2, x1:x2]  # Crop area plat nomor

        # Perbesar ukuran image 2x
        #enlarged_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Gunakan easyocr untuk membaca teks
        results = reader.readtext(plate_img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        text = results[0].strip() if results else "Unrecognized"
        extracted_texts.append(text)
        enlarged_images.append(plate_img)

    return extracted_texts[0], enlarged_images[0]