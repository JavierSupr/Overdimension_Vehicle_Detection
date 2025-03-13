import datetime
import cv2
from pymongo import MongoClient
from gridfs import GridFS
import numpy as np
import pytesseract
from ultralytics import YOLO

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['traffic_violations']
collection = db['violations']  # Main collection for metadata
fs = GridFS(db)  # GridFS setup
location = 'Jl. Pantura KM 23'

model = YOLO("best.pt")

def check_id_exists(track_id):
    """Check if a track_id already exists in MongoDB."""
    return collection.find_one({'licensePlate': f"ID_{track_id}"}) is not None

def draw_mask_on_detected_tracks(frame, tracked_objects):
    """
    Draws the segmentation mask for detected objects on the frame with different colors based on class names.

    Parameters:
        frame (numpy array): The current video frame.
        tracked_objects (list): List of detected objects with masks.
    """
    # Define colors for each class
    class_colors = {
        "Tampak Depan": (255, 0, 0),  # Blue
        "Tampak Samping": (0, 255, 0),  # Green
        "Truk": (0, 0, 255)  # Red
    }

    for track in tracked_objects:
        mask = track.get("mask")
        class_name = track.get("class_name")

        if mask is not None and mask.size > 0 and class_name in class_colors:
            color = class_colors[class_name]
            mask = mask.reshape((-1, 1, 2)).astype(np.int32)  # Reshape mask to contour format
            
            # Create an overlay for transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [mask], color)  # Draw mask
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # Blend mask with frame

            # Draw bounding box
            x1, y1, x2, y2 = track["bounding box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame



def convert_tracked_objects_to_dict(tracked_objects):
    """
    Converts a list of tracked object tuples into a list of dictionaries.

    Parameters:
        tracked_objects (list of tuples): Each tuple contains (track_object, class_name, track_id, mask).

    Returns:
        list of dict: A list where each entry is a dictionary with 'track_id' and 'mask'.
    """
    tracked_dicts = []
    for obj in tracked_objects:
        if isinstance(obj, tuple) and len(obj) == 4:
            track_obj, class_name, track_id, mask = obj
            tracked_dicts.append({
                "track_id": track_id,
                "class_name": class_name,
                "mask": mask
            })
        else:
            print(f"Warning: Invalid tracked object format - {obj}")
    
    return tracked_dicts

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


def save_violation_to_mongodb(frame, track_id, height, is_multicam, camera_name):
    """
    Saves violation data to MongoDB, including an image with a mask drawn and license plate detection.
    Parameters:
        frame (numpy array): The processed video frame.
        track_id (int): Unique ID of the tracked object.
        height (float): Estimated object height.
        is_multicam (bool): Indicates if multiple cameras are used.
        camera_name (str): Name of the camera capturing the violation.
    """
    try:
        timestamp = datetime.datetime.now()

        # Deteksi plat nomor menggunakan YOLOv8
        plates = detect_license_plate(frame)
        license_plate = extract_license_plate_text(frame, plates)

        # Convert frame to binary format for storage
        _, buffer = cv2.imencode('.jpg', frame)
        image_binary = buffer.tobytes()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Simpan gambar di GridFS
        image_id = fs.put(image_binary, filename=image_filename, content_type="image/jpeg")

        if is_multicam and check_id_exists(track_id) and camera_name == "Camera 1":
            collection.update_one(
                {"track_id": track_id},
                {"$push": {
                    "additional_cameras": {
                        "camera": camera_name,
                        "violationImageId": image_id,
                        "height": float(height)
                    }
                }}
            )
            is_multicam = False
        else:
            violation_data = {
                'timestamp': timestamp,
                'location': 'Jl. Pantura KM 23',
                'track_id': track_id,
                'license_plate': license_plate,  
                'camera': camera_name,
                'violationImageId': image_id,
                'height': float(height),
            }

            if is_multicam and camera_name != "Camera 1":
                violation_data['additional_cameras'] = []

            collection.insert_one(violation_data)
            print(f"Violation saved for Track ID {track_id} with License Plate: {license_plate}")

    except Exception as e:
        print(f"Error saving violation for Track ID {track_id}: {e}")
