import datetime
import cv2
from pymongo import MongoClient
from gridfs import GridFS
import numpy as np
import pytesseract
from ultralytics import YOLO
from ocr_license_plate import detect_license_plate, extract_license_plate_text

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['traffic_violations']
collection = db['violations']  # Main collection for metadata
fs = GridFS(db)  # GridFS setup
location = 'Jl. Pantura KM 23'

model = YOLO("best.pt")

def check_id_exists(track_id, camera_name=None):
    """Check if a track_id already exists in MongoDB and optionally check for a specific camera name."""
    query = {'track_id': track_id}
    
    if camera_name:
        query['camera'] = camera_name  # Ensure the main camera field matches
    
    return collection.find_one(query)

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
        #license_plate = 'B 1234 RF'

        # Convert frame to binary format for storage
        _, buffer = cv2.imencode('.jpg', frame)
        image_binary = buffer.tobytes()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Simpan gambar di GridFS
        image_id = fs.put(image_binary, filename=image_filename, content_type="image/jpeg")
        #if is_multicam and camera_name == "Camera 1":
        #    violation_data['additional_cameras'] = []  # Initialize empty list for additional cameras
        existing_data = check_id_exists(track_id)

        if is_multicam:
            if existing_data is not None:
                # There is already a main entry for this track_id
                if camera_name == "Camera 1":
                    # Check if there's already a document with this track_id
                    existing_doc = collection.find_one({"track_id": track_id})

                    # Proceed only if the current camera_name is different from the main camera in the existing document
                    if existing_doc and existing_doc.get("camera") != camera_name:
                        # Also check if this camera is already in additional_cameras to avoid duplicates
                        existing_camera1 = collection.find_one({
                            "track_id": track_id,
                            "additional_cameras.track_id": track_id
                        })

                        if not existing_camera1:
                            print("Camera 1 masuk sebagai additional")
                            collection.update_one(
                                {"track_id": track_id},
                                {"$push": {
                                    "additional_cameras": {
                                        "track_id": track_id,
                                        "camera": camera_name,
                                        "violationImageId": image_id,
                                        "height": float(height)
                                    }
                                }}
                            )
                        is_multicam = False
                elif camera_name == "Camera 2":
                    # Check if there's already a document with this track_id
                    existing_doc = collection.find_one({"track_id": track_id})

                    # Proceed only if the current camera_name is different from the main camera in the existing document
                    if existing_doc and existing_doc.get("camera") != camera_name:
                        # Also check if this camera is already in additional_cameras to avoid duplicates
                        existing_camera2 = collection.find_one({
                            "track_id": track_id,
                            "additional_cameras.track_id": track_id
                        })

                        if not existing_camera2:
                            print("Camera 2 masuk sebagai additional")
                            collection.update_one(
                                {"track_id": track_id},
                                {"$push": {
                                    "additional_cameras": {
                                        "track_id": track_id,
                                        "camera": camera_name,
                                        "violationImageId": image_id,
                                        "height": float(height)
                                    }
                                }}
                            )
                        is_multicam = False
        else:
            # No entry yet: Camera 1 or 2 will be inserted as the main data
            print(f"{camera_name} masuk sebagai main data")
            violation_data = {
                'timestamp': timestamp,
                'location': 'Jl. Pantura KM 23',
                'track_id': track_id,
                'license_plate': license_plate,  
                'camera': camera_name,
                'violationImageId': image_id,
                'height': float(height),
                'additional_cameras': []
            }
            collection.insert_one(violation_data)
            print(f"Violation saved for Track ID {track_id} with License Plate: {license_plate}")

    except Exception as e:
        print(f"Error saving violation for Track ID {track_id}: {e}")
