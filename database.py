import datetime
import cv2
from pymongo import MongoClient
from gridfs import GridFS
import numpy as np
import traceback
from ocr_license_plate import detect_license_plate, extract_license_plate_text

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['traffic_violations']
collection = db['violations']  # Main collection for metadata
fs = GridFS(db)  # GridFS setup
location = 'Jl. Pantura KM 23'

#model = YOLO("best.pt")

def check_id_exists(track_id, camera_name=None):
    """Check if a track_id already exists in MongoDB and optionally check for a specific camera name."""
    query = {'track_id': track_id}
    
    if camera_name:
        query['camera'] = camera_name  # Ensure the main camera field matches
    
    return collection.find_one(query)

def draw_mask_on_detected_tracks(frame, tracked_objects, track_id):
    """
    Draws the segmentation mask for detected objects on the frame with different colors based on class names.
    Ensures that each track_id is drawn only once.

    Parameters:
        frame (numpy array): The current video frame.
        tracked_objects (list): List of detected objects with masks.
    """
    import cv2
    import numpy as np

    # Define colors for each class
    class_colors = {
        "Tampak Depan": (255, 0, 0),  # Blue
        "Tampak Samping": (0, 255, 0),  # Green
        "Truk": (0, 0, 255)  # Red
    }
    truck_bbox = []
    #print(f" tracked object {tracked_objects}")

    for track in tracked_objects:
        if track['track_id'] == track_id:
        #print(f"track {track}")
            track_id = track['track_id']
            
            mask = track['mask']
            #print(f"track.get mask {track['mask']}")
            #print(f"track_id2 {track['track_id']}")
            class_name = track.get('class_name')
            #print(f"track.get('class_name') {track.get('class_name')}")

            if mask is not None and len(mask) > 0 and class_name in class_colors:
                color = class_colors[class_name]

                mask = np.array(mask, dtype=np.int32)  # Convert list to np.array with int32
                mask = mask.reshape((-1, 1, 2))   
                # Create an overlay for transparency
                overlay = frame.copy()
                cv2.fillPoly(overlay, [mask], color)  # Draw mask
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # Blend mask with frame
                # Draw bounding box
                x1, y1, x2, y2 = track.get("bounding box")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                height, width = frame.shape[:2]
                if class_name == "Tampak Depan":
                # Clamp values to stay within the image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    truck_bbox.append((x1, y1, x2, y2))


    return frame, truck_bbox



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

def save_violation_to_mongodb(frame_captured, track_id, height, is_multicam, camera_name, truck_bbox, frame):
    """
    Saves violation data to MongoDB, including an image with a mask drawn and license plate detection.
    """
    try:
        timestamp = datetime.datetime.now()
        #print(f"truck bbox {truck_bbox}")
        # Deteksi plat nomor
        plates = detect_license_plate(frame, truck_bbox)
        #print(f"plates {plates} - {camera_name}")

        license_plate_text, license_plate_img = extract_license_plate_text(frame, plates)

        # Convert full frame to binary
        _, buffer = cv2.imencode('.jpg', frame_captured)
        image_binary = buffer.tobytes()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        # Simpan gambar utama ke GridFS
        image_id = fs.put(image_binary, filename=image_filename, content_type="image/jpeg")

        # Simpan license plate image jika ada
        licensenumber_id = None
        if license_plate_img is not None:
            _, lp_buffer = cv2.imencode('.jpg', license_plate_img)
            lp_binary = lp_buffer.tobytes()
            lp_filename = f"plate_{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            licensenumber_id = fs.put(lp_binary, filename=lp_filename, content_type="image/jpeg")

        existing_data = check_id_exists(track_id)

        #if is_multicam:
            # Existing document update logic...
        if existing_data is not None:
            if camera_name == "Camera 1":
                existing_doc = collection.find_one({"track_id": track_id})
                if existing_doc and existing_doc.get("camera") != camera_name:
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
                existing_doc = collection.find_one({"track_id": track_id})
                if existing_doc and existing_doc.get("camera") != camera_name:
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
            print(f"{camera_name} masuk sebagai main data (multicam)")
            violation_data = {
                'timestamp': timestamp,
                'location': 'Jl. Pantura KM 23',
                'track_id': track_id,
                'license_plate': license_plate_text,
                'camera': camera_name,
                'violationImageId': image_id,
                'licensenumberid': licensenumber_id,  # 🔑 store license plate image ID here
                'height': float(height),
                'additional_cameras': []
            }
            collection.insert_one(violation_data)
            print(f"Violation saved for Track ID {track_id} with License Plate: {license_plate_text}")


        #else:
        #    # Insert new document
        #    print(f"{camera_name} masuk sebagai main data")
        #    violation_data = {
        #        'timestamp': timestamp,
        #        'location': 'Jl. Pantura KM 23',
        #        'track_id': track_id,
        #        'license_plate': license_plate_text,
        #        'camera': camera_name,
        #        'violationImageId': image_id,
        #        'licensenumberid': licensenumber_id,  # 🔑 store license plate image ID here
        #        'height': float(height),
        #        'additional_cameras': []
        #    }
        #    collection.insert_one(violation_data)
        #    print(f"Violation saved for Track ID {track_id} with License Plate: {license_plate_text}")

    except Exception as e:
        print(f"Error saving violation for Track ID {track_id} {camera_name}: {e}")
        traceback.print_exc()  
