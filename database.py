import datetime
import cv2
from pymongo import MongoClient
from gridfs import GridFS
import numpy as np

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['traffic_violations']
collection = db['violations']  # Main collection for metadata
fs = GridFS(db)  # GridFS setup
location = 'Jl. Pantura KM 23'
reference_number = 'XX123456'

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

def save_violation_to_mongodb(frame, track_id, height, reference_number):
    """
    Saves violation data to MongoDB, including an image with a mask drawn.

    Parameters:
        frame (numpy array): The processed video frame.
        track_id (int): Unique ID of the tracked object.
        height (float): Estimated object height.
        reference_number (str): Reference number for the record.
    """
    try:
        if check_id_exists(track_id):
            return #print(f"Violation for Track ID {track_id} already exists. Skipping save.")

        # Convert frame to binary format for storage
        _, buffer = cv2.imencode('.jpg', frame)
        image_binary = buffer.tobytes()

        # Generate timestamp and image filename
        timestamp = datetime.datetime.now()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Save the image in GridFS
        image_id = fs.put(image_binary, filename=image_filename, content_type="image/jpeg")

        # Prepare violation data
        violation_data = {
            'timestamp': timestamp,
            'location': 'Jl. Pantura KM 23',  # Fixed location
            'Reference Number': reference_number,
            'licensePlate': f"ID_{track_id}",
            'camera': "Camera 1",
            'violationImageId': image_id,  # Reference to the GridFS file ID
            'length': float(height),
            'width': float(1.8)
        }

        # Insert the violation record into the MongoDB collection
        collection.insert_one(violation_data)

        print(f"Violation saved for Track ID {track_id} with Image ID {image_id}")

    except Exception as e:
        print(f"Error saving violation for Track ID {track_id}: {e}")