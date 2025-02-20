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

def draw_mask_on_detected_tracks(frame, updated_tracks, detected_track_ids):
    """
    Draws the mask for each detected track_id on the given frame with different colors based on class.

    Parameters:
        frame (numpy.ndarray): The frame where the masks will be drawn.
        updated_tracks (list): List of dictionaries containing track info (track_id, class_name, mask).
        detected_track_ids (list): List of track_ids that were detected.

    Returns:
        numpy.ndarray: The frame with masks drawn.
    """
    # Define colors for each class
    class_colors = {
        "Tampak Depan": (255, 0, 0),   # Red
        "Tampak Samping": (0, 255, 0), # Green
        "Truk": (0, 0, 255)            # Blue
    }

    for track in updated_tracks:
        if isinstance(track, dict):  # Ensure track is a dictionary
            track_id = track.get("track_id")
            class_name = track.get("class_name", "Unknown")
            mask = track.get("mask")

            if track_id in detected_track_ids and isinstance(mask, np.ndarray):
                color = class_colors.get(class_name, (255, 255, 255))  # Default: White

                # Ensure mask dimensions match the frame
                if mask.shape[:2] == frame.shape[:2]:
                    frame[mask > 0] = frame[mask > 0] * 0.5 + np.array(color) * 0.5

                    # Draw bounding box and ID text
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #if contours:
                    #    x, y, w, h = cv2.boundingRect(contours[0])
                        #cv2.putText(frame, f"{class_name} (ID {track_id})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        #            0.7, color, 2, cv2.LINE_AA)
        else:
            print("Warning: Track data is not a dictionary:", track)

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

def save_violation_to_mongodb(track_id, height, frame, updated_tracks):
    """Save violation data and image to MongoDB, with masks drawn for detected tracks."""
    try:
        # Check if ID already exists
        #if check_id_exists(track_id):
        #    print(f"ID_{track_id} already exists in database, skipping...")
        #    return None
        height = 2.2
        # Get all detected track IDs for drawing masks
        detected_track_ids = [track_id]
        # Convert tracked_objects to the correct format
        updated_tracks = convert_tracked_objects_to_dict(updated_tracks)

        # Now call the function with the corrected format
        frame_with_mask = draw_mask_on_detected_tracks(frame, updated_tracks, detected_track_ids)

        if isinstance(height, dict):  # Extract value if height is a dictionary
            height = height.get("value", 0)

        # Convert height to float safely
        height = float(height)
        
        # Convert the frame to binary data
        _, buffer = cv2.imencode('.jpg', frame_with_mask)
        image_binary = buffer.tobytes()

        # Generate timestamp and image filename
        timestamp = datetime.datetime.now()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Save the image in GridFS
        image_id = fs.put(image_binary, filename=image_filename, content_type="image/jpeg")

        # Prepare violation data
        violation_data = {
            'timestamp': timestamp,
            'location' : location,
            'Reference Number' : reference_number,
            'licensePlate': f"ID_{track_id}",
            'camera': "Camera 1",
            'violationImageId': image_id,  # Reference to the GridFS file ID
            'length': float(height),
            'width': float(1.8)
        }

        # Insert metadata into the main collection
        result = collection.insert_one(violation_data)
        print(f"Saved new violation with ID: {result.inserted_id}")
        
        return result.inserted_id
    except Exception as e:
        print(f"Error saving violation: {e}")
        return None
