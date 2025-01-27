import datetime
import cv2
import os
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['traffic_violations']
SAVE_DIR = "captured_violations"
collection = db['violations']


def check_id_exists(track_id):
    """Check if a track_id already exists in MongoDB."""
    return collection.find_one({'licensePlate': f"ID_{track_id}"}) is not None


def save_violation_to_mongodb(track_id, height, frame):
    """Save violation data to MongoDB and capture image if ID doesn't exist."""
    try:
        # Check if ID already exists
        if check_id_exists(track_id):
            print(f"ID_{track_id} already exists in database, skipping...")
            return None
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        # Generate filename for captured image
        timestamp = datetime.datetime.now()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(SAVE_DIR, image_filename)
        
        # Capture and save the image
        cv2.imwrite(image_path, frame)
        
        # Prepare violation data
        violation_data = {
            'licensePlate': f"ID_{track_id}",
            'timestamp': timestamp,
            'camera': "Camera 1",
            'violationImage': image_filename,
            'length': float(height),
            'width': float(1.8)
        }
        
        # Insert into MongoDB
        result = collection.insert_one(violation_data)
        print(f"Saved new violation with ID: {result.inserted_id}")
        
        return result.inserted_id
    except Exception as e:
        print(f"Error saving violation: {e}")
        return None