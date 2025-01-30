import datetime
import cv2
import os
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

def save_violation_to_mongodb(track_id, height, frame):
    """Save violation data and image to MongoDB."""
    try:
        # Check if ID already exists
        if check_id_exists(track_id):
            print(f"ID_{track_id} already exists in database, skipping...")
            return None

        # Convert the frame to binary data
        _, buffer = cv2.imencode('.jpg', frame)
        decoded_image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("output.jpg", decoded_image)

        image_binary = buffer.tobytes()

        # Generate timestamp and image filename
        timestamp = datetime.datetime.now()
        image_filename = f"{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Save the image in GridFS
        image_id = fs.put(image_binary, filename=image_filename, content_type="image/jpeg")
        print(f"Image saved to GridFS with ID: {image_id}")

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

