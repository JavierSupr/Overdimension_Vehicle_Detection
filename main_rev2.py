import cv2
import socket
import numpy as np
import struct
import json
import multiprocessing
import queue
import time
import struct
import asyncio
import websockets
from matching_and_tracking import process_tracks_and_extract_features, match_features, draw_tracking_info, compute_reference_height, estimate_height
#from stream_handler import send_frame_via_websocket
from id_merging import merge_track_ids
from database import save_violation_to_mongodb, check_id_exists
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Tuple, Dict, List
import base64
from ultralytics import YOLO
from dimension_estimation import calculate_distance, calculate_vehicle_height


PORT_1 = 5010  # Port for first video stream
PORT_2 = 5011  # Port for second video stream
PORT_RESULT_1 = 5012
PORT_RESULT_2 = 5013
BUFFER_SIZE = 65536  # Max UDP packet size
UDP_IP = "0.0.0.0"
deepsort = DeepSort(max_age=10)

iou_threshold = 0.4
processed_tracks = set()  
#estimated_height = 1.8

WEBSOCKET_URL = "ws://localhost:8765"
WEBSOCKET_PORT = 8765

id_mappings = {}

async def websocket_sender(queue):
    """Sends frames from the queue to WebSocket clients asynchronously."""
    while True:
        try:
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                while True:
                    if not queue.empty():
                        camera_name, frame = queue.get()

                        # Encode frame
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')

                        message = json.dumps({
                            "type": "frame",
                            "camera": camera_name,
                            "data": frame_base64
                        })

                        await asyncio.wait_for(websocket.send(message), timeout=1)  # Non-blocking send

        except websockets.exceptions.ConnectionClosed:
            print("[ERROR] WebSocket connection lost, reconnecting...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[ERROR] WebSocket error: {e}")
            await asyncio.sleep(5)

def receive_video(sock, camera_name):
    """Receives video frames via UDP and synchronizes using frame_id."""
    try:
        chunks = []
        expected_chunks = None
        frame_id = None
        sock.settimeout(0.05)  # Prevents blocking

        # Receive frame ID first (4 bytes)
        try:
            frame_id_data, _ = sock.recvfrom(4)  # Receive 4-byte integer
            frame_id = struct.unpack("I", frame_id_data)[0]  # Unpack frame ID
        except socket.timeout:
            return None, None  # Skip if timeout

        # Receive the expected number of chunks
        try:
            data, _ = sock.recvfrom(1)  # 1 byte for chunk count
            expected_chunks = struct.unpack("B", data)[0]
        except socket.timeout:
            return None, None  # Skip if timeout

        if expected_chunks is None:
            print(f"[WARNING] {camera_name} - No chunk count received, skipping frame...")
            return None, None

        # Receive frame chunks
        for _ in range(expected_chunks):
            try:
                chunk, _ = sock.recvfrom(BUFFER_SIZE)
                chunks.append(chunk)
            except socket.timeout:
                print(f"[WARNING] {camera_name} - Incomplete frame received, skipping...")
                return None, None  # Skip incomplete frames

        # Merge and decode the frame
        frame_data = b"".join(chunks)
        np_data = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"[ERROR] {camera_name} - Frame decoding failed, skipping frame...")
            return None, None

        return frame, frame_id  # Return frame and its unique ID

    except Exception as e:
        print(f"[ERROR] Receiving video from {camera_name} failed: {e}")
        return None

def apply_detections_and_bounding_box(sock_result, camera_name, frame_id, frame):
    """Receives detections via UDP and applies them to the matching video frame using frame_id."""
    detections = []
    sock_result.settimeout(0.05)  # Prevent blocking

    try:
        data, _ = sock_result.recvfrom(65535)
        message = json.loads(data.decode())

        received_frame_id = message.get("frame_id", None)  # Extract frame ID from JSON

        # Match frame_id with the received detection results
        if received_frame_id == frame_id:
            for obj in message["objects"]:
                class_name = obj["label"]
                conf = float(obj["conf"])
                bbox = obj["bbox"]
                xmin, ymin, xmax, ymax = map(int, bbox)
                detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf, class_name))

                color = (0, 255, 0) if class_name == "Truk" else (255, 0, 0)
                #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                #cv2.putText(frame, f"{class_name} ({conf:.2f})", (xmin, ymin - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        return detections, frame  # Return modified frame with detections

    except socket.timeout:
        return [], frame  # Return frame as is if no detections received

def find_root(id_group, track_id):
        """Find the root ID for a given track ID."""
        if track_id not in id_group:
            return track_id
        while id_group[track_id] != track_id:
            track_id = id_group[track_id]
        return track_id

def process_and_stream_frames(video_port, result_port, camera_name, queue, video_path, shared_data):
    """Handles video processing and adds frames to queue instead of blocking WebSocket."""
  
    #sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock_video.bind(("0.0.0.0", video_port))
#
    #sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock_result.bind(("0.0.0.0", result_port))
    #estimated_heights = 2
    model = YOLO("best.pt") 
    cap = cv2.VideoCapture(video_path)
    
    while True:
        #try:
            #frame, frame_id = receive_video(sock_video, camera_name)
            #if frame is None or frame_id is None:
            #    continue  # Skip if invalid frame

            ## Receive inference results
            #detections, processed_frame = apply_detections_and_bounding_box(sock_result, camera_name, frame_id, frame)
                 # Use 'yolov8n.pt' for a small model

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640,480))
            # Run YOLO inference
            results = model(frame, verbose=False)[0]
            #print("sini 3")
            detections = []
            tracked_objects1, keypoints1, descriptors1 = None, None, None  
            tracked_objects2, keypoints2, descriptors2 = None, None, None  

            for det in results.boxes.data:
                xmin, ymin, xmax, ymax, conf, cls = det.cpu().numpy()
                class_name = results.names[int(cls)]

                # Check if masks exist
                mask = results.masks.xy if results.masks is not None else None

                detections.append((
                    [xmin, ymin, xmax - xmin, ymax - ymin],  # Bounding box
                    conf,  # Confidence score
                    class_name,  # Class name
                    mask  # Mask (None if not available)
                ))

            if detections:
                if camera_name == "Camera 1":
                    tracked_objects1, keypoints1, descriptors1 = process_tracks_and_extract_features(deepsort, detections, frame)
                    reference_height = compute_reference_height(tracked_objects1, detections)
                    estimated_height = estimate_height(tracked_objects1, reference_height)
                    frame = draw_tracking_info(frame, tracked_objects1, estimated_height)
                                    # Store data in shared dictionary
                    shared_data["tracked_objects1"] = tracked_objects1
                    shared_data["keypoints1"] = keypoints1
                    shared_data["descriptors1"] = descriptors1

                    # Check if Camera 2 data is available
                    if all(k in shared_data for k in ["tracked_objects2", "keypoints2", "descriptors2"]):
                        good_matches = match_features(
                            descriptors1, shared_data["descriptors2"],
                            tracked_objects1, shared_data["tracked_objects2"],
                            keypoints1, shared_data["keypoints2"]
                        )
                elif camera_name == "Camera 2":
                    tracked_objects2, keypoints2, descriptors2 = process_tracks_and_extract_features(deepsort, detections, frame)
                    reference_height = compute_reference_height(tracked_objects2, detections)
                    estimated_height = estimate_height(tracked_objects2, reference_height)
                    frame = draw_tracking_info(frame, tracked_objects2, estimated_height)

                    shared_data["tracked_objects2"] = tracked_objects2
                    shared_data["keypoints2"] = keypoints2
                    shared_data["descriptors2"] = descriptors2

                
                good_matches = match_features(descriptors1, descriptors2, tracked_objects1, tracked_objects2, keypoints1, keypoints2)


            if not queue.full():
                queue.put((camera_name, frame))

            # Display processed frame
            cv2.imshow(camera_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def start_websocket_process(queue):
    """Start the WebSocket process and pass the queue."""
    asyncio.run(websocket_sender(queue))

def start_process(video_port, result_port, camera_name, queue, video_path, shared_data):
    """Wrapper to run async function inside a multiprocessing process."""
    asyncio.run(process_and_stream_frames(video_port, result_port, camera_name, queue, video_path, shared_data))

if __name__ == "__main__":
    video_path = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/333 VID_20231011_170120.mp4"

    try:
        manager = multiprocessing.Manager()
        shared_data = manager.dict()  # Shared dictionary for inter-process communication
        frame_queue = multiprocessing.Queue(maxsize=10)  # Shared queue

        process_websocket = multiprocessing.Process(target=start_websocket_process, args=(frame_queue,))
        process1 = multiprocessing.Process(target=start_process, args=(PORT_1, PORT_RESULT_1, "Camera 1", frame_queue, video_path, shared_data))
        process2 = multiprocessing.Process(target=start_process, args=(PORT_2, PORT_RESULT_2, "Camera 2", frame_queue, video_path, shared_data))

        process_websocket.start()
        process1.start()
        process2.start()

        process1.join()
        process2.join()
        process_websocket.join()

    except KeyboardInterrupt:
        print("\n[INFO] Program stopped by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] All processes closed. Exiting...")

