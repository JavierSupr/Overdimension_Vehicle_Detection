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
from matching_and_tracking import process_tracks_and_extract_features, match_features, draw_tracking_info
#from stream_handler import send_frame_via_websocket
from id_merging import merge_track_ids
from database import save_violation_to_mongodb, check_id_exists
from deep_sort_realtime.deepsort_tracker import DeepSort
#from ultralytics import YOLO
from typing import Tuple, Dict, List
#from stream_handler import send_frame_via_websocket
import base64


PORT_1 = 5010  # Port for first video stream
PORT_2 = 5011  # Port for second video stream
PORT_RESULT_1 = 5012
PORT_RESULT_2 = 5013
BUFFER_SIZE = 65536  # Max UDP packet size
UDP_IP = "0.0.0.0"
deepsort = DeepSort(max_age=3)

iou_threshold = 0.4
processed_tracks = set()  
estimated_height = 1.8
# Frame buffer to sync detections
frame_queues = {
    "Camera 1": queue.Queue(maxsize=5),
    "Camera 2": queue.Queue(maxsize=5)
}
WEBSOCKET_URL = "ws://localhost:8765"
WEBSOCKET_PORT = 8765
connected_clients = set()

async def websocket_handler(websocket, path):
    """Handles incoming WebSocket connections and stores them in a set."""
    connected_clients.add(websocket)
    print(f"[INFO] New WebSocket client connected. Total clients: {len(connected_clients)}")
    try:
        async for message in websocket:
            print(f"[WebSocket] Received: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] WebSocket client disconnected.")
    finally:
        connected_clients.remove(websocket)

async def start_websocket_server():
    """Starts the WebSocket server."""
    print(f"[INFO] WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
    async with websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT):
        await asyncio.Future()  # Keep the server running

#async def send_frame_via_websocket(frame, camera_name):
#    """Encodes a frame and sends it to all connected WebSocket clients."""
#    if not connected_clients:
#        return  # No clients connected, skip sending
#
#    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#    frame_base64 = base64.b64encode(buffer).decode('utf-8')
#    message = json.dumps({
#        "type": "frame",
#        "camera": camera_name,
#        "data": frame_base64
#    })
#
#    # Send frame to all connected clients
#    disconnected_clients = []
#    for client in connected_clients:
#        try:
#            await client.send(message)
#        except websockets.exceptions.ConnectionClosed:
#            disconnected_clients.append(client)
#
#    # Remove disconnected clients
#    for client in disconnected_clients:
#        connected_clients.remove(client)


async def send_frame_via_websocket(websocket, frame, camera_name):
    """Send an encoded frame through WebSocket only if the connection is active."""
    try:
        print("[INFO] Encoding frame...")
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        if buffer is None:
            print("[ERROR] Frame encoding failed, skipping frame...")
            return
        
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        message = json.dumps({
            "type": "frame",
            "camera": camera_name,
            "data": frame_base64
        })

        print("[INFO] Sending frame via WebSocket...")
        await websocket.send(message)

    except cv2.error as e:
        print(f"[ERROR] OpenCV encoding error: {e}")
    except Exception as e:
        print(f"[ERROR] WebSocket sending failed: {e}")

import struct
import queue

BUFFER_SIZE = 65536  # Define buffer size for UDP packets
frame_queues = {}  # Ensure this exists before usage

def receive_video(sock, camera_name):
    """Receives video frames via UDP and stores them in a queue."""
    try:
        chunks = []
        expected_chunks = None
        max_attempts = 10  # Prevent infinite loop
        sock.settimeout(0.05)  # Prevents blocking

        # Wait for the expected chunk count
        for _ in range(max_attempts):
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                if len(data) == 1:  # Expected to receive a single byte
                    expected_chunks = struct.unpack("B", data)[0]
                    break
            except socket.timeout:
                return None  # Skip frame if timeout

        if expected_chunks is None:
            print("[WARNING] No chunk count received, skipping frame...")
            return None

        # Receive frame chunks
        for _ in range(expected_chunks):
            try:
                chunk, _ = sock.recvfrom(BUFFER_SIZE)
                chunks.append(chunk)
            except socket.timeout:
                print("[WARNING] Incomplete frame received, skipping...")
                return None  # Skip incomplete frames

        # Merge and decode the frame
        frame_data = b"".join(chunks)
        np_data = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            print("[ERROR] Frame decoding failed, skipping frame...")
            return None

        # Ensure frame queue exists for this camera
        if camera_name not in frame_queues:
            frame_queues[camera_name] = queue.Queue(maxsize=10)

        # Add frame to buffer queue with timestamp
        if not frame_queues[camera_name].full():
            frame_queues[camera_name].put((frame, time.time()))

        return frame

    except Exception as e:
        print(f"[ERROR] Receiving video from {camera_name} failed: {e}")
        return None



#def detect_objects(video_path, verbose=False):
#    # Load YOLO model
#    model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for a small model
#    
#    # Open video capture
#    cap = cv2.VideoCapture(video_path)
#    detections = []
#    
#    while cap.isOpened():
#        ret, frame = cap.read()
#        if not ret:
#            break
#        
#        # Run YOLO inference
#        results = model(frame, verbose=False)[0]
#        
#        frame_detections = []
#        for det in results.boxes.data:
#            xmin, ymin, xmax, ymax, conf, cls = det.cpu().numpy()
#            class_name = results.names[int(cls)]
#            frame_detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf, class_name))
#        
#        detections.append(frame_detections)  # Append per-frame detections to the list
#  
#    cap.release()
#    cv2.destroyAllWindows()
#    
#    if verbose:
#        print(detections)
#    
#    return detections


def apply_detections_and_bounding_box(sock_result, camera_name):
    """Receives detections via UDP and applies them to the latest matching video frame."""
    detections = []
    sock_result.settimeout(0.05)  # Avoid blocking

    try:
        data, _ = sock_result.recvfrom(65535)
        message = json.loads(data.decode())

        if not frame_queues[camera_name].empty():
            frame, frame_time = frame_queues[camera_name].get()  # Get the oldest frame
            
            # Draw bounding boxes
            for obj in message["objects"]:
                class_name = obj["label"]
                conf = float(obj["conf"])
                bbox = obj["bbox"]
                xmin, ymin, xmax, ymax = map(int, bbox)
                detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf, class_name))

                color = (0, 255, 0) if class_name == "Truk" else (255, 0, 0)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display synchronized frame with bounding boxes
            cv2.imshow(camera_name, frame)
            cv2.waitKey(1)

    except socket.timeout:
        pass  # No detections received

    return detections

#def apply_detections_and_bounding_box(
#    frame: np.ndarray, 
#    results, 
#    #class_names: List[str]
#) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
#    """
#    Apply bounding boxes and labels to a frame.
#    """
#    annotated_frame = frame.copy()
#    detections = []
#    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
#        #class_name = class_names[int(cls)]
#        box = box.cpu().numpy().astype(int)
#        xmin, ymin, xmax, ymax = box
#        width = xmax - xmin
#        height = ymax - ymin
#        conf = float(conf)
#        cls = int(cls)
#
#        detections.append(([xmin, ymin, width, height], conf, cls))
#        #print(f"detections{detections}")
#        #color = (0, 255, 0)
#        #cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
#        #label = f'{class_name} {conf:.2f}'
#        #cv2.putText(
#        #    annotated_frame, label, (box[0], box[1] - 10),
#        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
#        #)
#
#    return  detections

def find_root(id_group, track_id):
        """Find the root ID for a given track ID."""
        if track_id not in id_group:
            return track_id
        while id_group[track_id] != track_id:
            track_id = id_group[track_id]
        return track_id

async def process_and_stream_frames(video_port, result_port, camera_name):
    """Handles video processing, tracking, and streaming for a single camera."""
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind((UDP_IP, video_port))

    sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_result.bind((UDP_IP, result_port))

    deepsort = DeepSort(max_age=5)
    websocket_url = "ws://localhost:8765"
    #model = YOLO("yolov8n.pt")
    #cap = cv2.VideoCapture(video_path)

    prev_time = time.time()
    fps = 0
    try:
        async with websockets.connect(websocket_url) as websocket:
            while True:
                frame = receive_video(sock_video, camera_name)  # Fill frame queue
                if frame is None:
                    continue
                detections = apply_detections_and_bounding_box(sock_result, camera_name)
                #await send_frame_via_websocket(websocket, frame, camera_name)


                #ret, frame = cap.read()
                #frame = cv2.resize(frame, (640, 480))
                #result = model.predict(frame)[0]
                  # Fill frame queue
                #detections = apply_detections_and_bounding_box(frame,result)
                if detections and not frame_queues[camera_name].empty():
                    frame, _ = frame_queues[camera_name].get()  # Get the latest synchronized frame

                    # Run tracking and get updated tracks
                    tracks, keypoints, descriptor = process_tracks_and_extract_features(deepsort, detections, frame)
                    if camera_name == "Camera 1":
                        frame_tracked = draw_tracking_info(frame.copy(), tracks, is_cam1=True)
                    elif camera_name == "Camera 2":
                        frame_tracked = draw_tracking_info(frame.copy(), tracks, is_cam1=False)
                    # Draw tracking IDs on frame
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        
                        ltrb = track.to_ltrb()  # Get bounding box
                        track_id = track.track_id  # Get tracking ID
                        # Draw bounding box
                        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), 
                                      (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                        # Display tracking ID
                        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    # Display FPS on frame
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    #cv2.imshow(camera_name, frame)
                    #.waitKey(1)
                    print("masuk")

                    print("masuk2")
                    #await send_frame_via_websocket(websocket, frame, camera_name)
                    #await asyncio.sleep(0.05)
    except websockets.exceptions.ConnectionClosed:
        print("[ERROR] WebSocket connection lost.")
    except Exception as e:
        print(f"[ERROR] Unexpected error in sender: {e}")
    finally:
        sock_video.close()
        sock_result.close()

def start_process(video_port, result_port, camera_name):
    """Wrapper to run async function inside a multiprocessing process."""
    asyncio.run(process_and_stream_frames(video_port, result_port, camera_name))

if __name__ == "__main__":
    video_path = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4"

    #video_path = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Coba coba/DeepSORT - YOLOv8 - Object Measurement/Documentation/video.mp4"
    try:
                                                                
        #process1 = process_and_stream_frames(PORT_1, PORT_RESULT_1,"Camera 1", video_path)    
        process1 = multiprocessing.Process(target=start_process, args=(PORT_1, PORT_RESULT_1, "Camera 1"))
        process2 = multiprocessing.Process(target=start_process, args=(PORT_2, PORT_RESULT_2, "Camera 2"))

        #process2 = multiprocessing.Process(target=process_camera, args=(PORT_2, PORT_RESULT_2, "Camera 2", video_path))

        process1.start()
        process2.start()

        process1.join()
        process2.join()

    except KeyboardInterrupt:
        print("\n[INFO] Receiver program stopped by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] All processes closed. Exiting...")
