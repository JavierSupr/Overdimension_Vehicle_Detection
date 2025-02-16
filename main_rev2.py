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
from typing import Tuple, Dict, List
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

WEBSOCKET_URL = "ws://localhost:8765"
WEBSOCKET_PORT = 8765

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
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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

async def process_and_stream_frames(video_port, result_port, camera_name, queue):
    """Handles video processing and adds frames to queue instead of blocking WebSocket."""
    
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind(("0.0.0.0", video_port))

    sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_result.bind(("0.0.0.0", result_port))

    while True:
        try:
            frame, frame_id = receive_video(sock_video, camera_name)
            if frame is None or frame_id is None:
                continue  # Skip if invalid frame

            # Receive inference results
            detections, processed_frame = apply_detections_and_bounding_box(sock_result, camera_name, frame_id, frame)
            if detections:
                tracks, keypoints, descriptor = process_tracks_and_extract_features(deepsort, detections, processed_frame)
                print('tes')
                print(tracks)                

            # **Push to queue (WebSocket process will send it)**
            if not queue.full():
                queue.put((camera_name, processed_frame))

            # Display processed frame
            cv2.imshow(camera_name, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"[ERROR] {camera_name} processing failed: {e}")

    sock_video.close()
    sock_result.close()
                    #if processed_frame is not None:
                    #await send_frame_via_websocket(websocket, processed_frame, camera_name)


                #ret, frame = cap.read()
                #frame = cv2.resize(frame, (640, 480))
                #result = model.predict(frame)[0]
                  # Fill frame queue
                #detections = apply_detections_and_bounding_box(frame,result)
                #if detections and not frame_queues[camera_name].empty():
                #    frame, _ = frame_queues[camera_name].get()  # Get the latest synchronized frame
#
                #    # Run tracking and get updated tracks
                #    tracks, keypoints, descriptor = process_tracks_and_extract_features(deepsort, detections, frame)
                #    if camera_name == "Camera 1":
                #        frame_tracked = draw_tracking_info(frame.copy(), tracks, is_cam1=True)
                #    elif camera_name == "Camera 2":
                #        frame_tracked = draw_tracking_info(frame.copy(), tracks, is_cam1=False)
                #    # Draw tracking IDs on frame
                #    for track in tracks:
                #        if not track.is_confirmed():
                #            continue
                #        
                #        ltrb = track.to_ltrb()  # Get bounding box
                #        track_id = track.track_id  # Get tracking ID
                #        # Draw bounding box
                #        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), 
                #                      (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                #        # Display tracking ID
                #        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 5),
                #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #    # Calculate FPS
                #    current_time = time.time()
                #    fps = 1 / (current_time - prev_time)
                #    prev_time = current_time
                #    # Display FPS on frame
                #    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                #    #cv2.imshow(camera_name, frame)
                #    #.waitKey(1)
                #    print("masuk")
#
                #    print("masuk2")
                #    #await send_frame_via_websocket(websocket, frame, camera_name)
                #    #await asyncio.sleep(0.05)
def start_websocket_process(queue):
    """Start the WebSocket process and pass the queue."""
    asyncio.run(websocket_sender(queue))

def start_process(video_port, result_port, camera_name, queue):
    """Wrapper to run async function inside a multiprocessing process."""
    asyncio.run(process_and_stream_frames(video_port, result_port, camera_name, queue))

if __name__ == "__main__":
    try:
        frame_queue = multiprocessing.Queue(maxsize=10)  # Shared queue

        process_websocket = multiprocessing.Process(target=start_websocket_process, args=(frame_queue,))
        process1 = multiprocessing.Process(target=start_process, args=(PORT_1, PORT_RESULT_1, "Camera 1", frame_queue))
        process2 = multiprocessing.Process(target=start_process, args=(PORT_2, PORT_RESULT_2, "Camera 2", frame_queue))

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

