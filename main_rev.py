import socket
import struct
import numpy as np
import cv2
import json
import asyncio
import websockets
import threading
import base64
from deep_sort_realtime.deepsort_tracker import DeepSort
from matching_and_tracking import process_tracks_and_extract_features, match_features

PORT_1 = 5005
PORT_2 = 5006
PORT_RESULT_1 = 5007
PORT_RESULT_2 = 5008
BUFFER_SIZE = 65536
UDP_IP = "0.0.0.0"

deepsort1 = DeepSort(max_age=5)
deepsort2 = DeepSort(max_age=5)
sift = cv2.SIFT_create()
iou_threshold = 0.2
processed_tracks = set()
estimated_height = 1.8

global descriptors1, descriptors2, tracks1, tracks2, keypoints1, keypoints2

descriptors1 = None
descriptors2 = None
tracks1 = None
tracks2 = None
keypoints1 = None
keypoints2 = None

# Events to synchronize threads
camera1_ready = threading.Event()
camera2_ready = threading.Event()

sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.bind((UDP_IP, PORT_1))

sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.bind((UDP_IP, PORT_2))

sock_result1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_result1.bind((UDP_IP, PORT_RESULT_1))

sock_result2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_result2.bind((UDP_IP, PORT_RESULT_2))

async def send_frame_via_websocket(frame, camera_name):
    """Send an encoded frame through WebSocket."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    message = json.dumps({
        "type": "frame",
        "camera": camera_name,
        "data": frame_base64
    })
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send(message)

def receive_video(sock, camera_name):
    try:
        print("Receiving video from", camera_name)
        
        data, _ = sock.recvfrom(BUFFER_SIZE)
        if len(data) == 1:
            chunks_count = struct.unpack("B", data)[0]
        
        chunks = []
        for _ in range(chunks_count):
            chunk, _ = sock.recvfrom(BUFFER_SIZE)
            chunks.append(chunk)
        
        frame_data = b"".join(chunks)
        np_data = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        
        return frame
    except Exception as e:
        print(f"[ERROR] Receiving video from {camera_name} failed: {e}")

def apply_detections_and_bounding_box(sock_result, frame, camera_name):
    detections = []
    try:
        print("Receiving detection results")
        data, _ = sock_result.recvfrom(65535)
        message = json.loads(data.decode())
        
        if frame is not None:
            for obj in message["objects"]:
                class_name = obj["label"]
                conf = float(obj["conf"])
                bbox = obj["bbox"]
                
                xmin, ymin, xmax, ymax = map(int, bbox)
                width, height = xmax - xmin, ymax - ymin
                
                detections.append(([xmin, ymin, width, height], conf, class_name))
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} ({conf:.2f})",
                    (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            
            cv2.imshow(camera_name, frame)
            cv2.waitKey(1)
    except Exception as e:
        print(f"[ERROR] Processing detections failed: {e}")
    
    return detections

def thread_camera1():
    global descriptors1, tracks1, keypoints1
    while True:
        frame1 = receive_video(sock1, "Camera 1")
        if frame1 is not None:
            detections1 = apply_detections_and_bounding_box(sock_result1, frame1, "Camera 1")
            tracks1, keypoints1, descriptors1 = process_tracks_and_extract_features(deepsort1, detections1, frame1)
            camera1_ready.set()
            asyncio.run(send_frame_via_websocket(frame1, "Camera 1"))

def thread_camera2():
    global descriptors2, tracks2, keypoints2
    while True:
        frame2 = receive_video(sock2, "Camera 2")
        if frame2 is not None:
            detections2 = apply_detections_and_bounding_box(sock_result2, frame2, "Camera 2")
            tracks2, keypoints2, descriptors2 = process_tracks_and_extract_features(deepsort2, detections2, frame2)
            camera2_ready.set()
            asyncio.run(send_frame_via_websocket(frame2, "Camera 2"))

def process_matching():
    global descriptors1, descriptors2, tracks1, tracks2, keypoints1, keypoints2
    camera1_ready.wait()
    camera2_ready.wait()
    good_matches = match_features(descriptors1, descriptors2, tracks1, tracks2, keypoints1, keypoints2)

def start_threads():
    threading.Thread(target=thread_camera1).start()
    threading.Thread(target=thread_camera2).start()
    threading.Thread(target=process_matching).start()

def websocket_server():
    async def handler(websocket, path):
        print("WebSocket client connected")
        while True:
            await asyncio.sleep(1)
    
    start_server = websockets.serve(handler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    start_threads()
    #websocket_server()
