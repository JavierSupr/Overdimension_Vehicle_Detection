import cv2
import socket
import numpy as np
import struct
import json
import multiprocessing
import queue
import time
from matching_and_tracking import process_tracks_and_extract_features, match_features, draw_tracking_info
from deep_sort_realtime.deepsort_tracker import DeepSort

PORT_1 = 5010  # Port for first video stream
PORT_2 = 5011  # Port for second video stream
PORT_RESULT_1 = 5012
PORT_RESULT_2 = 5013
BUFFER_SIZE = 65536  # Max UDP packet size
UDP_IP = "0.0.0.0"

# Frame buffer to sync detections
frame_queues = {
    "Camera 1": queue.Queue(maxsize=5),
    "Camera 2": queue.Queue(maxsize=5)
}

def receive_video(sock, camera_name):
    """Receives video frames via UDP and stores them in a queue."""
    try:
        chunks = []
        expected_chunks = None
        sock.settimeout(0.05)  # Prevents blocking

        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                if len(data) == 1:
                    expected_chunks = struct.unpack("B", data)[0]
                    break
            except socket.timeout:
                return None  # Skip frame if timeout

        for _ in range(expected_chunks):
            try:
                chunk, _ = sock.recvfrom(BUFFER_SIZE)
                chunks.append(chunk)
            except socket.timeout:
                return None  # Skip frame if incomplete

        frame_data = b"".join(chunks)
        np_data = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        # Add frame to buffer queue with timestamp
        if not frame_queues[camera_name].full():
            frame_queues[camera_name].put((frame, time.time()))

        return frame

    except Exception as e:
        print(f"[ERROR] Receiving video from {camera_name} failed: {e}")
        return None

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

def process_camera(port_video, port_result, camera_name):
    """Handles video reception, tracking, and synchronized bounding box display."""
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind((UDP_IP, port_video))

    sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_result.bind((UDP_IP, port_result))

    deepsort = DeepSort(max_age=5, embedder_gpu=True, embedder="mobilenet")

    while True:
        receive_video(sock_video, camera_name)  # Fill frame queue
        detections = apply_detections_and_bounding_box(sock_result, camera_name)
        
        if detections and not frame_queues[camera_name].empty():
            frame, _ = frame_queues[camera_name].get()  # Get the latest synchronized frame
            tracks, keypoints, descriptor = process_tracks_and_extract_features(deepsort, detections, frame)  # Process tracking
            draw_tracking_info(frame,tracks, is_cam1=True)
    sock_video.close()
    sock_result.close()

if __name__ == "__main__":
    try:    
        process1 = multiprocessing.Process(target=process_camera, args=(PORT_1, PORT_RESULT_1, "Camera 1"))
        process2 = multiprocessing.Process(target=process_camera, args=(PORT_2, PORT_RESULT_2, "Camera 2"))

        process1.start()
        process2.start()

        process1.join()
        process2.join()

    except KeyboardInterrupt:
        print("\n[INFO] Receiver program stopped by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] All processes closed. Exiting...")
