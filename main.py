import cv2
import socket
import numpy as np
import struct
import json
import multiprocessing
import time
import struct
import asyncio
import websockets
from matching_and_tracking import process_tracks_and_extract_features, match_features, draw_tracking_info
from id_merging import merge_track_ids
from database import save_violation_to_mongodb, check_id_exists, draw_mask_on_detected_tracks
from deep_sort_realtime.deepsort_tracker import DeepSort
import base64
from ultralytics import YOLO
from dimension_estimation import  compute_reference_height, estimate_height, get_final_estimated_heights
from collections import deque
import traceback
PORT_1 = 5019  
PORT_2 = 5020  
PORT_RESULT_1 = 5012
PORT_RESULT_2 = 5013
BUFFER_SIZE = 65536  
UDP_IP = "0.0.0.0"

WEBSOCKET_URL = "ws://localhost:8765"
WEBSOCKET_PORT = 8765

sift = cv2.SIFT_create()

detections_buffer = {}

DETECTION_BUFFER_SIZE = 10
detection_buffer = deque(maxlen=DETECTION_BUFFER_SIZE)
frame_buffer = deque(maxlen=10) 


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
            frame_id_data, _ = sock.recvfrom(4) 
            frame_id = struct.unpack("I", frame_id_data)[0]  
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
# Buffer to store last N frames' detections
def delayed_receive_video(sock, camera_name):
    frame, frame_id = receive_video(sock, camera_name)
    frame_buffer.append((frame, frame_id))
    frame_ids = [fid for _, fid in frame_buffer]
    print("Buffered frame_ids:", frame_ids)

    # Ambil frame sebelumnya (delay 1 frame)
    if len(frame_buffer) != 0:
        delayed_frame, delayed_id = frame_buffer[0]
        return delayed_frame, delayed_id

    else:
        return None, None
    
def extract_detections(sock_result, frame_id):
    """Receives detections via UDP and returns a list: [bbox, confidence, class_name, segmentation]."""
    global detection_buffer
    sock_result.settimeout(0.05)  # Prevent blocking
    detections = []

    try:
        data, _ = sock_result.recvfrom(65535)
        message = json.loads(data.decode())

        received_frame_id = message.get("frame_id", None)  # Extract frame ID
        if received_frame_id is not None:
            detection_buffer.append((received_frame_id, message["objects"]))
        for buffered_id, objects in list(detection_buffer):
            if buffered_id == frame_id:
                for obj in objects:
                    class_name = obj["label"]
                    conf = float(obj["conf"])
                    bbox = obj["bbox"]  # [x1, y1, x2, y2]
                    seg = obj.get("seg", None)

                    xmin, ymin, xmax, ymax = map(int, bbox)
                    width = xmax - xmin
                    height = ymax - ymin

                    detections.append((
                        [xmin, ymin, width, height],  # bbox
                        conf,                        # confidence
                        class_name,                  # class name
                        seg                          # segmentation
                    ))
                all_class_names = [det[2] for det in detections]

                print(f"all_class_names { all_class_names} - frame id {buffered_id}")

                detection_buffer.remove((buffered_id, objects))  # Hapus dari buffer setelah dipakai
                break  # Stop searching

    except socket.timeout:
        pass  # No detection received, continue

    return detections

def keypoints_to_list(keypoints):
    """Convert cv2.KeyPoint objects to a list of tuples for serialization.
    
    If keypoints is an empty list, return an empty list.
    """
    return [(float(kp.pt[0]), float(kp.pt[1]), float(kp.size), float(kp.angle), 
             float(kp.response), int(kp.octave), int(kp.class_id)) for kp in keypoints] if keypoints else []

def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on a frame.
    """
    kp1_pts = [kp.pt for kp in keypoints]
    output_frame = frame.copy()
    for (x, y) in kp1_pts:
        cv2.circle(output_frame, (int(x), int(y)), 5, color, -1)
    return output_frame

def process_and_stream_frames(video_port, result_port, camera_name, queue, video_path, shared_data, key_name):
    """Handles video processing and adds frames to queue instead of blocking WebSocket."""
  
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind(('0.0.0.0', video_port))
    sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_result.bind(("0.0.0.0", result_port))
    model = YOLO("best_13-04-2025.pt") 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_per_frame = 1.0 / fps if fps > 0 else 1.0 / 30  # fallback ke 30 FPS

    deepsort = DeepSort(max_age=15)
    tampak_depan_data1 = {}
    height_records1 = {}
    final_heights1 = {}
    passed_limits1 = {}

    tampak_depan_data2 = {}
    height_records2 = {}
    final_heights2 = {}
    passed_limits2 = {}
    id_mapping = {}
    id_merge = {}

    is_multicam = False

    while True:
        try:

            #frame, frame_id = delayed_receive_video(sock_video, camera_name)
            #time.sleep(0.02)
            #if frame is None:
            #    continue  
            #detections = extract_detections(sock_result, frame_id)

            #if frame is None or frame_id is None:
            #   continue  # Skip if invalid frame
            is_append = False
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (854,480))
            detections = []
            results = model(frame, verbose=False, conf=0.65)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                class_names = [model.names[i] for i in class_ids]  # Class names
                masks = result.masks  # Dapatkan masks langsung dari output
            
                for index, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax = box
                    width, height = xmax - xmin, ymax - ymin
            
                    if masks:
                        seg = masks.xy[index] if index < len(masks.xy) else None
                    else:
                        seg = None
            
                    detections.append((
                        [xmin, ymin, width, height],  # Bounding box
                        confidences[index],  # Confidence score
                        class_names[index],  # Class ID
                        seg                    
                    ))
            
            
            if detections:
                if camera_name == "Camera 1":
                    try:
                        tracking_results1, frame = process_tracks_and_extract_features(deepsort, sift, detections, frame)
                        tracked_objects1 = merge_track_ids(tracking_results1, id_merge)
                        shared_data[key_name][:] = []
                        for track in tracked_objects1:
                            if track["class_name"] == "Tampak Depan":
                                data = {
                                    "track_id": track["track_id"],
                                    "class_name": track["class_name"],
                                    "bounding_box": track["bounding box"],
                                    "keypoints": keypoints_to_list(track["kp"]),
                                    "descriptor": track["des"],
                                    "mask": track["mask"]
                                }
                                shared_data[key_name].append(data)  
                        if all(k in shared_data for k in ["tracked_objects1", "tracked_objects2"]):
                            good_matches, updated_tracked_objects1, id_mapping = match_features(shared_data["tracked_objects1"], shared_data["tracked_objects2"], frame, id_mapping)
                        reverse_mapping = {v: k for k, v in id_mapping.items()}
                        for result in tracked_objects1:
                            original_id = int(result.get("track_id"))
                            if original_id in reverse_mapping:
                                result["track_id"] = reverse_mapping[original_id]
                        reference_height1 = compute_reference_height(tracked_objects1, tampak_depan_data1)
                        final_heights1, height_records1, passed_limits1 = estimate_height(tracked_objects1, reference_height1, height_records1, passed_limits1, final_heights1, camera_name, id_mapping)
                        for old_id, new_id in reverse_mapping.items():
                            if old_id in passed_limits1:
                                # Tambahkan hanya jika new_id belum ada
                                if new_id not in passed_limits1:
                                    passed_limits1[new_id] = passed_limits1[old_id]
                                # Hapus old_id bagaimanapun juga
                                del passed_limits1[old_id]
                        updated_final_heights1 = {
                            reverse_mapping.get(old_id, old_id): value
                            for old_id, value in final_heights1.items()
                        }
                        final_heights1 = updated_final_heights1
                        for track in tracked_objects1:
                            if track['track_id'] in passed_limits1 and passed_limits1[track['track_id']]["left"] and track['track_id'] in final_heights1:
                                frame_captured1, truck_bbox1 = draw_mask_on_detected_tracks(frame, tracked_objects1, track['track_id'])
                                passed_limits1[track['track_id']]["left"] = False
                                save_violation_to_mongodb(frame_captured1, track['track_id'], final_heights1[track['track_id']], is_multicam, camera_name, truck_bbox1, frame)
                        frame = draw_tracking_info(frame, tracked_objects1)

                    
                    except Exception as e:
                        print(f"Error matching features: {e}")
                        traceback.print_exc()  # This prints the full traceback
                        continue
                    
                elif camera_name == "Camera 2":
                    try:
                        tracking_results2, frame = process_tracks_and_extract_features(deepsort, sift, detections, frame, is_cam1=False)
                        tracked_objects2 = merge_track_ids(tracking_results2, id_merge)
                        reference_height2 = compute_reference_height(tracked_objects2, tampak_depan_data2)
                        final_heights2, height_records2, passed_limits2 = estimate_height(tracked_objects2, reference_height2, height_records2, passed_limits2, final_heights2, camera_name, id_mapping)
                        shared_data[key_name][:] = []

                        for track in tracked_objects2:

                            if track['track_id'] in passed_limits2 and passed_limits2[track['track_id']]["left"] and track['track_id'] in final_heights2:
                                frame_captured2, truck_bbox2 = draw_mask_on_detected_tracks(frame, tracked_objects2, track['track_id'])
                                passed_limits2[track['track_id']]["left"] = False

                                save_violation_to_mongodb(frame_captured2, track['track_id'], final_heights2[track['track_id']], is_multicam, camera_name, truck_bbox2, frame)

                        frame = draw_tracking_info(frame, tracked_objects2, is_cam1=False)
                        for track in tracked_objects2:
                            if track["class_name"] == "Tampak Depan":
                                data = {
                                    "track_id": track["track_id"],
                                    "class_name": track["class_name"],
                                    "bounding_box": track["bounding box"],
                                    "keypoints": keypoints_to_list(track["kp"]),
                                    "descriptor": track["des"],
                                    "mask": track["mask"]
                                }
                                shared_data[key_name].append(data)  
                    
                    except Exception as e:
                        print(f"Error processing Camera 2: {e}")
                        continue
            if not queue.full() and frame is not None and frame.size > 0:
                queue.put((camera_name, frame))
            if frame is not None and frame.size > 0:
                cv2.imshow(camera_name, frame)
            elapsed_time = time.time() - start_time
            time_to_wait = delay_per_frame - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            traceback.print_exc()
            continue
def start_websocket_process(queue):
    """Start the WebSocket process and pass the queue."""
    asyncio.run(websocket_sender(queue))

def start_process(video_port, result_port, camera_name, queue, video_path, shared_data, manager):
    """Wrapper to run async function inside a multiprocessing process."""
    asyncio.run(process_and_stream_frames(video_port, result_port, camera_name, queue, video_path, shared_data, manager))

if __name__ == "__main__":

    video_path1 = "C:/Users/javie/Documents/Kuliah/Semester 8/Dataset/Pengujian/Merged/Pengujian Cam 1 Tahap 2_1.mp4"
    video_path2 = "C:/Users/javie/Documents/Kuliah/Semester 8/Dataset/Pengujian/Merged/Pengujian Cam 1 Tahap 2_2.mp4"

    try:
        manager = multiprocessing.Manager()
        shared_data = manager.dict()  # Shared dictionary for inter-process communication
        frame_queue = multiprocessing.Queue(maxsize=10)  # Shared queue
        shared_data["tracked_objects1"] = manager.list()
        shared_data["tracked_objects2"] = manager.list()

        process_websocket = multiprocessing.Process(target=start_websocket_process, args=(frame_queue,))
        process1 = multiprocessing.Process(target=start_process, args=(PORT_1, PORT_RESULT_1, "Camera 1", frame_queue, video_path1, shared_data, "tracked_objects1"))
        process2 = multiprocessing.Process(target=start_process, args=(PORT_2, PORT_RESULT_2, "Camera 2", frame_queue, video_path2, shared_data, "tracked_objects2"))

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

