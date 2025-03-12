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
from id_merging import merge_track_ids
from database import save_violation_to_mongodb, check_id_exists, draw_mask_on_detected_tracks
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Tuple, Dict, List
import base64
from ultralytics import YOLO
from dimension_estimation import  compute_reference_height, estimate_height, get_final_estimated_heights
from collections import deque
import traceback
PORT_1 = 5010  # Port for first video stream
PORT_2 = 5011  # Port for second video stream
PORT_RESULT_1 = 5012
PORT_RESULT_2 = 5013
BUFFER_SIZE = 65536  # Max UDP packet size
UDP_IP = "0.0.0.0"

WEBSOCKET_URL = "ws://localhost:8765"
WEBSOCKET_PORT = 8765

detections_buffer = {}

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
# Buffer to store last N frames' detections
DETECTION_BUFFER_SIZE = 10
detection_buffer = deque(maxlen=DETECTION_BUFFER_SIZE)

def extract_detections(sock_result, frame_id):
    """Receives detections via UDP and applies them to the matching video frame using frame_id."""
    global detection_buffer
    sock_result.settimeout(0.05)  # Prevent blocking
    detections = []
    masks = []


    try:
        data, _ = sock_result.recvfrom(65535)
        message = json.loads(data.decode())

        received_frame_id = message.get("frame_id", None)  # Extract frame ID

        if received_frame_id is not None:
            # Store the detection in buffer
            detection_buffer.append((received_frame_id, message["objects"]))

        # Search buffer for matching frame_id
        for buffered_id, objects in list(detection_buffer):
            if buffered_id == frame_id:
                for obj in objects:
                    class_name = obj["label"]
                    conf = float(obj["conf"])
                    bbox = obj["bbox"]
                    mask = obj.get("seg", None)
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf, class_name))
                    masks.append((mask))
                break  # Stop searching once we find a match

    except socket.timeout:
        pass  # No detection received, continue

    return detections, masks

def keypoints_to_list(keypoints):
    """Convert cv2.KeyPoint objects to a list of tuples for serialization.
    
    If keypoints is an empty list, return an empty list.
    """
    return [(float(kp.pt[0]), float(kp.pt[1]), float(kp.size), float(kp.angle), 
             float(kp.response), int(kp.octave), int(kp.class_id)) for kp in keypoints] if keypoints else []


def list_to_keypoints(kp_list):
    """Convert a list of tuples back into cv2.KeyPoint objects and print them."""
    keypoints = []
    for kp in kp_list:
        if len(kp) == 7:  # Ensure tuple has 7 elements
            x, y, size, angle, response, octave, class_id = kp  # Unpack tuple
            keypoint = cv2.KeyPoint(
                x=float(x), y=float(y), size=float(size), 
                angle=float(angle), response=float(response), 
                octave=int(octave), class_id=int(class_id)
            )
            keypoints.append(keypoint)
        else:
            print(f"Invalid keypoint format {kp} (Expected 7 elements)")
    
    # Print keypoints
    #for i, kp in enumerate(keypoints):
        #print(f"Keypoint {i}: (x={kp.pt[0]}, y={kp.pt[1]}, size={kp.size}, angle={kp.angle}, response={kp.response}, octave={kp.octave}, class_id={kp.class_id})")
    
    return keypoints
CLASS_COLORS = {
    "Truk": (0, 0, 255),         # Red
    "Tampak Depan": (255, 0, 0), # Blue
    "Tampak Samping": (0, 255, 0) # Green
}

def display_video_with_masks(frame, updated_tracks):
        """
        Display the video with overlayed segmentation masks from the updated_tracks list.

        Args:
            video_path (str): Path to the input video.
            updated_tracks (list): List of tuples (track, class_name, track_id, mask).
        """

        for track, class_name, track_id, mask in updated_tracks:
            color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default white if class not found
            
            if mask is not None:
                mask_image = np.zeros_like(frame[:, :, 0], dtype=np.uint8)  # Create blank mask

                # Ensure mask is in the correct format for OpenCV
                if isinstance(mask, list) and isinstance(mask[0], (list, np.ndarray)):  
                    # If mask is a list of multiple polygon segments
                    for segment in mask:
                        segment_np = np.array(segment, dtype=np.int32)
                        if len(segment_np.shape) == 2 and segment_np.shape[1] == 2:  # Ensure valid shape
                            cv2.fillPoly(mask_image, [segment_np], 255)
                elif isinstance(mask, np.ndarray) and len(mask.shape) == 2:
                    # If mask is already a binary image
                    mask_image = mask.astype(np.uint8) * 255

                # Overlay mask with transparency using the class color
                mask_colored = np.zeros_like(frame, dtype=np.uint8)
                mask_colored[mask_image > 0] = color  # Apply the class-specific color
                frame = cv2.addWeighted(mask_colored, 0.5, frame, 0.5, 0)

            # Draw track information
            bbox = track.to_tlbr()  # Get bounding box [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box in class color
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)       #if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        
        return frame

def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on a frame.
    """
    #print("masukk")
    kp1_pts = [kp.pt for kp in keypoints]
    #print(f"kp1_idx {kp1_pts}")
    output_frame = frame.copy()
    for (x, y) in kp1_pts:
        cv2.circle(output_frame, (int(x), int(y)), 5, color, -1)
    return output_frame

def process_and_stream_frames(video_port, result_port, camera_name, queue, video_path, shared_data):
    """Handles video processing and adds frames to queue instead of blocking WebSocket."""
  
    #sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock_video.bind(("0.0.0.0", video_port))
    #sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock_result.bind(("0.0.0.0", result_port))
    model = YOLO("best.pt") 
    cap = cv2.VideoCapture(video_path)
    deepsort = DeepSort(max_age=10)
    tampak_depan_data1 = {}
    height_records1 = {}
    final_heights1 = {}
    passed_limits1 = {}

    tampak_depan_data2 = {}
    height_records2 = {}
    final_heights2 = {}
    passed_limits2 = {}
    while True:
        try:
            #frame, frame_id = receive_video(sock_video, camera_name)
            #if frame is None or frame_id is None:
            #    continue  # Skip if invalid frame
            # Receive inference results
            #print("-2")
            #detections, masks = extract_detections(sock_result, frame_id)
            # Use 'yolov8n.pt' for a small model
            #print("-1")
            estimated_height = 2
            is_append = False
            ret, frame = cap.read()
            #if not ret:
            #    break
            frame = cv2.resize(frame, (640,480))
            detections = []
            #masks = []
            results = model(frame, verbose=False, conf=0.8)[0]
            detections2 = []  # List untuk menyimpan masks
#
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
                #print("0000")
                if camera_name == "Camera 1":
                    try:
                        tracking_results1, frame = process_tracks_and_extract_features(detections, frame)
                        tracked_objects1 = merge_track_ids(tracking_results1)
                        reference_height1 = compute_reference_height(tracked_objects1, tampak_depan_data1)
                        final_heights1, height_records1, passed_limits1 = estimate_height(tracked_objects1, reference_height1, height_records1, passed_limits1, final_heights1)
                        for track in tracked_objects1:
                            #print(f"passed_limits {passed_limits}")
                            if track['kp']:
                                frame = draw_keypoints(frame, track['kp'] )

                            if track['track_id'] in passed_limits1 and passed_limits1[track['track_id']]["left"] and track['track_id'] in final_heights1:
                                frame_captured = draw_mask_on_detected_tracks(frame, tracked_objects1)
                                passed_limits1[track['track_id']]["left"] = False

                                save_violation_to_mongodb(frame_captured, track['track_id'], final_heights1[track['track_id']], "REF_12345")


                        #print(object_tracking_status)
                        #final_heights = get_final_estimated_heights(height_records)
                        #print("4")
                        #frame = draw_tracking_info(frame, tracked_objects1, estimated_height)
                        #print("5")
                        #print(f"keypoints1 {keypoints1}")
                        # Store data in shared dictionary
                        for track in tracked_objects1:
                            if track["class_name"] == "Tampak Depan":
                                shared_data[f"tracked_objects1"] = {
                                    "track_id": track["track_id"],
                                    "class_name": track["class_name"],
                                    "bounding_box": track["bounding box"],
                                    "keypoints": keypoints_to_list(track["kp"]),
                                    "descriptor": track["des"],
                                    "mask": track["mask"]
                                }
                                #print(shared_data[f"tracked_objects1"])



                        if all(k in shared_data for k in ["tracked_objects1", "tracked_objects2"]):
                                #print("masuk -2")                                   
                                    #print("masuk 0")
                                    good_matches, updated_tracked_objects1, id_mapping = match_features(shared_data["tracked_objects1"], shared_data["tracked_objects2"], frame)
                                    #print("6")
                                    #print(f"id mapping {id_mapping}")
                                    #for track_id in id_mapping.values():  # or id_mapping.keys() depending on your structure
                                       #if not check_id_exists(track_id):
                                       #print(f"tracked object {tracked_objects1}")
                                      #save_violation_to_mongodb(track_id, frame, tracked_objects1)
                                    reverse_mapping = {v: k for k, v in id_mapping.items()}
                                    #print(tracked_objects1)

                                    for entry in tracked_objects1:
                                        if entry["track_id"] in reverse_mapping:
                                            entry["track_id"] = reverse_mapping[entry["track_id"]]
                                   
                        frame = draw_tracking_info(frame, tracked_objects1)
                                    #print("7")
                    except Exception as e:
                        print(f"Error matching features: {e}")
                        traceback.print_exc()  # This prints the full traceback
                        continue
                    
                elif camera_name == "Camera 2":
                    try:
                        tracking_results2, frame = process_tracks_and_extract_features(detections, frame, is_cam1=False)
                        tracked_objects2 = merge_track_ids(tracking_results2)
                        reference_height2 = compute_reference_height(tracked_objects2, tampak_depan_data2)
                        final_heights2, height_records2, passed_limits2 = estimate_height(tracked_objects2, reference_height2, height_records2, passed_limits2, final_heights2)
                        for track in tracked_objects2:

                            if track['track_id'] in passed_limits2 and passed_limits2[track['track_id']]["left"] and track['track_id'] in final_heights2:
                                frame_captured = draw_mask_on_detected_tracks(frame, tracked_objects2)
                                passed_limits2[track['track_id']]["left"] = False

                                save_violation_to_mongodb(frame_captured, track['track_id'], final_heights2[track['track_id']], "REF_12345")

                        frame = draw_tracking_info(frame, tracking_results2, is_cam1=False)

                        for track in tracking_results2:
                            if track["class_name"] == "Tampak Depan":
                                shared_data[f"tracked_objects2"] = {
                                    "track_id": track["track_id"],
                                    "class_name": track["class_name"],
                                    "bounding_box": track["bounding box"],
                                    "keypoints": keypoints_to_list(track["kp"]),
                                    "descriptor": track["des"],
                                    "mask": track["mask"]
                                }
                    
                    except Exception as e:
                        print(f"Error processing Camera 2: {e}")
                        continue
            if not queue.full():
                queue.put((camera_name, frame))

            cv2.imshow(camera_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            continue
def start_websocket_process(queue):
    """Start the WebSocket process and pass the queue."""
    asyncio.run(websocket_sender(queue))

def start_process(video_port, result_port, camera_name, queue, video_path, shared_data):
    """Wrapper to run async function inside a multiprocessing process."""
    asyncio.run(process_and_stream_frames(video_port, result_port, camera_name, queue, video_path, shared_data))

if __name__ == "__main__":
    video_path1 = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/MVI_0800.MOV"
    video_path2 = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/MVI_0800.MOV"
    
    #video_path1 = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20241019_114942.mp4"
    #video_path2 = "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/WIN_20241019_11_49_42_Pro.mp4"

    try:
        manager = multiprocessing.Manager()
        shared_data = manager.dict()  # Shared dictionary for inter-process communication
        frame_queue = multiprocessing.Queue(maxsize=10)  # Shared queue

        process_websocket = multiprocessing.Process(target=start_websocket_process, args=(frame_queue,))
        process1 = multiprocessing.Process(target=start_process, args=(PORT_1, PORT_RESULT_1, "Camera 1", frame_queue, video_path1, shared_data))
        process2 = multiprocessing.Process(target=start_process, args=(PORT_2, PORT_RESULT_2, "Camera 2", frame_queue, video_path2, shared_data))

        #process_websocket.start()
        process1.start()
        process2.start()

        process1.join()
        process2.join()
        #process_websocket.join()

    except KeyboardInterrupt:
        print("\n[INFO] Program stopped by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] All processes closed. Exiting...")

