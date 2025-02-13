from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Dict, List
import websockets
import asyncio
from deep_sort_realtime.deepsort_tracker import DeepSort
import socket
import json
import struct
from matching_and_tracking import process_tracks_and_extract_features, match_features, draw_tracking_info
from stream_handler import send_frame_via_websocket
from id_merging import merge_track_ids
from database import save_violation_to_mongodb, check_id_exists
#from video_receiver import receive_video
import threading
import queue
deepsort1 = DeepSort(max_age=5)
deepsort2 = DeepSort(max_age=5)
sift = cv2.SIFT_create()
iou_threshold = 0.2
processed_tracks = set()  
estimated_height = 1.8

UDP_IP = "0.0.0.0"

# UDP Configuration
PORT_1 = 5005 
PORT_2 = 5006 
PORT_RESULT = 5007
BUFFER_SIZE = 65536  

# Create sockets for two streams
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.bind(("0.0.0.0", PORT_1))

sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.bind(("0.0.0.0", PORT_2))

sock_result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_result.bind((UDP_IP, PORT_RESULT))

# Flag to control the threads
#running = True
PORT_1 = 5005
PORT_2 = 5006
BUFFER_SIZE = 65536
running = True

# Shared frame queues
frame_queues = {
    "Camera 1": queue.Queue(maxsize=5),
    "Camera 2": queue.Queue(maxsize=5),
}

def receive_video(sock, camera_name):
    #"""Receives video frames over UDP in a separate thread."""
    #running =True
    #while running:
        try:
            print("yes")
            # Receive number of chunks
            data, _ = sock.recvfrom(BUFFER_SIZE)
            if len(data) == 1:
                chunks_count = struct.unpack("B", data)[0]
            #else:
                #continue  # Skip invalid packets

            # Receive the chunks
            chunks = []
            for _ in range(chunks_count):
                chunk, _ = sock.recvfrom(BUFFER_SIZE)
                chunks.append(chunk)

            # Combine chunks and decode frame
            frame_data = b"".join(chunks)
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is not None:
                if not frame_queues[camera_name].full():
                    frame_queues[camera_name].put(frame)

        except Exception as e:
            print(f"[ERROR] Receiving video from {camera_name} failed: {e}")
            running = False
            #break


#def apply_detections_and_bounding_box(
#    frame: np.ndarray, 
#    results, 
#    class_names: List[str]
#) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
#    """
#    Apply bounding boxes and labels to a frame.
#    """
#    annotated_frame = frame.copy()
#    detections = []
#    
#    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
#        class_name = class_names[int(cls)]
#        box = box.cpu().numpy().astype(int)
#        xmin, ymin, xmax, ymax = box
#        width = xmax - xmin
#        height = ymax - ymin
#        conf = float(conf)
#        cls = int(c#ls)

#        detections.append(([xmin, ymin, width, height], conf, cls))
#        #print(f"detections{detections}")
#        #color = (0, 255, 0)
#        #cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
#        #label = f'{class_name} {conf:.2f}'
#        #cv2.putText(
#        #    annotated_frame, label, (box[0], box[1] - 10),
#        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
#       # #)

#    return  detections

def apply_detections_and_bounding_box():
        """Fetches frames, applies bounding boxes, and displays them."""
    #running = True
    #global running
    #while running:
        print("yes1")
        try:
            if not any(not q.empty() for q in frame_queues.values()):
                print("yes4")
                #running = False
                #continue  # Lewati iterasi jika semua frame_queues kosong
                
            for camera_name in frame_queues.keys():
                if frame_queues[camera_name].empty():
                    running = False
                    continue  # Lewati iterasi ini jika queue kamera tertentu kosong
                    
                frame = frame_queues[camera_name].get()
                print("yes2")

                # Receive detection results
                try:
                    print("yes3")
                    data, _ = sock_result.recvfrom(65535)  # Buffer size
                    message = json.loads(data.decode())

                    for obj in message["objects"]:
                        class_name = obj["label"]
                        conf = float(obj["conf"])
                        bbox = obj["bbox"]

                        xmin, ymin, xmax, ymax = map(int, bbox)
                        width, height = xmax - xmin, ymax - ymin

                        # Draw bounding box
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

                    # Display the frame
                    cv2.imshow(camera_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        running = False
                        break

                except Exception as e:
                    print(f"[ERROR] Receiving detection results failed: {e}")
                    running = False
                    break

        except Exception as e:
            print(f"[ERROR] Receiving detection results failed: {e}")
            running = False
            #break

def apply_mask_and_annotations(
    annotated_frame: np.ndarray,
    results,
    colors: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Apply masks and annotations to a frame based on detection results.
    """
    if results.masks is not None:
        for det_idx, cls in enumerate(results.boxes.cls):
            color = colors[int(cls)]
            if len(results.masks.data) > det_idx:
                mask = results.masks.data[det_idx].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[mask > 0] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

    return annotated_frame

def find_root(id_group, track_id):
        """Find the root ID for a given track ID."""
        if track_id not in id_group:
            return track_id
        while id_group[track_id] != track_id:
            track_id = id_group[track_id]
        return track_id


async def process_and_stream_frames(websocket):
    """
    Combined processing and streaming of frames to ensure fresh frames are always sent
    """
    try:

        class_names = model.names
        print(class_names)
        colors = {cls_idx: tuple(np.random.randint(0, 256, 3).tolist()) 
                 for cls_idx in class_names}
        print("masuk siniii2")
        video_task1 = asyncio.create_task(receive_video(sock1).__anext__())
        video_task2 = asyncio.create_task(receive_video(sock2).__anext__())
        inference_task = asyncio.create_task(apply_detections_and_bounding_box())
        while running:
            print("masuk")
            frame1, frame2 = await asyncio.gather(receive_video(sock1).__anext__(), receive_video(sock2).__anext__())
            detections = await asyncio.gather(apply_detections_and_bounding_box().__anext__())
        
#print(f"Frame1 shape: {frame1.shape}, Frame2 shape: {frame2.shape}")
            frame1 = cv2.resize(frame1, (256, 256))
            frame2 = cv2.resize(frame2, (256, 256))
             #Process frames with YOLO
            #results1 = model.predict(frame1, task='segment', conf=0.25)[0]
            #results2 = model.predict(frame2, task='segment', conf=0.25)[0]
             #Apply detections and masks
            apply_detections_and_bounding_box()
            detections2 = apply_detections_and_bounding_box(frame2)
    
            tracks1, keypoints1, descriptors1 = process_tracks_and_extract_features(deepsort1, detections1, frame1)
            tracks2, keypoints2, descriptors2 = process_tracks_and_extract_features(deepsort2, detections2, frame2)
    #
            good_matches = match_features(descriptors1, descriptors2, tracks1, tracks2, keypoints1, keypoints2)
            
            frame1_tracked = draw_tracking_info(frame1.copy(), tracks1, is_cam1=True)
            frame2_tracked = draw_tracking_info(frame2.copy(), tracks2, is_cam1=False)
            
            id_group = merge_track_ids(tracks1, iou_threshold)
            for track in tracks1:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                # Find the root (merged) ID for this track
                merged_track_id = find_root(id_group, track.track_id)
                #print(merged_track_id)

                bbox = track.to_ltwh()
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                if not check_id_exists(merged_track_id):
                    save_violation_to_mongodb(merged_track_id, estimated_height, frame1_tracked)
                    processed_tracks.add(merged_track_id)
                # Optionally: Draw annotations on the frame
                #cv2.rectangle(frame1_tracked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame1_tracked, f"ID: {merged_track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

               # Display the annotated frame (optional)
               #cv2.imshow('Annotated Frame', frame1_tracked)
            frame1 = apply_mask_and_annotations(frame1, results1, colors)
            frame2 = apply_mask_and_annotations(frame2, results2, colors)
            if good_matches:
                frame_matches = cv2.drawMatches(frame1_tracked, keypoints1, frame2_tracked, keypoints2, good_matches, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                frame_matches = cv2.hconcat([frame1_tracked, frame2_tracked])
            else:
               frame_matches = cv2.hconcat([frame1_tracked, frame2_tracked])
        
            cv2.imshow('YOLOv8 Vehicle Tracking with DeepSORT and SIFT Matching', frame_matches)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await send_frame_via_websocket(websocket, frame1, "Camera 1")
            await send_frame_via_websocket(websocket, frame2, "Camera 2")
    
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket Connection closed with error: {e}")
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket closed normally")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("WebSocket closed, releasing resources.")

async def handle_connection(websocket):
    # Initialize YOLO model
    #model = YOLO('yolov8n-seg.pt')
    #model = YOLO('best.pt')
    #cap1 = cv2.VideoCapture(receive_video(sock=sock1, stream_name= "Camera 1"))
    #cap2 = receive_video(sock=sock2, stream_name= "Camera 2")
    # Initialize video captures
    #cap1 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4")
    #cap2 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4")
    #cap1 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20241019_114942.mp4")
    #cap2 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/WIN_20241019_11_49_42_Pro.mp4")

    try:
        await process_and_stream_frames(websocket)
    except Exception as e:
        print(f"Error in handle_connection: {e}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    #try:
        #asyncio.run(main())
        print("tes")
        while True:
            thread3 = threading.Thread(target=apply_detections_and_bounding_box)
            print("yes5")
            thread1 = threading.Thread(target=receive_video, args=(sock1, "Camera 1"))
            thread2 = threading.Thread(target=receive_video, args=(sock2, "Camera 2"))

            thread3.start()
            thread1.start()
            thread2.start()

            
        #receive_video(sock=sock1, stream_name= "Camera 1")
        #print("tes1")
        #asyncio.run(main())
        
   # except KeyboardInterrupt:
   #     print("\nShutting down server...")
   # except Exception as e:
   #     print(f"Unexpected error: {e}")
   # finally:
   #     print("js")
   #     cv2.destroyAllWindows()
   #     sock1.close()