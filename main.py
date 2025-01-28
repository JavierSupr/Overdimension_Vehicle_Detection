from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Dict, List
import websockets
import asyncio
from deep_sort_realtime.deepsort_tracker import DeepSort
from matching_and_tracking import process_tracks_and_extract_features, match_features, draw_tracking_info
from stream_handler import send_frame_via_websocket
from id_merging import merge_track_ids
from database import save_violation_to_mongodb, check_id_exists

deepsort1 = DeepSort(max_age=5)
deepsort2 = DeepSort(max_age=5)
sift = cv2.SIFT_create()
iou_threshold = 0.2
processed_tracks = set()  
estimated_height = 1.8

def apply_detections_and_bounding_box(
    frame: np.ndarray, 
    results, 
    class_names: List[str]
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    Apply bounding boxes and labels to a frame.
    """
    annotated_frame = frame.copy()
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_name = class_names[int(cls)]
        box = box.cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        conf = float(conf)
        cls = int(cls)

        detections.append(([xmin, ymin, width, height], conf, cls))
        #print(f"detections{detections}")
        #color = (0, 255, 0)
        #cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        #label = f'{class_name} {conf:.2f}'
        #cv2.putText(
        #    annotated_frame, label, (box[0], box[1] - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        #)

    return  detections

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

async def process_and_stream_frames(websocket, model, cap1, cap2):
    """
    Combined processing and streaming of frames to ensure fresh frames are always sent
    """
    try:
        class_names = model.names
        colors = {cls_idx: tuple(np.random.randint(0, 256, 3).tolist()) 
                 for cls_idx in class_names}
        while True:
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                # Reset videos if they end
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
            # Resize frames
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))
            # Process frames with YOLO
            results1 = model.predict(frame1, task='segment', conf=0.25)[0]
            results2 = model.predict(frame2, task='segment', conf=0.25)[0]
            #print(f"result {results1}")
            # Apply detections and masks
            detections1 = apply_detections_and_bounding_box(frame1, results1, class_names)
            detections2 = apply_detections_and_bounding_box(frame2, results2, class_names)
    
            tracks1, keypoints1, descriptors1 = process_tracks_and_extract_features(deepsort1, detections1, frame1)
            tracks2, keypoints2, descriptors2 = process_tracks_and_extract_features(deepsort2, detections2, frame2)
    
            good_matches = match_features(descriptors1, descriptors2, tracks1, tracks2, keypoints1, keypoints2)
            
            frame1_tracked = draw_tracking_info(frame1.copy(), tracks1, is_cam1=True)
            frame2_tracked = draw_tracking_info(frame2.copy(), tracks2, is_cam1=False)
            
            id_group = merge_track_ids(tracks1, iou_threshold)
            for track in tracks1:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                # Find the root (merged) ID for this track
                merged_track_id = find_root(id_group, track.track_id)

                bbox = track.to_ltwh()
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                if not check_id_exists(merged_track_id):
                    save_violation_to_mongodb(merged_track_id, estimated_height, frame1_tracked)
                    processed_tracks.add(merged_track_id)
                # Optionally: Draw annotations on the frame
                cv2.rectangle(frame1_tracked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame1_tracked, f"ID: {merged_track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the annotated frame (optional)
                cv2.imshow('Annotated Frame', frame1_tracked)
            #frame1 = apply_mask_and_annotations(frame1, results1, colors)
            #frame2 = apply_mask_and_annotations(frame2, results2, colors)
            if good_matches:
                frame_matches = cv2.drawMatches(frame1_tracked, keypoints1, frame2_tracked, keypoints2, good_matches, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                #frame_matches = cv2.hconcat([frame1_tracked, frame2_tracked])
            else:
                frame_matches = cv2.hconcat([frame1_tracked, frame2_tracked])
            
            cv2.imshow('YOLOv8 Vehicle Tracking with DeepSORT and SIFT Matching', frame_matches)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await send_frame_via_websocket(websocket, frame1_tracked, "Camera 1")
            await send_frame_via_websocket(websocket, frame2_tracked, "Camera 2")
    
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket Connection closed with error: {e}")
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket closed normally")
    except Exception as e:
        print(f"Unexpected error2: {e}")
    finally:
        print("WebSocket closed, releasing resources.")

async def handle_connection(websocket):
    # Initialize YOLO model
    model = YOLO('yolov8n-seg.pt')
    
    # Initialize video captures
    #cap1 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4")
    #cap2 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4")
    cap1 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20241019_114942.mp4")
    cap2 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/WIN_20241019_11_49_42_Pro.mp4")

    try:
        await process_and_stream_frames(websocket, model, cap1, cap2)
    except Exception as e:
        print(f"Error in handle_connection: {e}")
    finally:
        cap1.release()
        cap2.release()

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Unexpected error: {e}")