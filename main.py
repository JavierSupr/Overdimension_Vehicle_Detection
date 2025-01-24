from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Dict, List
import websockets
import asyncio
import base64
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
from matching_and_tracking import initialize, match_feature
deepsort1 = DeepSort(max_age=5)
deepsort2 = DeepSort(max_age=5)

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
        print(f"detections{detections}")
        color = (0, 255, 0)
        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f'{class_name} {conf:.2f}'
        cv2.putText(
            annotated_frame, label, (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return annotated_frame, detections

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

async def process_and_stream_frames(websocket, model, cap1, cap2):
    """
    Combined processing and streaming of frames to ensure fresh frames are always sent
    """
    #try:
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
        print(f"result {results1}")
        # Apply detections and masks
        frame1, detections1 = apply_detections_and_bounding_box(frame1, results1, class_names)
        frame2, detections2 = apply_detections_and_bounding_box(frame2, results2, class_names)
    
        initialize(detections1, frame1)
        #descriptor2, track2, keypoint2 = initialize(detections=detections2, frame=frame2)
        #
        #match_feature(descriptor1=descriptor1, descriptor2=descriptor2, tracks1=track1, tracks2=track2, keypoints1=keypoint1, keypoints2=keypoint2)
        
        frame1 = apply_mask_and_annotations(frame1, results1, colors)
        frame2 = apply_mask_and_annotations(frame2, results2, colors)
        
        
        frame_matches = cv2.hconcat([frame1, frame2])
        cv2.imshow('YOLOv8 Vehicle Tracking with DeepSORT and SIFT Matching', frame_matches)
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            # Encode and send frames
            #_, buffer1 = cv2.imencode('.jpg', frame1)
            #frame1_based64 = base64.b64encode(buffer1).decode('utf-8')
            #message1 = json.dumps({"type": "frame", "camera": "Camera 1", "data": frame1_based64})
            #await websocket.send(message1)

            #_, buffer2 = cv2.imencode('.jpg', frame2)
            #frame2_based64 = base64.b64encode(buffer2).decode('utf-8')
            #message2 = json.dumps({"type": "frame", "camera": "Camera 2", "data": frame2_based64})
            #await websocket.send(message2)

            #await asyncio.sleep(0.03)  # Control frame rate

   # except websockets.exceptions.ConnectionClosedError as e:
   #     print(f"WebSocket Connection closed with error: {e}")
   # except websockets.exceptions.ConnectionClosedOK:
   #     print("WebSocket closed normally")
   # except Exception as e:
   #     print(f"Unexpected error2: {e}")
   # finally:
   #     print("WebSocket closed, releasing resources.")

async def handle_connection(websocket):
    # Initialize YOLO model
    model = YOLO('yolov8n-seg.pt')
    
    # Initialize video captures
    cap1 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4")
    cap2 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20250111_121008.mp4")

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