from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Dict, List
import time
from matching_and_tracking import initialize, match_feature
import websockets
from stream_handler import video_stream, start_stream_server
import asyncio



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

        detections.append(([xmin, ymin, width, height], conf, class_name))
        color = (0, 255, 0)  # You can dynamically assign colors per class
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

            # Apply mask if available
            if len(results.masks.data) > det_idx:
                mask = results.masks.data[det_idx].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[mask > 0] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

    return annotated_frame

async def handle_websocket(websocket):
    global shared_frames
    await video_stream(websocket, shared_frames[0], shared_frames[1])

async def start_websocket_server():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # run forever

async def detect_vehicles(video_paths: List[str]):
    """
    Detect vehicles from multiple video sources and stream via WebSocket
    """
    global shared_frames
    shared_frames = [None, None]
    
    model = YOLO('yolov8n-seg.pt')
    class_names = model.names
    colors = {cls_idx: tuple(np.random.randint(0, 256, 3).tolist()) for cls_idx in class_names}
    video_size = (640, 480)

    # Initialize video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Couldn't open one or more video sources")
        return

    try:
        while True:
            start_time = time.time()

            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Couldn't read frame from one or more sources")
                    return
                frame = cv2.resize(frame, video_size)
                frames.append(frame)

            for i, frame in enumerate(frames):
                results = model(frame, conf=0.25)[0]
                annotated_frame, detections = apply_detections_and_bounding_box(frame, results, class_names)
                annotated_frame = apply_mask_and_annotations(annotated_frame, results, colors)

                fps = 1 / (time.time() - start_time)
                cv2.putText(
                    annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                shared_frames[i] = annotated_frame

            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    except Exception as e:
        print(f"Error in detection loop: {e}")
    finally:
        for cap in caps:
            cap.release()

async def main():
    video_paths = [
        "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20241019_114942.mp4",
        "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/WIN_20241019_11_49_42_Pro.mp4",
    ]
    
    # Create tasks for both the detection and websocket server
    detection_task = asyncio.create_task(detect_vehicles(video_paths))
    websocket_task = asyncio.create_task(start_websocket_server())
    
    # Wait for both tasks
    await asyncio.gather(detection_task, websocket_task)

if __name__ == "__main__":
    asyncio.run(main())