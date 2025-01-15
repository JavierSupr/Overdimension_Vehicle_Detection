from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Dict, List
import time
from matching_and_tracking import initialize, match_feature


def apply_detections_and_bounding_box(frame: np.ndarray, results, class_names: List[str]) -> Tuple[np.ndarray, List[str, float]]:
    """
    Apply bounding boxes and labels to a frame
    """
    annotated_frame = frame.copy()
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_name = class_names[int(cls)]
        box = box.cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        detections.append(([xmin, ymin, width, height], conf, cls))
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

def detect_vehicles(video_paths: List[str]):
    """
    Detect vehicles from multiple video sources
    """
    model = YOLO('yolov8n-seg.pt')

    class_names = model.names
    colors = {cls_idx: tuple(np.random.randint(0, 256, 3).tolist()) for cls_idx in class_names}
    video_size = (640, 480)

    # Initialize video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]

    # Check if all video captures are opened successfully
    if not all(cap.isOpened() for cap in caps):
        print("Error: Couldn't open one or more video sources")
        return

    windows = [f'Vehicle Detection - Camera {i + 1}' for i in range(len(caps))]
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    is_paused = False

    descriptors = [None] * len(caps)
    tracks = [None] * len(caps)
    keypoints = [None] * len(caps)

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

        if not is_paused:
            for i, frame in enumerate(frames):
                results = model(frame, conf=0.25)[0]

                # Apply bounding boxes
                annotated_frame, detections = apply_detections_and_bounding_box(frame, results, class_names)

                descriptors[i], tracks[i], keypoints[i] = initialize(detections, frame)

                if i == 1 and descriptors[0] and descriptors[1]:
                    match_feature(descriptors[0], descriptors[1], tracks[0], tracks[1], keypoints[0], keypoints[1])


                # Apply masks if available
                annotated_frame = apply_mask_and_annotations(
                    annotated_frame, results, class_names, colors
                )

                # Calculate FPS
                fps = 1 / (time.time() - start_time)
                cv2.putText(
                    annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                # Display the result
                cv2.imshow(windows[i], annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_paused = not is_paused
            status = "PAUSED" if is_paused else "PLAYING"
            print(f"Video {status}")

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage with two video sources
    video_paths = [
        "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/20241019_114942.mp4",
        "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/WIN_20241019_11_49_42_Pro.mp4",
    ]
    detect_vehicles(video_paths)
