from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Dict, List
import time

def apply_mask_and_annotations(
    frame: np.ndarray,
    results,
    class_names: List[str],
    colors: Dict[str, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Apply masks and annotations to a frame based on detection results
    """
    annotated_frame = frame.copy()
    
    if results.masks is not None:
        for det_idx, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, 
                                                     results.boxes.cls, 
                                                     results.boxes.conf)):
            class_name = class_names[int(cls)]
            color = colors[class_name]

            # Apply mask if available
            if results.masks is not None and len(results.masks.data) > det_idx:
                mask = results.masks.data[det_idx].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

            # Draw bounding box and label
            box = box.cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f'{class_name} {conf:.2f}'
            cv2.putText(annotated_frame, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame

def detect_vehicles(video_paths: List[str]):
    """
    Detect vehicles from multiple video sources
    """
    model = YOLO('best.pt')
    
    vehicle_classes = [0, 1, 2] 
    class_names = ['tampak depan', 'tampak samping', 'truck']

    colors = {
        'tampak depan': (0, 255, 0),
        'tampak samping': (255, 0, 0),
        'truck': (0, 255, 255)
    }
    
    video_size = (640, 480)
    
    # Initialize video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # Check if all video captures are opened successfully
    if not all(cap.isOpened() for cap in caps):
        print("Error: Couldn't open one or more video sources")
        return

    windows = [f'Vehicle Detection - Camera {i+1}' for i in range(len(caps))]
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Initialize FPS variables for each camera
    fps_start_times = [0] * len(caps)
    fps_counters = [0] * len(caps)
    fps_values = [0] * len(caps)

    while True:
        frames = []
        # Read frames from all cameras
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame from one or more sources")
                return
            frame = cv2.resize(frame, video_size)
            frames.append(frame)
        
        # Process each frame
        for i, frame in enumerate(frames):
            results = model(frame, classes=vehicle_classes, conf=0.25)[0]
            
            # Apply masks and annotations using the separate function
            annotated_frame = apply_mask_and_annotations(frame, results, class_names, colors)
            
            # Display the result
            cv2.imshow(windows[i], annotated_frame)
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_paused = not is_paused
            status = "PAUSED" if is_paused else "PLAYING"
            print(f"Video {status}")

    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with two video sources
    video_paths = [
        "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Compressed/IPS_2024-01-27.14.30.40.3410.mp4",
        "C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/video test raw.mov",
    ]
    detect_vehicles(video_paths)