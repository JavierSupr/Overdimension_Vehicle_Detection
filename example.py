import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (segmentation)
model_path = "best.pt"  # Ganti dengan model Anda
model = YOLO(model_path)

deep_sort_tracker = DeepSort(max_age=30)  # Inisialisasi DeepSORT

# Load video
cap = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/MVI_0800.MOV")  # Ganti dengan sumber video Anda

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Warna untuk bounding box setiap ID
np.random.seed(42)
frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Jalankan YOLOv8 untuk deteksi
    results = model(frame)
    detections = []
    tracking_results = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box koordinat (x1, y1, x2, y2)
        confs = result.boxes.conf.cpu().numpy()  # Confidence score
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class ID
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1
            if conf > 0.3:  # Confidence threshold
                detections.append(([x1, y1, width, height], conf, class_id))
                
                # Tampilkan mask jika tersedia dan valid
                mask = masks[i] if len(masks) > i else None
                if mask is not None:
                    mask = cv2.resize(mask, (frame_width, frame_height))  # Resize ke ukuran frame
                    mask = (mask > 0.5).astype(np.uint8)  # Binarisasi mask (0 atau 1)
                
                tracking_results.append({
                    "track_id": None,  # Akan diupdate setelah tracking
                    "class_name": model.names[class_id],
                    "bounding_box": [x1, y1, x2, y2],
                    "mask": mask
                })
    
    # Jalankan tracking dengan DeepSORT
    tracks = deep_sort_tracker.update_tracks(detections, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Bounding box (left, top, right, bottom)
        class_id = track.det_class  # Class ID
        x1, y1, x2, y2 = map(int, ltrb)
        color = (0, 255, 0)

        for item in tracking_results:
            iou = calculate_iou(item["bounding_box"], [x1, y1, x2, y2])
            if iou > 0.8:
                item["track_id"] = track_id
                print(f"Tracking results: {tracking_results}")
                
                if item["mask"] is not None:
                    mask = (item["mask"] * 255).astype(np.uint8)
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[:, :, 0] = mask * color[0]
                    color_mask[:, :, 1] = mask * color[1]
                    color_mask[:, :, 2] = mask * color[2]
                    frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)
                    print("masukkk2")
                break
        
        # Gambar bounding box dan ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}: {model.names[class_id]}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Tampilkan hasil
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
