from ultralytics import YOLO
import cv2
import numpy as np

def mask_result(frame, results, class_names, colors):
    annotated_frame = frame.copy()  # Salin frame asli

    if len(results) > 0:
        detection = results[0]  # Ambil hasil untuk frame ini

        # Cek apakah mask tersedia
        if detection.masks is not None:  
            print(f"Hasil mask: {detection.masks}")
            for det_idx, (box, cls, conf) in enumerate(
                zip(detection.boxes.xyxy, detection.boxes.cls, detection.boxes.conf)
            ):
                # Nama kelas dan warna
                class_name = class_names[int(cls)]
                color = colors[class_name]

                # Proses mask
                if len(detection.masks.data) > det_idx:  # Cek apakah mask untuk deteksi ini ada
                    mask = detection.masks.data[det_idx].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)  # Skala mask menjadi 0-255

                    # Buat mask berwarna
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask > 0] = color

                    # Gabungkan mask berwarna dengan frame
                    annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

                # Bounding box
                box = box.cpu().numpy().astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Label
                label = f'{class_name} {conf:.2f}'
                cv2.putText(annotated_frame, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_frame
class_names = ['tampak depan', 'tampak samping', 'truck']

# Contoh warna untuk setiap kelas
colors = {
    'tampak depan': (255, 0, 0),  # Merah
    'tampak samping': (0, 255, 0),     # Hijau
    'truck': (0, 0, 255)    # Biru
}

model = YOLO('yolov8n-seg.pt')

cap1 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/MVI_0795.mp4")
cap2 = cv2.VideoCapture("C:/Users/javie/Documents/Kuliah/Semester 7/Penulisan Ilmiah/Dokumentasi/Video TA/Video2/MVI_0795.mp4")

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video/camera")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    if not ret1 or not ret2:
        print("End of video or camera stream")
        break

    result1 = model(frame1)
    result2 = model(frame2)

    combined_mask1 = mask_result(frame1, result1,class_names, colors)
    combined_mask2 = mask_result(frame2, result2,class_names, colors)

    cv2.imshow('YOLOv8 Segmentation', combined_mask1)
    #cv2.imshow('YOLOv8 ', masked_frame1)

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()