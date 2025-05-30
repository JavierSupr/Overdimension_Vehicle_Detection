# 🚚 Overdimension Vehicle Detection with YOLOv8n-Seg and Multi-Camera Setup

<p align="center">
  <img src="assets/mask_segmentation.png" alt="YOLOv8 Segmentation" width="35%" />
  <img src="assets/Multi_Kamera_3.gif" alt="Multi Camera" width="55%" />
</p>
This project focuses on real-time detection of overdimension vehicles using YOLOv8n segmentation model and a multi-camera system. It is designed to identify and measure height using segmentation masks and match vehicle views across different camera angles for accurate monitoring and verification.

---

## 📌 Features

- ✅ **Vehicle Segmentation** using YOLOv8n-seg (Ultralytics)
- 📷 **Multi-Camera Integration** (left and right front views)
- 📏 **Dimension Estimation** (height from mask based on reference height)
- 🔄 **Object Re-identification** across cameras (using DeepSORT or SIFT + BFMatcher, KNN, Lowe's Ratio Test)
- 🌐 **Real-Time Video Streaming**  Supports real-time video stream processing using WebSocket protocol.
- 💾 **Data Logging** (plate number, time, image evidence, height, etc)
- 🌐 **Web Dashboard** to monitor detections and view historical data

---

## 🧠 Technologies Used

| Component              | Description                                  |
|------------------------|----------------------------------------------|
| `YOLOv8n-seg`          | Lightweight segmentation model (Ultralytics) |
| `OpenCV`               | Video processing and visualization           |
| `DeepSORT`             | Object tracking                              |
| `BFMatcher, KNN, Lowe's Ratio Test`      | Feature matching                             |
| `MongoDB`              | Data storage for logs and detections         |
| `WebSocket`            | Streaming protocol                           |
| `EasyOCR`            | Character extractor                           |


## 🗂️ Project Structure
```
├── main.py                          # Main pipeline orchestrating the entire system
├── dimension_estimation.py          # Dimension calculation based on mask appearance
├── database.py                      # Database operations for storing tracking data
├── matching_and_tracking.py         # Object tracking, feature extraction, and matching
├── id_merging.py                    # Merge duplicate IDs for the same vehicle
├── ocr_license_plate.py            # License plate recognition from captured images
└── websocket_server.py             # WebSocket server for real-time video streaming
```
