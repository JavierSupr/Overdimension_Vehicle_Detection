
def iou(box1, box2):
    """Menghitung Intersection over Union (IoU) antara dua bounding box dengan format (x, y, w, h)."""
    x1, y1, w1, h1 = box1  # Detection Box (x, y, w, h)
    x2, y2, w2, h2 = box2  # Track Box (x, y, w, h)
    
    # Hitung koordinat interseksi
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Hitung luas interseksi
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2    
    
    # Hitung IoU
    iou_score = inter_area / box1_area
    return iou_score


def merge_track_ids(tracks, detections):
    """Assign the same track ID to 'Tampak Depan' and 'Tampak Samping' if they overlap with 'Truk'."""
    truck_tracks = []
    for track, detection in zip(tracks, detections):
        bbox, conf, class_name, mask = detection
        if class_name == "Truk":
            truck_tracks.append((track, bbox))
    
    updated_tracks = []
    for track, detection in zip(tracks, detections):
        bbox, conf, class_name, mask = detection
        track_id = track.track_id
        
        if class_name in ["Tampak Depan", "Tampak Samping"]:
            for truck_track, truck_bbox in truck_tracks:
                iou_score = iou(bbox, truck_bbox) 
                if iou_score > 0.3:  # Overlapping with a truck
                    #print(f"bbox {bbox} truck {truck_bbox} iou score {iou_score}")

                    track_id = truck_track.track_id
                    break
        
        updated_tracks.append((track, class_name, track_id, mask))
        
    
    return updated_tracks