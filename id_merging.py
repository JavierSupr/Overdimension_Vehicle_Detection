
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



def merge_track_ids(tracking_results):
    """Assign the same track ID to 'Tampak Depan' and 'Tampak Samping' if they overlap with 'Truk'."""
    truck_tracks = []
    for track in tracking_results:
        if track['class_name'] == "Truk":
            truck_tracks.append((track['track_id'], track['bounding box']))
    
    updated_tracks = []
    for track in tracking_results:
        track_id = track['track_id']
        if track['class_name'] in ["Tampak Depan", "Tampak Samping"]:
            for truck_id, truck_bbox in truck_tracks:
                if iou(track['bounding box'], truck_bbox) > 0.3:
                    track_id = truck_id
                    break
        
        updated_tracks.append({
            'track_id': track_id,
            'class_name': track['class_name'],
            'bounding box': track['bounding box'],
            'kp': track['kp'],
            'des': track['des'],
            'mask': track['mask']
        })

    
    return updated_tracks
