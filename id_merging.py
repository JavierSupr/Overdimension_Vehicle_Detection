def iou(box1, box2):
    """
    Mengembalikan True jika box1 setidaknya 80% berada di dalam box2.
    Box dalam format (x, y, w, h).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Hitung koordinat interseksi
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    # Hitung luas interseksi
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)

    # Cek apakah minimal 80% box1 berada dalam box2
    return inter_area / box1_area if box1_area != 0 else 0



def merge_track_ids(tracking_results, id_merge):
    """
    Assign the same track ID to 'Tampak Depan' and 'Tampak Samping' if they overlap with a 'Truk'.
    Uses a dictionary to avoid recomputing IOU for the same track IDs.
    """
    truck_tracks = []
    
    # Kumpulkan semua bounding box untuk "Truk"
    for track in tracking_results:
        if track['class_name'] in ["Truk Besar", "Truk Kecil"]:
            truck_tracks.append((track['track_id'], track['bounding box']))
    
    updated_tracks = []
    
    for track in tracking_results:
        old_id = track['track_id']
        class_name = track['class_name']
        bbox = track['bounding box']

        # Cek apakah track ini perlu digabungkan ke ID truk
        if class_name in ["Tampak Depan", "Tampak Samping"]:
            if old_id in id_merge:
                # Gunakan ID yang sudah dipetakan
                new_id = id_merge[old_id]
            else:
                # Cek IOU dengan setiap truk
                new_id = old_id  # default pakai ID lama
                for truck_id, truck_bbox in truck_tracks:
                    if iou(bbox, truck_bbox) >= 0.8:
                        new_id = truck_id
                        id_merge[old_id] = truck_id  # simpan mapping
                        break
        else:
            new_id = old_id
        #print(f"id mapping {id_mapping}")
        updated_tracks.append({
            'track_id': new_id,
            'class_name': class_name,
            'bounding box': bbox,
            'kp': track['kp'],
            'des': track['des'],
            'mask': track['mask']
        })

    return updated_tracks

