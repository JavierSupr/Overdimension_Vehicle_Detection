import numpy as np
from collections import defaultdict, Counter

REFERENCE_HEIGHT_METERS = 1.7 
height_records = {}
frame_width = 640

def compute_reference_height(tracking_results, tampak_depan_data):
    """
    Compute and store the reference height of 'Tampak Depan' objects from their mask data.
    """    
    for track in tracking_results:
        if track['class_name'] == "Tampak Depan":
            #print(f"tampak depan id {track['track_id']}")
            height_from_mask = compute_height_from_mask(track['mask'])
            if height_from_mask:
                tampak_depan_data[track['track_id']] = height_from_mask
                #print(f"tampak depan data {tampak_depan_data}")
    
    return tampak_depan_data

#def compute_height_from_mask(mask):
    if mask is None or len(mask) == 0:
        return None  
    y_coords = [point[1] for point in mask]  
    return max(y_coords) - min(y_coords)

def compute_height_from_mask(mask_xy, rounding=1.0):
    """
    Compute the maximum vertical height (ymax - ymin) for vertical lines at same/similar x positions.
    
    Args:
        mask_xy: A list of (x, y) coordinates representing the object contour.
        rounding: The step size to round x values (e.g., 1.0 or 0.5) for grouping similar x values.

    Returns:
        max_height: The maximum vertical height found for any x group.
    """
    if mask_xy is None or len(mask_xy) == 0:
        return None
    #print(mask_xy)
    mask_xy = np.array(mask_xy)
    grouped_y = defaultdict(list)

    # Group y-values by rounded x-values
    #print(f"mask xy {mask_xy}")
    for x, y in mask_xy:
        x_rounded = round(x / rounding) * rounding
        grouped_y[x_rounded].append(y)

    # Compute height for each x group
    max_height = 0
    #print(f" nilai y {grouped_y.values()}")
    for y_list in grouped_y.values():
        if len(y_list) >= 2:
            height = max(y_list) - min(y_list)
            #print(f" {max(y_list)} - {min(y_list)} = {height}")
            if height > max_height:
                max_height = height
    #print(f"max height {max_height}")
    #print()

    return max_height

def estimate_height(tracking_results, tampak_depan_data, height_records, passed_limits, final_heights, camera_name, id_mapping):
    estimated_heights = {}
    truck_types_by_id = {}  # Menyimpan jenis truk per ID

    if camera_name == "Camera 1":
        left_limit = frame_width * (7/8)
        right_limit = frame_width * (0.34)
    else:  # Assume Camera 2
        left_limit = frame_width * (0.22)
        right_limit = frame_width * (7/8)

    # --- Simpan jenis truk berdasarkan ID ---
    for t in tracking_results:
        if "Truk" in t['class_name']:
            truck_types_by_id[t['track_id']] = t['class_name']

    for track in tracking_results:
        track_id = track['track_id']
        class_name = track['class_name']

        truck_type = truck_types_by_id.get(track_id)
        if not truck_type:
            continue  # Lewati jika tidak ada info jenis truk

        # Tetapkan tinggi referensi
        if truck_type == "Truk Kecil":
            REFERENCE_HEIGHT_METERS = 1.7
            tambahan_tinggi = 1.0
        elif truck_type == "Truk Besar":
            REFERENCE_HEIGHT_METERS = 2.2
            tambahan_tinggi = 1.6
        else:
            continue

        # Proses hanya jika Tampak Samping dan ada data mask tampak depan
        if class_name == "Tampak Samping" and track_id in tampak_depan_data:
            x_min, y_min, x_max, y_max = track['bounding box']

            if track_id not in passed_limits:
                passed_limits[track_id] = {"left": None, "right": False}

            if camera_name == "Camera 1":
                if x_max >= right_limit:
                    passed_limits[track_id]["right"] = True
                if x_max <= right_limit:
                    continue
            else:  # Camera 2
                if x_min <= right_limit:
                    passed_limits[track_id]["right"] = True
                if x_min >= right_limit:
                    continue

            current_mask_height = compute_height_from_mask(track['mask'])
            known_mask_height = tampak_depan_data[track_id]

            if passed_limits[track_id]["right"] and passed_limits[track_id]["left"] is None:
                if current_mask_height and known_mask_height:
                    estimated_height = (current_mask_height / known_mask_height) * REFERENCE_HEIGHT_METERS
                    if track_id not in height_records:
                        height_records[track_id] = []
                    height_records[track_id].append(estimated_height)
                    estimated_heights[track_id] = estimated_height

                if (camera_name == "Camera 1" and x_max >= left_limit) or \
                   (camera_name == "Camera 2" and x_min <= left_limit):                    
                    passed_limits[track_id]["left"] = True
                    if passed_limits[track_id]["left"] and passed_limits[track_id]["right"]:
                        final_heights = get_final_estimated_heights(height_records, final_heights,truck_types_by_id)
                        #print(f"final estimates {list(final_estimates.items())[-1]}")
                        ## Tambahkan tambahan_tinggi ke nilai final
                        #last_tid, last_val = list(final_estimates.items())[-1]
                        #truck_type = truck_types_by_id.get(last_tid, "")
                        #print(f"tid {last_tid} - val {last_val} - truck_type {truck_type}")
                        #if truck_type == "Truk Kecil":
                        #    final_heights[last_tid] = last_val + 1.0
                        #elif truck_type == "Truk Besar":
                        #    final_heights[last_tid] = last_val + 1.6
                        #else:
                        #    print("masuk1")
                        #    final_heights[last_tid] = last_val
                        print(f"final heights {final_heights}")
                        passed_limits[track_id]["right"] = False
                    #print()
                    continue

    return final_heights, height_records, passed_limits



def get_final_estimated_heights(height_records, final_heights, truck_types_by_id):
    if not height_records:
        return final_heights

    # Ambil hanya entry terakhir yang masuk
    last_track_id = list(height_records.keys())[-1]
    heights = height_records[last_track_id]

    if heights:
        # Pembulatan ke 1 angka di belakang koma
        rounded_heights = [round(h, 1) for h in heights]
        count = Counter(rounded_heights)

        # Ambil hanya nilai dengan frekuensi >= 3
        valid_heights = [h for h, c in count.items() if c >= 2]

        if valid_heights:
            base_height = max(valid_heights)
        else:
            base_height = 0  # fallback, bisa diganti rata-rata misal np.mean(rounded_heights)

        # Tambahkan tinggi berdasarkan jenis truk
        truck_type = truck_types_by_id.get(last_track_id, "")
        if truck_type == "Truk Kecil":
            final_heights[last_track_id] = base_height + 1.0
        elif truck_type == "Truk Besar":
            final_heights[last_track_id] = base_height + 1.6
        else:
            final_heights[last_track_id] = base_height

    return final_heights
