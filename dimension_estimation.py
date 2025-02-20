import numpy as np

REFERENCE_HEIGHT_METERS = 2.2 
height_records = {}
frame_width = 256

def compute_reference_height(updated_tracks, detections, tampak_depan_data):
    for (track, class_name, track_id, mask), detection in zip(updated_tracks, detections):
        bbox, conf, class_name_detection, mask_detection = detection
        if class_name_detection == "Tampak Depan":
            height_from_mask = compute_height_from_mask(mask_detection)
            if height_from_mask:
                tampak_depan_data[track_id] = height_from_mask
    return tampak_depan_data  

def compute_height_from_mask(mask):
    if mask is None or len(mask) == 0:
        return None  
    y_coords = [point[1] for point in mask]  
    return max(y_coords) - min(y_coords)

def is_within_range(mask, frame_width):
    """
    Check if the object's mask is within the specified range (1/4 to 3/4 of frame width).
    """
    if mask is None or len(mask) == 0:
        return False
    
    x_coords = [point[0] for point in mask]
    min_x, max_x = min(x_coords), max(x_coords)
    
    left_bound = frame_width / 4
    right_bound = 3 * frame_width / 4
    
    return min_x >= left_bound and max_x <= right_bound

def estimate_height(tracked_objects, tampak_depan_data):
    estimated_heights = {}
    for track, class_name, track_id, mask in tracked_objects:
        if class_name == "Tampak Samping" and track_id in tampak_depan_data:
            current_mask_height = compute_height_from_mask(mask)
            known_mask_height = tampak_depan_data[track_id]
            
            if current_mask_height and known_mask_height:
                estimated_height = (current_mask_height / known_mask_height) * REFERENCE_HEIGHT_METERS
                
                if is_within_range(mask, frame_width):
                    if track_id not in height_records:
                        height_records[track_id] = []
                    height_records[track_id].append(estimated_height)
                
                estimated_heights[track_id] = estimated_height
                print(f"estimated height {estimated_height}")
    
    return estimated_heights, height_records

def get_final_estimated_heights(height_records):
    final_heights = {}
    for track_id, heights in height_records.items():
        if heights:
            final_heights[track_id] = np.mean(heights)
    
    print("Final Estimated Heights:")
    for track_id, height in final_heights.items():
        print(f"Track ID {track_id}: {height:.2f} meters")
    
    return final_heights
