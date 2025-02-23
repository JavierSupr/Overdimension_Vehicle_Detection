import numpy as np

REFERENCE_HEIGHT_METERS = 2.2 
height_records = {}
frame_width = 480

def compute_reference_height(updated_tracks, detections, tampak_depan_data):
    for (track, class_name, track_id, mask), detection in zip(updated_tracks, detections):
        bbox, conf, class_name_detection, mask_detection = detection
        if class_name_detection == "Tampak Depan":
            #print('masuk')
            height_from_mask = compute_height_from_mask(mask_detection)
            if height_from_mask:
                tampak_depan_data[track_id] = height_from_mask
    return tampak_depan_data  

#def compute_height_from_mask(mask):
    if mask is None or len(mask) == 0:
        return None  
    y_coords = [point[1] for point in mask]  
    return max(y_coords) - min(y_coords)
def compute_height_from_mask(mask):
    """
    Compute the height of an object from a binary mask by finding the maximum vertical extent.

    Args:
        mask: A 2D numpy array (binary mask) where nonzero values indicate the object.

    Returns:
        height: The computed height in pixels.
    """
    if mask is None or len(mask) == 0:
        return None  # No valid mask

    mask = np.array(mask)  # Ensure it's a numpy array

    # Get the row indices (y-coordinates) where the mask has nonzero values
    y_coords = np.where(mask > 0)[0]  # Extract only Y-axis values
    
    if len(y_coords) == 0:
        return None  # No object detected in mask

    # Compute height as the difference between max and min y-coordinates
    height = max(y_coords) - min(y_coords)
    return height





def is_within_range(mask, frame_width):
    """
    Check if the object's mask is within the specified range (1/4 to 3/4 of frame width).
    """
    if mask is None or len(mask) == 0:
        return False
    
    mask = np.array(mask)  # Ensure mask is a numpy array
    x_coords = np.where(mask > 0)[1]  # Extract X-axis values
    if len(x_coords) == 0:
        return False
    
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
                
                #if is_within_range(mask, frame_width):
                #    print("masuk2")
                if track_id not in height_records:
                    height_records[track_id] = []
                height_records[track_id].append(estimated_height)
                #print(f"estimated height {height_records}")
                estimated_heights[track_id] = estimated_height
    
    return estimated_heights, height_records

def get_final_estimated_heights(height_records):
    final_heights = {}
    for track_id, heights in height_records.items():
        if heights:
            final_heights[track_id] = np.mean(heights)
    
    for track_id, height in final_heights.items():
        print(f"Final Estimated Heights :Track ID {track_id}: {height:.2f} meters")
    
    return final_heights
