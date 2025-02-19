import numpy as np

REFERENCE_HEIGHT_METERS = 2.2 

def compute_reference_height(updated_tracks, detections, tampak_depan_data):

    for (track, class_name, track_id, mask), detection in zip(updated_tracks, detections):
        bbox, conf, class_name_detection, mask_detection = detection
        if class_name_detection == "Tampak Depan":
            height_from_mask = compute_height_from_mask(mask_detection)
            if height_from_mask:
                tampak_depan_data[track_id] = height_from_mask

    return tampak_depan_data  # Tambahkan return jika ingin mengembalikan data

def compute_height_from_mask(mask):
    """
    Compute the height of an object from a binary mask by finding the maximum vertical extent.
    
    Args:
        mask: A 2D numpy array (binary mask) where nonzero values indicate the object.
    
    Returns:
        height: The computed height in pixels.
    """
    if mask is None or mask.size == 0:
        return None  # No valid mask

    # Get the row indices (y-coordinates) where the mask has nonzero values
    y_coords = np.where(mask > 0)[0]  # Extract only Y-axis values
    
    if len(y_coords) == 0:
        return None  # No object detected in mask

    # Compute height as the difference between max and min y-coordinates
    height = max(y_coords) - min(y_coords)
    return height



def estimate_height(tracked_objects, tampak_depan_data):
    """
    Estimates the height of 'Tampak Samping' based on the height of 'Tampak Depan' (fixed at 2.2m).
    
    Args:
        tracked_objects: List of tuples containing (track, class_name, track_id, mask, bbox).
        tampak_depan_data: Dictionary containing mask-based heights of 'Tampak Depan' by track_id.
    
    Returns:
        estimated_heights: Dictionary with track_id as key and estimated height in meters as value.
    """
    estimated_heights = {}
    #print(f"tampak_depan_data[track_id] {tampak_depan_data}")
    #print("0")
    for track, class_name, track_id, mask in tracked_objects:
        if class_name == "Tampak Samping":
            current_mask_height = compute_height_from_mask(mask)
            #print(f"current_mask_height tampak samping {current_mask_height} track id {track_id}")

            print(f"track_id {track_id}")
            if track_id in tampak_depan_data:
            # print(f"track id {track_id}, tipe data: {type(track_id)}")
                known_mask_height = tampak_depan_data[track_id]  # Height from mask
                current_mask_height = compute_height_from_mask(mask)
                #print("2")
                if current_mask_height and known_mask_height:
                   # print("3")
                    ## Scale using the reference height (2.2m for 'Tampak Depan')
                    estimated_height = (current_mask_height / known_mask_height) * REFERENCE_HEIGHT_METERS
                    estimated_heights[track_id] = estimated_height
                    print(f"Estimated height for Tampak Samping (Track ID {track_id}): {current_mask_height}/{known_mask_height} : {estimated_height:.2f} meters")

    return estimated_heights