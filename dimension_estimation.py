import numpy as np
from collections import defaultdict

REFERENCE_HEIGHT_METERS = 2.2 
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
    #print()

    return max_height

def estimate_height(tracking_results, tampak_depan_data, height_records, passed_limits, final_heights):
    estimated_heights = {}
    left_limit = frame_width * (1/2)   # 1/4 of the screen width (exit point)
    right_limit = frame_width * (7/8) 

    for track in tracking_results:
        if track['class_name'] == "Tampak Samping" and track['track_id'] in tampak_depan_data:
            x_min, y_min, x_max, y_max = track['bounding box']  # Extract bounding box coordinates

            if track['track_id'] not in passed_limits:
                passed_limits[track['track_id']] = {"left": None, "right": False}

            # Check if object has crossed left limit
            #if x_min <= left_limit:
            #    passed_limits[track['track_id']]["left"] = True

            # Check if object has crossed right limit
            if x_min <= right_limit:
                passed_limits[track['track_id']]["right"] = True

            if x_min >= right_limit:
                continue

            current_mask_height = compute_height_from_mask(track['mask'])
            known_mask_height = tampak_depan_data[track['track_id']]

            if passed_limits[track['track_id']]["right"] == True and passed_limits[track['track_id']]["left"] == None:
                if current_mask_height and known_mask_height:
                    estimated_height = (current_mask_height / known_mask_height) * REFERENCE_HEIGHT_METERS

                    if track['track_id'] not in height_records:
                        height_records[track['track_id']] = []
                    height_records[track['track_id']].append(estimated_height)

                    estimated_heights[track['track_id']] = estimated_height
                    #print(f"height_records {height_records}")
                #print(f"passed limits {passed_limits}")
            # Check if the object is within the desired horizontal range
                if x_min <= left_limit:
                    passed_limits[track['track_id']]["left"] = True
                    if (passed_limits[track['track_id']]["left"] and passed_limits[track['track_id']]["right"]):
                        final_heights = get_final_estimated_heights(height_records, final_heights)
                        passed_limits[track['track_id']]["right"] = False
                    continue
                
    # Ensure function always returns values
    return final_heights, height_records, passed_limits


def get_final_estimated_heights(height_records, final_heights):

    for track_id, heights in height_records.items():
        if heights:
            final_heights[track_id] = np.mean(heights)
            #print(f"final height {final_heights}")
    
    #for track_id, height in final_heights.items():
        #print(f"Final Estimated Heights :Track ID {track_id}: {height:.2f} meters")
    
    return final_heights
