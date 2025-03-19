import numpy as np

REFERENCE_HEIGHT_METERS = 2.2 
height_records = {}
frame_width = 640

def compute_reference_height(tracking_results, tampak_depan_data):
    """
    Compute and store the reference height of 'Tampak Depan' objects from their mask data.
    """    
    for track in tracking_results:
        if track['class_name'] == "Tampak Depan":
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
def compute_height_from_mask(mask_xy):
    """
    Compute the height of an object from a mask contour.

    Args:
        mask_xy: A list of (x, y) coordinates representing the object contour.

    Returns:
        height: The computed height in pixels.
    """
    if mask_xy is None or len(mask_xy) == 0:
        return None  # No valid mask

    mask_xy = np.array(mask_xy)  # Ensure it's a numpy array

    # Get only y-coordinates from the contour
    y_coords = mask_xy[:, 1]  # Extract Y-axis values
    #print(f"y_coords{y_coords}")

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

def estimate_height(tracking_results, tampak_depan_data, height_records, passed_limits, final_heights):
    estimated_heights = {}
    left_limit = frame_width * (6/8)   # 1/4 of the screen width (exit point)
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
