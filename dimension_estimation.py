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

def estimate_height(tracking_results, tampak_depan_data, height_records, object_tracking_status, is_append):
    estimated_heights = {}
    detected_track_ids = []  # To store track_ids that passed for frame capture
    left_limit = frame_width * (6/8)   # 1/4 of the screen width (exit point)
    right_limit = frame_width * (7/8) 
    detected_track_ids = []  # To store track_ids that passed for frame capture
    to_remove = []  # Temporary list to store track IDs to be removed

    for track in tracking_results:
        if track['class_name'] == "Tampak Samping" and track['track_id'] in tampak_depan_data:
            x_min, y_min, x_max, y_max = track['bounding box']  # Extract bounding box coordinates
            print(f"xmin {x_min}")
            track_id = track['track_id']
            # Track object entering from the right
            if x_min >= right_limit and x_min <= left_limit: 
                print("masukk") 
                continue

            # Check if the object has now exited past the left limit
            if track_id in object_tracking_status and x_min <= left_limit:
                print(f"The object {track_id} passed")  # Print when the object fully passes
                detected_track_ids.append(track_id)  # Store the ID for mask drawing
                to_remove.append(track_id)  # Mark for removal

            # Height estimation process (only for valid tracking range)
            if right_limit >= x_min >= left_limit:
                current_mask_height = compute_height_from_mask(track['mask'])
                known_mask_height = tampak_depan_data[track_id]

                if current_mask_height and known_mask_height:
                    estimated_height = (current_mask_height / known_mask_height) * REFERENCE_HEIGHT_METERS

                    if track_id not in height_records:
                        height_records[track_id] = []
                    height_records[track_id].append(estimated_height)

                    estimated_heights[track_id] = estimated_height
                    print(f"height_records {height_records}")

    # Remove objects that have passed (outside the loop to avoid modifying the dictionary while iterating)
    for track_id in to_remove:
        object_tracking_status.pop(track_id, None)  # Remove safely

    # Capture frame if an object passed
    if detected_track_ids:
        #captured_frame = draw_mask_on_detected_tracks(frame, updated_tracks, detected_track_ids)
        #cv2.imwrite(f"captured_object_passed.png", captured_frame)  # Save the frame
        print("Frame captured and saved!")

    return estimated_heights, height_records, object_tracking_status, is_append

def get_final_estimated_heights(height_records):
    final_heights = {}
    for track_id, heights in height_records.items():
        if heights:
            final_heights[track_id] = np.mean(heights)
            #print(f"final height {final_heights}")
    
    #for track_id, height in final_heights.items():
        #print(f"Final Estimated Heights :Track ID {track_id}: {height:.2f} meters")
    
    return final_heights
