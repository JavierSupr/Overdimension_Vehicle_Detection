import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

deepsort = DeepSort(max_age=10)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
id_mappings = {}


def extract_sift_features(track, gray_frame):
    if not track.is_confirmed():
        return [], None
    
    ltrb = track.to_ltrb()
    x1,y1,x2,y2 = map(int, ltrb)

    h, w = gray_frame.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, w)
    y2 = min(y2, h)

    if x2 <= x1 or y2 <= y1:
        return [], None
    
    roi = gray_frame[y1:y2, x1:x2]

    if roi.size == 0 or roi.dtype != np.uint8:
        return [], None
    
    kp, des = sift.detectAndCompute(roi, None)

    if kp is not None:
        for kp_item in kp:
            kp_item.pt = (kp_item.pt[0] + x1, kp_item.pt[1] + y1)

    return kp, des

def update_id_mappings(updated_tracks1, updated_tracks2, keypoints1, keypoints2, good_matches, id_mapping):
    """
    Updates ID mappings by assigning Camera 1 object IDs to the corresponding Camera 2 object IDs.
    
    Args:
        tracks1 (list): Bounding boxes and IDs from Camera 1.
        tracks2 (list): Bounding boxes and IDs from Camera 2.
        keypoints1 (list): Keypoints from Camera 1.
        keypoints2 (list): Keypoints from Camera 2.
        good_matches (list): List of good matches between keypoints from both cameras.
    
    Returns:
        dict: Updated ID mappings {old_id: new_id}.
    """
    # Create a mapping from Camera 2 IDs to Camera 1 IDs
    
    for match in good_matches:
        kp1_idx = match.queryIdx  # Index of keypoint in Camera 1
        kp2_idx = match.trainIdx  # Index of keypoint in Camera 2
        
        # Find the track ID associated with the matched keypoints
        id1, id2 = None, None
        
        for track_obj in updated_tracks2:
            track2, class_name2, track_id2, mask2 = track_obj
            if is_point_in_bbox(keypoints2[kp2_idx].pt, track2.to_tlbr()):  # Use track.to_tlbr() for bounding box
                id2 = track_id2  # Get the track ID
                break
        
        for track_obj in updated_tracks1:
            track1, class_name1, track_id1, mask1 = track_obj
            if is_point_in_bbox(keypoints1[kp1_idx].pt, track1.to_tlbr()):  # Use track.to_tlbr()
                id1 = track_id1  # Get the track ID
                break
        
        if id1 is not None and id2 is not None:
            # Assign Camera 2 ID to Camera 1 ID
            if id2 not in id_mapping:
                id_mapping[id2] = id1
        
    # Update tracks in Camera 1 with IDs from Camera 2
    for i, (track1, class_name1, track_id1, mask1) in enumerate(updated_tracks1):
        if track_id1 in id_mapping.values():
            for id2, id1 in id_mapping.items():
                if track_id1 == id1:
                    updated_tracks1[i] = (track1, class_name1, id2, mask1)  # Update ID
                    print("ID On Camera 1 Updated")
                    break
    return id_mapping


def is_point_in_bbox(point, bbox):
    """
    Checks if a point is inside a bounding box.
    
    Args:
        point (tuple): (x, y) coordinates of the point.
        bbox (list): Bounding box in format [x_min, y_min, width, height].
    
    Returns:
        bool: True if the point is inside the bounding box, otherwise False.
    """
    x, y = point
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    return x_min <= x <= x_max and y_min <= y <= y_max


def match_features(descriptors1, descriptors2, updated_tracks1, updated_tracks2, keypoints1, keypoints2):
    """
    Match SIFT features between two frames and update ID mappings.
    
    Args:
        descriptors1: SIFT descriptors from first frame
        descriptors2: SIFT descriptors from second frame
        updated_tracks1: Updated tracks from first frame (list of tuples containing (track, class_name, track_id, mask))
        updated_tracks2: Updated tracks from second frame
        keypoints1: SIFT keypoints from first frame
        keypoints2: SIFT keypoints from second frame
    
    Returns:
        list: List of good matches between frames
    """
    id_mapping = {}  # {id2: id1}

    good_matches = []

    if descriptors1 is not None and descriptors2 is not None:
        try:
            descriptors1 = np.array(descriptors1)
            descriptors2 = np.array(descriptors2)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            id_mapping = update_id_mappings(updated_tracks1, updated_tracks2, keypoints1, keypoints2, good_matches, id_mapping)
            
        except Exception as e:
            print(f"Error in feature matching: {e}")
    
    return good_matches, id_mapping

def process_tracks_and_extract_features(deepsort, detections, frame):
    """
    Updates tracks using DeepSORT, converts frame to grayscale,
    and extracts SIFT features for tracked objects while pairing each track with its detection class.
    
    Args:
        deepsort: DeepSORT object for tracking.
        detections: List of detections (each detection is a tuple of (bbox, conf, class_name, mask)).
        frame: The current video frame.
    
    Returns:
        tracked_objects: List of tuples containing (track, detection_class, track_id, mask).
        keypoints: List of extracted keypoints.
        descriptors: List of extracted descriptors.
    """
    # Update tracks with detections
    tracks = deepsort.update_tracks(detections, frame=frame)

    # Convert frame to grayscale for SIFT
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints, descriptors = [], []
    
    for track in tracks:
        # Extract SIFT features for each tracked object
        kp, des = extract_sift_features(track, gray_frame)
        if kp:
            keypoints.extend(kp)
            if des is not None:
                descriptors.extend(des)
    
    return tracks, keypoints, descriptors


def draw_tracking_info(frame, tracks, estimated_heights, id_mappings=None, is_cam1=True):
    """Draws tracking information on the frame, including bounding boxes, IDs, and estimated heights."""
    if id_mappings is None:
        id_mappings = {}

    for track_obj in tracks:
        track, class_name, track_id, mask = track_obj
        if not track.is_confirmed():
            continue
        
        if not is_cam1:
            track_id = id_mappings.get(track_id, track_id)
        
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for tracked objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and class name
        label = f'ID: {track_id} ({class_name})'
        cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check if the estimated height is available
        height_label = ""
        if track_id in estimated_heights:
            estimated_height = estimated_heights[track_id]
            height_label = f'Height: {estimated_height:.2f}m'
            cv2.putText(frame, height_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


