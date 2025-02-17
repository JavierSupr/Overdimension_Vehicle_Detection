import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

deepsort = DeepSort(max_age=10)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
id_mappings = {}

tampak_depan_data = {}

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

    #return descriptor, keypoints
def update_id_mappings(tracks1, tracks2, keypoints1, keypoints2, good_matches):
    track1_matches = {}
    track2_matches = {}
    
    for track1 in tracks1:
        if track1.is_confirmed():
            track1_matches[track1.track_id] = get_track_feature_points(track1, keypoints1, good_matches, True)
    
    for track2 in tracks2:
        if track2.is_confirmed():
            track2_matches[track2.track_id] = get_track_feature_points(track2, keypoints2, good_matches, False)
    
    for track2_id, matches2 in track2_matches.items():
        if not matches2:
            continue
        
        best_track1_id = None
        max_matches = 0
        
        for track1_id, matches1 in track1_matches.items():
            common_matches = len(set(matches2) & set(matches1))
            if common_matches > max_matches:
                max_matches = common_matches
                best_track1_id = track1_id
        
        if max_matches >= 2:  # Threshold for minimum matches
            id_mappings[track2_id] = best_track1_id

def get_track_feature_points(track, keypoints, good_matches, is_cam1):
    ltrb = track.to_ltrb()
    x1, y1, x2, y2 = map(int, ltrb)
    track_points = []
    
    for match in good_matches:
        pt = keypoints[match.queryIdx].pt if is_cam1 else keypoints[match.trainIdx].pt
        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
            track_points.append(match)
    
    return track_points

def match_features(descriptors1, descriptors2, tracks1, tracks2, keypoints1, keypoints2):
    """
    Match SIFT features between two frames and update ID mappings.
    
    Args:
        descriptors1: SIFT descriptors from first frame
        descriptors2: SIFT descriptors from second frame
        tracks1: DeepSORT tracks from first frame
        tracks2: DeepSORT tracks from second frame
        keypoints1: SIFT keypoints from first frame
        keypoints2: SIFT keypoints from second frame
    
    Returns:
        list: List of good matches between frames
    """
    good_matches = []
    print (f" descriptor1 {descriptors1}, descriptor2 {descriptors2}, track1 {tracks1}, track2 {tracks2}, keypoint 1 {keypoints1}, keypoint 2 {keypoints2}")
    if descriptors1 and descriptors2:
        try:
            descriptors1 = np.array(descriptors1)
            descriptors2 = np.array(descriptors2)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            update_id_mappings(tracks1, tracks2, keypoints1, keypoints2, good_matches)
            
        except Exception as e:
            print(f"Error in feature matching: {e}")
    
    return good_matches


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


def merge_track_ids(tracks, detections):
    """Assign the same track ID to 'Tampak Depan' and 'Tampak Samping' if they overlap with 'Truk'."""
    truck_tracks = []
    for track, detection in zip(tracks, detections):
        bbox, conf, class_name, mask = detection
        if class_name == "Truk":
            truck_tracks.append((track, bbox))
    
    updated_tracks = []
    for track, detection in zip(tracks, detections):
        bbox, conf, class_name, mask = detection
        track_id = track.track_id
        
        if class_name in ["Tampak Depan", "Tampak Samping"]:
            for truck_track, truck_bbox in truck_tracks:
                iou_score = iou(bbox, truck_bbox) 
                if iou_score > 0.3:  # Overlapping with a truck
                    #print(f"bbox {bbox} truck {truck_bbox} iou score {iou_score}")

                    track_id = truck_track.track_id
                    break
        
        updated_tracks.append((track, class_name, track_id, mask))
    
    return updated_tracks


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
    
    # Merge track IDs based on IoU
    tracked_objects = merge_track_ids(tracks, detections)
    #print(f"tracked_objects {tracked_objects}")
    
    # Convert frame to grayscale for SIFT
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints, descriptors = [], []
    
    for track, class_name, track_id, mask in tracked_objects:
        # Extract SIFT features for each tracked object
        kp, des = extract_sift_features(track, gray_frame)
        if kp:
            keypoints.extend(kp)
            if des is not None:
                descriptors.extend(des)
    
    return tracked_objects, keypoints, descriptors


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



REFERENCE_HEIGHT_METERS = 2.2  # Fixed reference height for 'Tampak Depan'

def compute_reference_height(updated_tracks, detections):

    for (track, class_name, track_id, mask), detection in zip(updated_tracks, detections):
        bbox, conf, class_name_detection, mask_detection = detection
        if class_name_detection == "Tampak Depan":
            height_from_mask = compute_height_from_mask(mask_detection)
            if height_from_mask:
                tampak_depan_data[track.track_id] = height_from_mask

    return tampak_depan_data  # Tambahkan return jika ingin mengembalikan data

def compute_height_from_mask(mask):
    """
    Compute the height of an object based on the mask by finding the maximum vertical distance.
    
    Args:
        mask: List of (x, y) coordinates forming the segmentation mask, possibly as a NumPy array.

    Returns:
        height: The computed height from the mask.
    """
    if mask is None or len(mask) == 0:
        return None  # No valid mask

    # Ensure mask is a list of points, as some YOLO segmentation models return multiple polygons
    if isinstance(mask, list):
        # Flatten all y-coordinates if the mask contains multiple segments
        y_coords = [point[1] for segment in mask for point in segment]
    else:
        # Single segment case
        y_coords = mask[:, 1]  # Extract y-values assuming it's a NumPy array

    if len(y_coords) == 0:
        return None  # No valid y-coordinates

    # Compute the height as the vertical distance between max and min y-coordinates
    height = float(max(y_coords) - min(y_coords))  # Convert to float for safety
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
    print(f"tampak_depan_data[track_id] {tampak_depan_data}")
    #print("0")
    for track, class_name, track_id, mask in tracked_objects:
        print("1")
        if class_name == "Tampak Samping" and track_id in tampak_depan_data:
            print(f"track id {track_id}, tipe data: {type(track_id)}")
            known_mask_height = tampak_depan_data[track_id]  # Height from mask
            current_mask_height = compute_height_from_mask(mask)
            print("2")
            if current_mask_height and known_mask_height:
                print("3")
                # Scale using the reference height (2.2m for 'Tampak Depan')
                estimated_height = (current_mask_height / known_mask_height) * REFERENCE_HEIGHT_METERS
                estimated_heights[track_id] = estimated_height
                print(f"Estimated height for Tampak Samping (Track ID {track_id}): {estimated_height:.2f} meters")

    return estimated_heights

