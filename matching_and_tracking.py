import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

deepsort = DeepSort(max_age=5)
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

def process_tracks_and_extract_features(deepsort, detections, frame):
    """
    Updates tracks using DeepSORT, converts frame to grayscale,
    and extracts SIFT features for tracked objects.
    
    Args:
        deepsort: DeepSORT object for tracking.
        detections: List of detections to be processed.
        frame: The current video frame.
    
    Returns:
        keypoints: List of extracted keypoints.
        descriptors: List of extracted descriptors.
    """
    # Update tracks with detections
    tracks = deepsort.update_tracks(detections, frame=frame)
    
    # Convert frame to grayscale for SIFT
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize keypoints and descriptors lists
    keypoints, descriptors = [], []
    
    # Extract SIFT features for tracked objects
    #for track in tracks:
        #kp, des = extract_sift_features(track, gray_frame)
        #if kp:
        #    keypoints.extend(kp)
        #    if des is not None:
        #        descriptors.extend(des)
    
    return tracks, keypoints, descriptors

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


def draw_tracking_info(frame, tracks, is_cam1=True):
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        if not is_cam1:
            track_id = id_mappings.get(track_id, track_id)
        
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for tracked objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID
        label = f'ID: {track_id}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame