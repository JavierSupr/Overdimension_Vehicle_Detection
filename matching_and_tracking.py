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
            kp_item.pt = (kp_item.pt[0] + x1, kp_item.pt[1] + 1)

    return kp, des

def initialize(detections,frame):
    tracks = deepsort.update_tracks(detections, frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptor = [], []

    for track in tracks:
        kp, des = extract_sift_features(track, gray_frame)
        if kp:
            keypoints.extend(kp)
            if des is not None:
                descriptor.extend(des)

    return descriptor, tracks, keypoints
def update_id_mappings(tracks1, tracks2, keypoints1, keypoints2, good_matches):
    """Update ID mappings based on SIFT matches"""
    # Create a mapping of matches per track
    track1_matches = {}
    track2_matches = {}
    
    for track1 in tracks1:
        if track1.is_confirmed():
            track1_matches[track1.track_id] = get_track_feature_points(track1, keypoints1, good_matches, True)
    
    for track2 in tracks2:
        if track2.is_confirmed():
            track2_matches[track2.track_id] = get_track_feature_points(track2, keypoints2, good_matches, False)
    
    # Find tracks with the most matching features
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
        
        if max_matches >= 5:  # Threshold for minimum matches
            id_mappings[track2_id] = best_track1_id


def get_track_feature_points(track, keypoints, good_matches, is_cam1):
    """Get the feature points associated with a specific track"""
    ltrb = track.to_ltrb()
    x1, y1, x2, y2 = map(int, ltrb)
    track_points = []
    
    for match in good_matches:
        pt = keypoints[match.queryIdx].pt if is_cam1 else keypoints[match.trainIdx].pt
        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
            track_points.append(match)
    
    return track_points

def match_feature(descriptor1, descriptor2, tracks1, tracks2, keypoints1, keypoints2):
    good_matches = []
    if descriptor1 and descriptor2:
        try:
            descriptor1 = np.array(descriptor1)
            descriptor2 = np.array(descriptor2)
            matches = bf.knnMatch(descriptor1, descriptor2, k=2)

            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            update_id_mappings(tracks1, tracks2, keypoints1, keypoints2, good_matches)
        except Exception as e:
            print(f"Error in feature matching: {e}")