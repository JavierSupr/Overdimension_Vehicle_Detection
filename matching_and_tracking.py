import cv2
import numpy as np
import random
from deep_sort_realtime.deepsort_tracker import DeepSort

deepsort = DeepSort(max_age=30)  # Inisialisasi DeepSORT

sift = cv2.SIFT_create()
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
    Updates ID mappings by assigning Camera 2 object IDs to the corresponding Camera 1 object IDs.
    
    Args:
        updated_tracks1 (list): Bounding boxes and IDs from Camera 1.
        updated_tracks2 (list): Bounding boxes and IDs from Camera 2.
        keypoints1 (list): Keypoints from Camera 1.
        keypoints2 (list): Keypoints from Camera 2.
        good_matches (list): List of good matches between keypoints from both cameras.
        id_mapping (dict): Existing ID mappings {Camera2_ID: Camera1_ID}.
    
    Returns:
        list: Updated tracks1 with assigned IDs from Camera 2.
    """
    # Create a mapping from Camera 2 IDs to Camera 1 IDs
    for match in good_matches:
        kp1_idx = match.queryIdx  # Index of keypoint in Camera 1
        kp2_idx = match.trainIdx  # Index of keypoint in Camera 2
        
        # Find the track ID associated with the matched keypoints
        id1, id2 = None, None
        
        for track_obj in updated_tracks2:
            track2, class_name2, track_id2, mask2 = track_obj
            if is_point_in_bbox(keypoints2[kp2_idx].pt, track2.to_tlbr()):
                id2 = track_id2
                break
        
        for track_obj in updated_tracks1:
            track1, class_name1, track_id1, mask1 = track_obj
            if is_point_in_bbox(keypoints1[kp1_idx].pt, track1.to_tlbr()):
                id1 = track_id1
                break
        
        if id1 is not None and id2 is not None:
            # Assign Camera 2 ID to Camera 1 ID
            id_mapping[id2] = id1
    
    # Update tracks in Camera 1 with IDs from Camera 2
    updated_tracks = []
    for track1, class_name1, track_id1, mask1 in updated_tracks1:
        new_id = track_id1  # Default to the existing ID
        for id2, id1 in id_mapping.items():
            if track_id1 == id1:
                new_id = id2  # Assign the Camera 2 ID
                print(f"Updated Camera 1 ID {track_id1} to Camera 2 ID {new_id}")
                break
        
        updated_tracks.append((track1, class_name1, new_id, mask1))
    
    return updated_tracks



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
    #print(f"keypoint1 {keypoints1}")
    #print(f"keypoint2 {keypoints2}")
    if descriptors1 is not None and descriptors2 is not None and len(descriptors1) > 0 and len(descriptors2) > 0:
        try:
            # Ensure descriptors are numpy arrays of type float32
            descriptors1 = np.asarray(descriptors1, dtype=np.float32)
            descriptors2 = np.asarray(descriptors2, dtype=np.float32)

            # Check shape consistency
            if descriptors1.shape[1] != descriptors2.shape[1]:
                print(f"Shape mismatch: descriptors1.shape={descriptors1.shape}, descriptors2.shape={descriptors2.shape}")
                return [], updated_tracks1

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)


            updated_tracks1 = update_id_mappings(updated_tracks1, updated_tracks2, keypoints1, keypoints2, good_matches, id_mapping)

        except Exception as e:
            print(f"Error in feature matching: {e}")

    return good_matches, updated_tracks1



def extract_sift_features(track, gray_frame):
    """
    Extracts SIFT features from the bounding box of the given track in the grayscale frame.

    Args:
        track: A DeepSORT track object with bbox in [x, y, w, h] format.
        gray_frame: The grayscale video frame.

    Returns:
        keypoints: List of detected keypoints.
        descriptors: Corresponding descriptors of the keypoints.
    """
    x, y, w, h = map(int, track.to_tlwh())  # Convert bbox to integer

    # Ensure bounding box is within frame limits
    h, w = gray_frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + w), min(h, y + h)

    # Extract ROI from grayscale frame
    roi = gray_frame[y1:y2, x1:x2]

    if roi.size == 0:
        return [], None  # Return empty if ROI is invalid

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect and compute keypoints & descriptors
    keypoints, descriptors = sift.detectAndCompute(roi, None)

    return keypoints, descriptors

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_tracks_and_extract_features(detections, frame):
    
    detections2 = [det[:3] for det in detections]
    tracks = deepsort.update_tracks(detections2, frame=frame)
    # Konversi frame ke grayscale untuk ekstraksi fitur SIFT
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Inisialisasi list hasil
    tracking_results = []
    filtered_tracks = []
    filtered_detections = []
    keypoints_result = []
    descriptors_result = []
    xyxy = []
    detection_dict = {
        det[2]: det[3]  # { class_name: mask }
        for det in detections
    }

    for track, detection in zip(tracks, detections):
        #class_name = detection[2]  # Nama kelas objek yang terdeteksi

        x, y, w, h = detection[0]
        #mask = None
        xw = x + w
        yh = y + h
        det_bbox = [x, y, xw, yh]
        

        track_id = track.track_id
        class_id = track.det_class
        mask = detection_dict.get(class_id, None)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        track_bbox = [x1, y1, x2, y2]

        if class_id == "Tampak Depan":
            #print("masukkk2")
            # Ekstraksi fitur SIFT hanya untuk 'Tampak Depan'
            kp, des = extract_sift_features(track, gray_frame)
            if kp:
                keypoints_result.extend(kp)
            if des is not None:
                descriptors_result.extend(des.tolist())
            kp = 2
            des = 3
            tracking_results.append({"track_id" : track_id,
                         "class_name" : class_id,
                         "bounding box" : [x1, y1, x2, y2],
                         "kp" : kp,
                         "des" : des,
                         "mask" : mask
                        })
        else:
            tracking_results.append({"track_id" : track_id,
             "class_name" : class_id,
             "bounding box" : [x1, y1, x2, y2],
             "kp" : None,
             "des" : None,
             "mask" : mask
            })
            # Untuk 'Truk' dan 'Tampak Samping', hanya simpan tracks & detections
            keypoints_result = []
            descriptors_result = []

        #label = f'ID: {track_id}'
        #print(f"tracking result {tracking_results}")
        #cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Simpan hasil sesuai kategori
        #filtered_tracks.append((track, detection_without_mask))
        #print(f"filtered tracks {filtered_tracks}")
        #filtered_detections.append(detection_without_mask)
    #()
    #print()
    #print()
    # Cetak hasil deteksi tanpa mask dan keypoints
    #print("\nFiltered Detections (Tanpa Mask):")
    #for det in filtered_detections:
    #    print(det)
#
    #print("\nKeypoints Result:")
    #print(keypoints_result)

    return tracking_results, frame


def draw_tracking_info(frame, tracking_results, is_cam1=True):   
    """Draws tracking information on the frame, including bounding boxes, IDs, and estimated heights, and overlays masks."""
    
    for track in tracking_results:
        track_id = track['track_id']
        x1, y1, x2, y2 = map(int, track['bounding box'])
        mask = track.get('mask')  # Assuming the mask is in the tracking result
        
        if track['class_name'] == "Truk":
            color = (0, 255, 0)  # Green
        elif track['class_name'] == "Tampak Samping":
            color = (0, 255, 255)  # Yellow
        elif track['class_name'] == "Tampak Depan":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 255)  # Default white
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and class name
        label = f'ID: {track_id}'
        cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay mask if available
        if mask is not None and mask.size > 0:
            mask = mask.reshape((-1, 1, 2)).astype(np.int32)  # Reshape mask to contour format
            overlay = frame.copy()
            cv2.fillPoly(overlay, [mask], color)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # Blend mask with the frame
    
    return frame
