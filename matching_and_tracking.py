import cv2
import numpy as np
import traceback

def list_to_keypoints(kp_list):
    """Convert a list of tuples back into cv2.KeyPoint objects and print them."""
    keypoints = []
    for kp in kp_list:
        if len(kp) == 7:  # Ensure tuple has 7 elements
            x, y, size, angle, response, octave, class_id = kp  # Unpack tuple
            keypoint = cv2.KeyPoint(
                x=float(x), y=float(y), size=float(size), 
                angle=float(angle), response=float(response), 
                octave=int(octave), class_id=int(class_id)
            )
            keypoints.append(keypoint)
        else:
            print(f"Invalid keypoint format {kp} (Expected 7 elements)")
    return keypoints

def update_id_mappings(tracked_objects1, tracked_objects2, good_matches, id_mapping):
    """
    Updates ID mappings by assigning Camera 2 object IDs to the corresponding Camera 1 object IDs.

    Args:
        tracked_objects1 (dict): Tracked objects from Camera 1.
        tracked_objects2 (dict): Tracked objects from Camera 2.
        good_matches (list): List of good matches between keypoints from both cameras.
        id_mapping (dict): Existing ID mappings {Camera2_ID: Camera1_ID}.

    Returns:
        dict: Updated tracked_objects1 with assigned IDs from Camera 2.
    """
    # Convert keypoints from list to OpenCV KeyPoint objects
    keypoints1 = list_to_keypoints(tracked_objects1.get("keypoints", []))
    keypoints2 = list_to_keypoints(tracked_objects2.get("keypoints", []))
    #print(f"panjang kp 1 {len(keypoints1)} panjang kp 2 {len(keypoints2)}")

    # Create a mapping from Camera 2 IDs to Camera 1 IDs
    for match in good_matches:
        kp1_idx = match.queryIdx  # Index of keypoint in Camera 1
        kp2_idx = match.trainIdx  # Index of keypoint in Camera 2
        
        
        id1, id2 = None, None
        
        # Find corresponding track_id from Camera 2 keypoints
        if 0 <= kp2_idx < len(keypoints2):
            kp2_pt = keypoints2[kp2_idx].pt  # Get (x, y) coordinates of matched keypoint
            #print(f"Type of tracked_objects2: {type(tracked_objects2)}")
            #print(f"Contents of tracked_objects2: {tracked_objects2}")
            #print(f"kp1_idx {kp1_idx}")
            #print(f"kp2_pt match {kp2_pt}")
            if isinstance(tracked_objects2, dict):  # Directly access its keys
                obj_data2 = tracked_objects2
                if "bounding_box" in obj_data2:
                    #print(f"objdata2 {obj_data2['bounding_box']}")
                    if is_point_in_bbox(kp2_pt, obj_data2["bounding_box"]):  # Correct key name
                        
                        id2 = obj_data2["track_id"]
                        #break

        # Find corresponding track_id from Camera 1 keypoints
        if 0 <= kp1_idx < len(keypoints1):
            #print("masukkkk")
            kp1_pt = keypoints1[kp1_idx].pt  # Get (x, y) coordinates of matched keypoint
            #print(f"kp1_pt match {kp1_pt}")
            if isinstance(tracked_objects1, dict):  # Directly access its keys
                obj_data1 = tracked_objects1
                if "bounding_box" in obj_data1:
                    #print(f"objdata1 {obj_data1['bounding_box']}")
                    if is_point_in_bbox(kp1_pt, obj_data1["bounding_box"]):  # Correct key name
                        #print(f"obj_data1['track_id'] {obj_data1['track_id']}")
                        id1 = obj_data1["track_id"]
                        #break


        if id1 is not None and id2 is not None:
            #print("yes")
            id_mapping[id2] = id1  # Assign Camera 2 ID to Camera 1 ID
        #print(f"kp1_idx {kp1_idx}, Coordinates: {kp1_pt} | kp2_idx {kp2_idx}, Coordinates: {kp2_pt}")
    # Update tracked_objects1 with IDs from Camera 2
    #print()

    updated_tracked_objects1 = tracked_objects1.copy()  # Copy the original dictionary

    #print(id_mapping)
    # Directly update track_id instead of looping
    if "track_id" in updated_tracked_objects1:
        new_id = updated_tracked_objects1["track_id"]  # Default to existing ID
        #print("masukk")

        for id2, id1 in id_mapping.items():
            #print("masuk2")
            if updated_tracked_objects1["track_id"] == id1:
               # print("masuk3")
                new_id = id2  # Assign Camera 2 ID
                #print(f"Updated Camera 1 ID {id1} to Camera 2 ID {new_id}")
                break

        updated_tracked_objects1["track_id"] = new_id  # Update ID
    kp1_pts = [kp.pt for kp in keypoints1]
    return updated_tracked_objects1, kp1_pts, id_mapping

def is_point_in_bbox(point, bbox):
    """
    Checks if a point is inside a bounding box.

    Args:
        point (tuple): (x, y) coordinates of the point.
        bbox (list): Bounding box in format [x_min, y_min, x_max, y_max].

    Returns:
        bool: True if the point is inside the bounding box, otherwise False.
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max
def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on a frame.
    """
    #print("masukk")
    output_frame = frame.copy()
    for (x, y) in keypoints:
        cv2.circle(output_frame, (int(x), int(y)), 5, color, -1)
    return output_frame

def match_features(tracked_objects1, tracked_objects2, frame):
    """
    Match SIFT features between two frames and update ID mappings.

    Args:
        tracked_objects1 (dict): Tracked objects from first frame.
        tracked_objects2 (dict): Tracked objects from second frame.

    Returns:
        list: List of good matches between frames.
        dict: Updated tracked_objects1 with assigned IDs.
    """
    # {id2: id1}
    good_matches = []
    updated_tracked_objects1 = {}
    id_mapping = {}

    # Ensure tracked_objects1 and tracked_objects2 are dictionaries
    if not isinstance(tracked_objects1, dict) or not isinstance(tracked_objects2, dict):

        print("Error: tracked_objects1 and tracked_objects2 must be dictionaries.")
        return [], tracked_objects1
    descriptors1 = tracked_objects1.get("descriptor")
    descriptors2 = tracked_objects2.get("descriptor")
    keypoints1 = list_to_keypoints(tracked_objects1.get("keypoints", []))
    keypoints2 = list_to_keypoints(tracked_objects2.get("keypoints", []))
    #print(f"panjang kpp 1 {len(keypoints1)} panjang kp 2 {len(keypoints2)}")

    # Ensure descriptors are NumPy arrays and not None
    if isinstance(descriptors1, np.ndarray) and descriptors1.size > 0:
        descriptors1 = np.asarray(descriptors1, dtype=np.float32)
    else:
        descriptors1 = None

    if isinstance(descriptors2, np.ndarray) and descriptors2.size > 0:
        descriptors2 = np.asarray(descriptors2, dtype=np.float32)
    else:
        descriptors2 = None
    #print(f"descriptor1 {descriptors1}")
    #print(f"descriptor2 {descriptors2}")
    if descriptors1 is not None and descriptors2 is not None:
        try:
            #print("masuk3")
            # Ensure descriptors have the same feature vector size
            if descriptors1.shape[1] != descriptors2.shape[1]:
                print(f"Shape mismatch: descriptors1.shape={descriptors1.shape}, descriptors2.shape={descriptors2.shape}")
                return [], tracked_objects1
            #print("masuk4")
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            #print("masuk5")
            for match_pair in matches:
                #print("masuk6")
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            #print(f"(good_matches) {good_matches}")
            #print("masuk7")
            updated_tracked_objects1, kp1_pts, id_mapping = update_id_mappings(tracked_objects1, tracked_objects2, good_matches, id_mapping)
            #frame = draw_keypoints(frame, kp1_pts, (0, 255, 0))
            #print(f"id mapping {id_mapping}")
        except Exception as e:
            print(f"Error in feature matching: {e}")
            traceback.print_exc() 
            return [], tracked_objects1
    #print("masuk8")
    return good_matches, updated_tracked_objects1, id_mapping




def extract_sift_features(sift, track, gray_frame):
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
    frame_h, frame_w = gray_frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

    # Extract ROI from grayscale frame
    roi = gray_frame[y1:y2, x1:x2]

    if roi.size == 0:
        return [], None  # Return empty if ROI is invalid

    # Initialize SIFT

    # Detect and compute keypoints & descriptors
    keypoints, descriptors = sift.detectAndCompute(roi, None)

    # **Adjust keypoint coordinates to frame coordinates**
    for kp in keypoints:
        kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)  # Add offset to get global position

    return keypoints, descriptors

def process_tracks_and_extract_features(deepsort, sift, detections, frame, is_cam1=True):
    
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
        if track.is_confirmed() and track.time_since_update <= 1:
            base_id = int(track.track_id)  # Convert track_id to integer

            # Ensure odd IDs for cam1, even IDs for cam2
            if is_cam1:
                unique_id = base_id * 2 - 1  # 1, 3, 5, 7...
            else:
                unique_id = base_id * 2  # 2, 4, 6, 8...

            x, y, w, h = detection[0]
            xw = x + w
            yh = y + h
            
            class_id = track.det_class
            mask = detection_dict.get(class_id, None)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            track_bbox = [x1, y1, x2, y2]

            if class_id == "Tampak Depan":
                # Extract SIFT features for 'Tampak Depan'
                kp, des = extract_sift_features(sift, track, gray_frame)
                if kp:
                    keypoints_result.extend(kp)
                if des is not None:
                    descriptors_result.extend(des.tolist())

                tracking_results.append({
                    "track_id": unique_id,
                    "class_name": class_id,
                    "bounding box": [x1, y1, x2, y2],
                    "kp": kp,
                    "des": des,
                    "mask": mask
                })
            else:
                tracking_results.append({
                    "track_id": unique_id,
                    "class_name": class_id,
                    "bounding box": [x1, y1, x2, y2],
                    "kp": None,
                    "des": None,
                    "mask": mask
                })            # Untuk 'Truk' dan 'Tampak Samping', hanya simpan tracks & detections

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
