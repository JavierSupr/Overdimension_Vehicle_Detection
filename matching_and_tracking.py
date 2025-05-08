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
    #print(f"len good matches {len(good_matches)} - obj 1 {tracked_objects1['track_id']} - obj 2 {tracked_objects2['track_id']}")
    # Create a mapping from Camera 2 IDs to Camera 1 IDs
    for match in good_matches:
        kp1_idx = match.queryIdx  # Index of keypoint in Camera 1
        kp2_idx = match.trainIdx  # Index of keypoint in Camera 
        #print(f"kp1 {kp1_idx}")
        #print(f"kp2 {kp2_idx}")
        
        
        id1, id2 = None, None
        
        # Find corresponding track_id from Camera 2 keypoints
        if 0 <= kp2_idx < len(keypoints2):
            kp2_pt = keypoints2[kp2_idx].pt  # Get (x, y) coordinates of matched keypoint
            #print(f"Type of tracked_objects2: {type(tracked_objects2)}")
            #print(f"Contents of tracked_objects2: {tracked_objects2}")
            #print(f"kp1_idx {kp1_idx}"
            #print(f"kp2_pt match {kp2_pt} - track_id {tracked_objects2['track_id']}")
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
            #print(f"kp1_pt match {kp1_pt} - track_id {tracked_objects1['track_id']}")
            if isinstance(tracked_objects1, dict):  # Directly access its keys
                obj_data1 = tracked_objects1
                if "bounding_box" in obj_data1:
                    #print(f"objdata1 {obj_data1['bounding_box']}")
                    if is_point_in_bbox(kp1_pt, obj_data1["bounding_box"]):  # Correct key name
                        #print(f"obj_data1['track_id'] {obj_data1['track_id']}")
                        #print(f"kp1_pt match after {kp1_pt} - {obj_data1['track_id']}")

                        id1 = obj_data1["track_id"]
                        #break


        if id1 is not None and id2 is not None:
            #print("yes")
            id_mapping[id2] = id1  # Assign Camera 2 ID to Camera 1 ID
            #print(f"id {id1} diubah ke {id2}")
        #print(f"kp1_idx {kp1_idx}, Coordinates: {kp1_pt} | kp2_idx {kp2_idx}, Coordinates: {kp2_pt}")
    # Update tracked_objects1 with IDs from Camera 2


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
    #print()
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

def match_features(tracked_objects1_list, tracked_objects2_list, frame, id_mapping):
    """
    Match features between multiple tracked objects from two frames.

    Args:
        tracked_objects1_list (list): List of tracked objects from camera 1.
        tracked_objects2_list (list): List of tracked objects from camera 2.
        frame (np.ndarray): Current frame (optional for visualization).

    Returns:
        list: All good matches.
        list: List of updated tracked_objects1 with updated IDs.
        dict: Mapping from Camera2 IDs to Camera1 IDs.
    """
    all_good_matches = []
    updated_tracked_objects1_list = []
    for obj1 in tracked_objects1_list:
        for obj2 in tracked_objects2_list:
            descriptors1 = obj1.get("descriptor")
            descriptors2 = obj2.get("descriptor")
            keypoints1 = list_to_keypoints(obj1.get("keypoints", []))
            keypoints2 = list_to_keypoints(obj2.get("keypoints", []))

            # Validate descriptors
            if not (isinstance(descriptors1, np.ndarray) and descriptors1.size > 0):
                continue
            if not (isinstance(descriptors2, np.ndarray) and descriptors2.size > 0):
                continue

            descriptors1 = np.asarray(descriptors1, dtype=np.float32)
            descriptors2 = np.asarray(descriptors2, dtype=np.float32)
            
            if descriptors1.shape[1] != descriptors2.shape[1]:
                print(f"Shape mismatch: {descriptors1.shape} vs {descriptors2.shape}")
                continue

            try:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descriptors1, descriptors2, k=2)

                good_matches = []
                used_train_indices = set()

                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance and m.trainIdx not in used_train_indices:
                            good_matches.append(m)
                            used_train_indices.add(m.trainIdx)
                #print(f"Jumlah Pasangan Cocock {len(good_matches)} - ID Cam 1 {obj1.get('track_id')} - ID Cam 2 {obj2.get('track_id')}")
                if len(good_matches) >= 10:
                    all_good_matches.extend(good_matches)
                    updated_obj1, kp1_pts, id_mapping = update_id_mappings(obj1, obj2, good_matches, id_mapping)
                    updated_tracked_objects1_list.append(updated_obj1)

            except Exception as e:
                print(f"Error in feature matching between objects: {e}")
                traceback.print_exc()
                continue
    
    #print()
    #print()
    return all_good_matches, updated_tracked_objects1_list, id_mapping

#def match_features(tracked_objects1, tracked_objects2, frame):
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
            #print(f"panjang match{len(matches)}")
            #print("masuk5")
            used_train_indices = set() 
            for match_pair in matches:
                #print("masuk6")
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        #print(f"m = {m.distance} n = {n.distance}")
                        if m.trainIdx not in used_train_indices:
                            good_matches.append(m)
                            used_train_indices.add(m.trainIdx)
                        #print(f"good matches {len(good_matches)}- {good_matches}")
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

def calculate_iou(bbox1, bbox2):
    # Menghitung Intersection over Union (IoU) antara dua bounding box
    x1, y1, x2, y2 = bbox1
    x1_prime, y1_prime, x2_prime, y2_prime = bbox2
    
    # Hitung area intersection
    inter_x1 = max(x1, x1_prime)
    inter_y1 = max(y1, y1_prime)
    inter_x2 = min(x2, x2_prime)
    inter_y2 = min(y2, y2_prime)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        intersection_area = 0  # Tidak ada intersection

    # Hitung area union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_prime - x1_prime) * (y2_prime - y1_prime)
    
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area != 0 else 0

def process_tracks_and_extract_features(deepsort, sift, detections, frame, is_cam1=True):
    
    detections2 = [det[:3] for det in detections]
    tracks = deepsort.update_tracks(detections2, frame=frame)
    frame_height, frame_width = frame.shape[:2]
    # Konversi frame ke grayscale untuk ekstraksi fitur SIFT
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Inisialisasi list hasil
    tracking_results = []
    keypoints_result = []
    descriptors_result = []
    detection_dict = {}     
    for det in detections:
        class_name = det[2]  # kelas objek
        mask = det[3]        # mask objek
        bbox = det[0]        # bounding box

        # Jika kelas sudah ada dalam dictionary, tambahkan deteksi baru ke list
        if class_name in detection_dict:
            detection_dict[class_name].append({'mask': mask, 'bbox': bbox})
        else:
            # Jika kelas belum ada, buat list baru untuk kelas ini
            detection_dict[class_name] = [{'mask': mask, 'bbox': bbox}]

    #print(f"detection dict {detection_dict}")
    for track in tracks:
            #if not track.is_confirmed():
            #    print(f"Track {track.track_id} belum dikonfirmasi")
            #if track.time_since_update > 1:
            #    print(f"Track {track.track_id} {track.time_since_update}")
            if track.is_confirmed() :#and track.time_since_update <= 1:
                base_id = int(track.track_id)  # Convert track_id to integer

                # Ensure odd IDs for cam1, even IDs for cam2
                if is_cam1:
                    unique_id = base_id * 2 - 1  # 1, 3, 5, 7...
                else:
                    unique_id = base_id * 2  # 2, 4, 6, 8...

                class_id = track.det_class
                class_detections = detection_dict.get(class_id, [])
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                x1 = max(0, min(x1, frame_width - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                y2 = max(0, min(y2, frame_height - 1))

                for det in class_detections:
                    det_bbox = det['bbox']
                    x = det_bbox[0]
                    y = det_bbox[1]
                    w = det_bbox[2]
                    h = det_bbox[3]
                    xw = x + w
                    yh = y + h
                    matching_mask = det['mask']
                    # Menghitung IoU untuk perbandingan
                    iou = calculate_iou([x1, y1, x2, y2], [x, y, xw, yh])
                    print(f"iou {iou}")
                    if iou >= 0.8 and class_id in detection_dict :
                        mask = matching_mask
                        break  # Jika menemukan match, berhenti mencari lebih lanjut

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
    print(f"tracking result {len(tracking_results)}")

    return tracking_results, frame


def draw_tracking_info(frame, tracking_results, is_cam1=True):   
    """Draws tracking information on the frame, including bounding boxes, IDs, and estimated heights, and overlays masks."""

    height, width = frame.shape[:2]
    line_x1 = int((1 / 2) * width)  # x-coordinate at 2/5 of frame width
    line_x2 = int((7 / 8) * width)  # x-coordinate at 2/5 of frame width
    # Draw the vertical blue line
    #cv2.line(frame, (line_x1, 0), (line_x1, height), (255, 0, 0), 2)  # Blue color (BGR)
    #cv2.line(frame, (line_x2, 0), (line_x2, height), (255, 0, 0), 2)  # Blue color (BGR)
    count_tampak_depan = 0
    count_truk = 0
    seg2 = []
    for result in tracking_results:
        seg = result['mask'] 
        class_name = result['class_name']

        if class_name == "Tampak Depan" and seg is not None:
            count_tampak_depan += 1
            seg2.append(seg)

        #sprint(f"Total kelas 'Tampak Depan': {count_tampak_depan} - {seg}")
    for track in tracking_results:
        track_id = track['track_id']
        x1, y1, x2, y2 = map(int, track['bounding box'])
        mask = track['mask']  # Assuming the mask is in the tracking result
        
        if track['class_name'] == "Truk Besar" or track['class_name'] == "Truk Kecil":
            color = (0, 255, 0)  # Green
        #elif track['class_name'] == "Truk Kecil":
        #    color = (255, 255, 0)  # Green
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
        if mask is not None and len(mask) > 0:
            mask = np.array(mask, dtype=np.int32)  # Convert list to np.array
            mask = mask.reshape((-1, 1, 2))         # Bentuk ke format kontur OpenCV
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [mask], color)    # Warnai polygon di overlay
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # Blend overlay ke frame
    return frame
