def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    

    # Convert to x_min, y_min, x_max, y_max format
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union

def find_root(id_group, track_id):
    """Find the root ID for a given track ID with path compression."""
    if track_id not in id_group:
        return track_id
    
    # Path compression: Update all nodes in the path to point directly to the root
    if id_group[track_id] != track_id:
        id_group[track_id] = find_root(id_group, id_group[track_id])
    return id_group[track_id]

def union_groups(id_group, id1, id2):
    """Merge two groups by their root IDs."""
    root1 = find_root(id_group, id1)
    root2 = find_root(id_group, id2)
    if root1 != root2:  # Only merge if they're not already in the same group
        # Always use the smaller root ID as the representative
        if root1 < root2:
            id_group[root2] = root1
        else:
            id_group[root1] = root2

def find_overlapping_groups(tracks, iou_threshold):
    """Find groups of overlapping objects using a graph-based approach."""
    from collections import defaultdict
    
    # Initialize adjacency list to store overlapping relationships
    overlap_graph = defaultdict(set)
    id_group = {track.track_id: track.track_id for track in tracks}

    # Build graph of overlapping objects
    for i, track in enumerate(tracks):
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        bbox1 = track.to_ltwh()
        
        for j in range(i + 1, len(tracks)):
            other_track = tracks[j]
            if not other_track.is_confirmed() or other_track.time_since_update > 1:
                continue

            bbox2 = other_track.to_ltwh()
            iou = calculate_iou(bbox1, bbox2)

            if iou > iou_threshold:
                # Add bidirectional edges
                overlap_graph[track.track_id].add(other_track.track_id)
                overlap_graph[other_track.track_id].add(track.track_id)

    return overlap_graph, id_group

def merge_track_ids(tracks, iou_threshold):
    """Merge all overlapping tracks transitively."""
    # Find all overlapping relationships
    overlap_graph, id_group = find_overlapping_groups(tracks, iou_threshold)
    
    # Process each connected component in the graph
    processed = set()
    for track_id in overlap_graph:
        if track_id in processed:
            continue
            
        # Use DFS to find all connected tracks
        stack = [track_id]
        group = set()
        
        while stack:
            current_id = stack.pop()
            if current_id not in group:
                group.add(current_id)
                stack.extend(overlap_graph[current_id] - group)
        
        # Merge all tracks in the connected component
        if len(group) > 1:
            min_id = min(group)  # Use the smallest ID as the representative
            for other_id in group:
                if other_id != min_id:
                    union_groups(id_group, min_id, other_id)
        
        processed.update(group)

    return {track_id: find_root(id_group, track_id) for track_id in id_group}
