def calculate_vehicle_height(bbox_height_pixel, distance, focal_length):
    """Calculate estimated vehicle height based on pixel height, distance, and focal length."""
    return (bbox_height_pixel * distance) / focal_length

def calculate_distance(vehicle_width_real, bbox_width_pixel, focal_length):
    """Calculate estimated distance to the vehicle."""
    return (vehicle_width_real * focal_length) / bbox_width_pixel
