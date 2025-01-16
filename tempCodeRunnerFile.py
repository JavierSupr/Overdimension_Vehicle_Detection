ed on detection results.
    """
    if results.masks is not None:
        for det_idx, cls in enumerate(results.boxes.cls):
            c