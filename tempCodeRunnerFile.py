                mask = results.masks.data[det_idx].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[mask > 0] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

    return annotated_frame

async def process_and_stream_frames(websocket, model, cap1, cap2):
    """
    Combined processing and streaming of frames to ensure fresh frames are always sent
    """
    #try:
    class_names = model.names
    colors = {cls_idx: tuple(np.random.randint(0, 256, 3).tolist()) 
             for cls_idx in class_names}
    while True:
        # Read frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            # Reset videos if they end
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
        # Resize frames
        frame1 = cv2.resize(frame1, (640, 480))