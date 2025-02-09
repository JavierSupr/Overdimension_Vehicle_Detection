import cv2
import socket
import struct
import numpy as np
import queue

PORT_1 = 5005
PORT_2 = 5006
BUFFER_SIZE = 65536
running = True

frame_queues = {
    "Camera 1": queue.Queue(maxsize=1),
    "Camera 2": queue.Queue(maxsize=1),
}

async def receive_video(sock):
    """Receive and yield video frames over UDP asynchronously."""
    global running
    while running:
        try:
            # Receive number of chunks
            data, _ = sock.recvfrom(BUFFER_SIZE)
            if len(data) == 1:
                chunks_count = struct.unpack("B", data)[0]
            else:
                continue  # Skip invalid packets

            # Receive the chunks
            chunks = []
            for _ in range(chunks_count):
                chunk, _ = sock.recvfrom(BUFFER_SIZE)
                chunks.append(chunk)

            # Combine chunks and decode frame
            frame_data = b"".join(chunks)
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow('frame', frame)
                yield frame  # Yield the received frame

        except Exception as e:
            print(f"[ERROR] Receiving video failed: {e}")
            running = False
            break