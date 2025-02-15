import asyncio
import websockets
import cv2

SERVER_URI = "ws://localhost:8765"

async def send_video():
    async with websockets.connect(SERVER_URI) as websocket:
        cap = cv2.VideoCapture(0)  # Open webcam
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)

            # Send frame over WebSocket
            await websocket.send(buffer.tobytes())

        cap.release()

if __name__ == "__main__":
    asyncio.run(send_video())
