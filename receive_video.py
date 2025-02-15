import asyncio
import websockets
import cv2
import numpy as np

SERVER_URI = "ws://localhost:8765"

async def receive_video():
    async with websockets.connect(SERVER_URI) as websocket:
        while True:
            try:
                # Receive frame
                message = await websocket.recv()

                # Convert bytes to NumPy array
                np_data = np.frombuffer(message, dtype=np.uint8)

                # Decode image
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    cv2.imshow("Received Video Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(receive_video())
