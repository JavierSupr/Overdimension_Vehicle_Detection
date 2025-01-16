import asyncio
import websockets
import base64
import json
import cv2

async def video_stream(websocket, frame1, frame2):
    try:
        while True:
            _, buffer1 = cv2.imencode('.jpg', frame1)
            frame1_based64 = base64.b64encode(buffer1).decode('utf-8')
            message1 = json.dumps({"type": "frame", "camera": "Camera 1", "data": frame1_based64})
            print(f"message1{message1}")
            await websocket.send(message1)

            _, buffer2 = cv2.imencode('.jpg', frame2)
            frame2_based64 = base64.b64encode(buffer2).decode('utf-8')
            message2 = json.dumps({"type": "frame", "camera": "Camera 2", "data": frame2_based64})
            await websocket.send(message2)

            await asyncio.sleep(0.03)

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket Connection closed with error: {e}")
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket closed normally")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("WebSocket closed, releasing resources.")
    
async def start_stream_server(frame1, frame2):
    print("try to start")
    async def handler(websocket, path):
        await video_stream(websocket, frame1, frame2)
        print("started")

    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()
    
