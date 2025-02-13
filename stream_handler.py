import asyncio
import websockets
import base64
import json
import cv2

async def send_frame_via_websocket(websocket, frame, camera_name):
    """Send an encoded frame through WebSocket."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    message = json.dumps({
        "type": "frame",
        "camera": camera_name,
        "data": frame_base64
    })
    await websocket.send(message)

async def start_stream_server(frame1, frame2):
    async def handler(websocket, path):
        await video_stream(websocket, frame1, frame2)

    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()
    
